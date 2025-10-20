from __future__ import annotations

import sys
from pathlib import Path
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from data_loader import DeepONetPointDataset
from deeponet import build_model

CONFIG: Dict[str, object] = {
    "data": {
        "data_path": Path("train/data/deeponet_terzaghi.h5"),
        "batch_size": 2048,
        "flatten_branch": True,
        "dtype": "float32",
        "seed": 42,
        "train_samples_per_epoch": 200_000,
        "val_samples_per_epoch": 40_000,
        "val_fraction": 0.1,
    },
    "model": {},
    "training": {
        "epochs": 50,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "checkpoint_dir": Path("train/checkpoints"),
        "log_dir": Path("train/runs/deeponet"),
        "resume": True,
        "print_every": 50,
        "checkpoint_interval": 1,
        "grad_clip_norm": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


class SplitPointDataset(DeepONetPointDataset):
    """Dataset restricted to a subset of solution fields."""

    def __init__(
        self,
        cfg: Dict[str, object],
        solution_indices: Iterable[int],
        *,
        samples_per_epoch: Optional[int] = None,
        seed_offset: int = 0,
    ) -> None:
        cfg_local = cfg.copy()
        if samples_per_epoch is not None:
            cfg_local["samples_per_epoch"] = samples_per_epoch
        if cfg_local.get("seed") is not None:
            cfg_local["seed"] = int(cfg_local["seed"]) + seed_offset

        super().__init__(cfg_local)
        indices = np.asarray(list(solution_indices), dtype=np.int32)
        self.solution_indices = indices
        self.num_solutions = int(indices.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.base_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.base_seed + index)

        local_idx = int(rng.integers(0, self.num_solutions))
        solution_idx = int(self.solution_indices[local_idx])
        point_idx = int(rng.integers(0, self.points_per_solution))

        u0 = np.asarray(self.u_data[solution_idx], dtype=np.float32)
        coord = np.asarray(self.coord_data[solution_idx, point_idx], dtype=np.float32)
        target = float(self.target_data[solution_idx, point_idx])
        cv_scalar = float(self.cv_data[solution_idx])

        return self._prepare_sample(u0, coord, cv_scalar, target)


def split_solution_indices(
    data_path: Path, val_fraction: float, seed: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(data_path, "r") as file:
        total_solutions = int(file["u"].shape[0])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_solutions)
    val_size = max(1, int(math.ceil(total_solutions * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_dataloaders(cfg: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    data_path = Path(data_cfg["data_path"])  # type: ignore[arg-type]
    batch_size = int(data_cfg["batch_size"])
    seed = data_cfg.get("seed")
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    
    # You can optionally control the number of samples per epoch for training and validation.
    # If not specified, we use all available samples for each epoch. `DeepONetPointDataset` will handle this.
    train_samples_per_epoch = data_cfg.get("train_samples_per_epoch")
    val_samples_per_epoch = data_cfg.get("val_samples_per_epoch")
    
    train_indices, val_indices = split_solution_indices(data_path, val_fraction, seed)

    train_dataset = SplitPointDataset(
        data_cfg,
        train_indices,
        samples_per_epoch=train_samples_per_epoch,
        seed_offset=0,
    )
    val_dataset = SplitPointDataset(
        data_cfg,
        val_indices,
        samples_per_epoch=val_samples_per_epoch,
        seed_offset=10_000,
    )

    num_workers = int(data_cfg.get("num_workers", 0) or 0)
    pin_memory = bool(data_cfg.get("pin_memory", True))
    drop_last = bool(data_cfg.get("drop_last", False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    checkpoints = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda path: int(path.stem.split("_")[-1]),
    )


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, object],
    global_step: int,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(state, ckpt_path)
    torch.save(state, ckpt_dir / "latest.pt")


def load_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", epoch))
    return epoch, global_step


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float],
) -> Tuple[float, int]:
    model.train()
    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        u = batch["u"].to(device)
        cv = batch["cv"].to(device)
        coord = batch["coord"].to(device)
        target = batch["s"].to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(u, cv, coord)
        loss = criterion(output, target)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = target.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    return avg_loss, total_samples


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, int]:
    model.eval()
    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        u = batch["u"].to(device)
        cv = batch["cv"].to(device)
        coord = batch["coord"].to(device)
        target = batch["s"].to(device)

        output = model(u, cv, coord)
        loss = criterion(output, target)

        batch_size = target.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    return avg_loss, total_samples


def main() -> None:
    cfg = CONFIG.copy()
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device(train_cfg["device"])

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model({"device": device, **model_cfg}).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    log_dir = Path(train_cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    global_step = 0
    if train_cfg.get("resume", True):
        latest_ckpt = get_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer, device)
            start_epoch += 1

    epochs = int(train_cfg["epochs"])
    print_every = int(train_cfg.get("print_every", 50))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 1))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm else None

    history = []

    for epoch in range(start_epoch, epochs):
        train_loss, train_samples = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm,
        )
        global_step += len(train_loader)

        val_loss, val_samples = evaluate(model, val_loader, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("samples/train", train_samples, epoch)
        writer.add_scalar("samples/val", val_samples, epoch)

        if ((epoch + 1) % checkpoint_interval == 0) or ((epoch + 1) == epochs):
            save_checkpoint(ckpt_dir, epoch, model, optimizer, cfg, global_step)

        history.append((epoch + 1, train_loss, val_loss))

        if (epoch + 1) % print_every == 0 or (epoch + 1) == epochs:
            print(
                f"Epoch {epoch+1:04d}/{epochs:04d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

    print("Training complete.")
    print("Epoch\tTrainLoss\tValLoss")
    for entry in history:
        epoch_idx, train_loss, val_loss = entry
        print(f"{epoch_idx:04d}\t{train_loss:.6f}\t{val_loss:.6f}")

    writer.close()


if __name__ == "__main__":
    main()
