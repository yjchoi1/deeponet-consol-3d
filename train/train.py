from __future__ import annotations

import sys
from pathlib import Path
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import DeepONetPointDataset
from deeponet import build_model

CONFIG: Dict[str, object] = {
    "data": {
        "batch_size": 2048,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
        "flatten_branch": True,
        "dtype": "float32",
        "train": {
            "data_path": Path("train/data/deeponet_terzaghi_train.h5"),
            "seed": 42,
        },
        "val": {
            "data_path": Path("train/data/deeponet_terzaghi_val.h5"),
            "seed": 43,
        },
    },
    "model": {},
    "training": {
        "epochs": 50,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-4,
        "checkpoint_dir": Path("train/checkpoints"),
        "log_dir": Path("train/runs/deeponet"),
        "resume": False,
        "print_every": 1,
        "checkpoint_interval": 1,
        "grad_clip_norm": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def build_dataloaders(cfg: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])

    shared_dataset_options = {
        "flatten_branch": data_cfg["flatten_branch"],
        "dtype": data_cfg["dtype"],
    }

    train_dataset = DeepONetPointDataset(
        {
            **shared_dataset_options,
            "data_path": data_cfg["train"]["data_path"],
            "seed": int(data_cfg["train"]["seed"]),
        }
    )
    val_dataset = DeepONetPointDataset(
        {
            **shared_dataset_options,
            "data_path": data_cfg["val"]["data_path"],
            "seed": int(data_cfg["val"]["seed"]),
        }
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
        drop_last=bool(data_cfg["drop_last"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=bool(data_cfg["pin_memory"]),
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
    if train_cfg["resume"]:
        latest_ckpt = get_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            start_epoch, global_step = load_checkpoint(latest_ckpt, model, optimizer, device)
            start_epoch += 1

    epochs = int(train_cfg["epochs"])
    print_every = int(train_cfg["print_every"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])
    grad_clip_norm_cfg = train_cfg["grad_clip_norm"]
    grad_clip_norm = float(grad_clip_norm_cfg) if grad_clip_norm_cfg is not None else None

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
