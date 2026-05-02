# DeepONet for 3D Terzaghi Consolidation

A DeepONet surrogate model for the 3D Terzaghi consolidation PDE, capable of predicting excess pore water pressure (PWP) fields given arbitrary initial conditions and consolidation coefficients. The repository also includes a POD-based ROM baseline and a Monte Carlo uncertainty quantification (UQ) pipeline.

**Paper:** [DOI 10.1007/s13369-025-10602-2](https://doi.org/10.1007/s13369-025-10602-2)

---

## Repository Structure

```
datagen/        Data generation scripts (DeepONet and ROM training data)
solver/         Batch 3D FDM solver for the Terzaghi consolidation PDE
train/          DeepONet model definitions and training pipeline (Hydra config)
  conf/         Training configuration YAML files
  models/       Branch and trunk network architectures
rom/            POD-based ROM: basis computation, training, and evaluation
  conf/         ROM training configuration
eval/           Evaluation scripts (full-field comparison, MSE stats, runtime benchmark)
uq/             Monte Carlo UQ simulation and postprocessing
data/           Large data files — populated by extracting the archive (see below)
test_models.py  Smoke test: verifies all model variants can do a forward pass
MODEL_CONFIGS.md  Guide for selecting DeepONet architecture via config flags
```

---

## Data and Model Archive

Training data, POD basis, and trained model weights are stored in a separate archive due to file size (~3.95 GB total). Download the archive and extract it in the repository root:

```bash
# Download (replace URL with actual storage link)
wget <ARCHIVE_URL> -O archive.tar.gz

# Extract — files unpack directly into the correct locations
tar -xzf archive.tar.gz
```

The archive contains:

| Path after extraction | Description | Size |
|---|---|---|
| `data/rom_fields.h5` | Full-field FDM solutions for ROM (150 samples × 51⁴ grid) | 3.80 GB |
| `data/train.h5` | DeepONet training data (sampled space-time points) | 14 MB |
| `data/val.h5` | DeepONet validation data | 3.4 MB |
| `data/basis.npz` | POD basis (50 modes) computed from `rom_fields.h5` | 28 MB |
| `train/model/*/latest.pt` | Trained DeepONet weights for all 7 experiment cases | ~11 MB each |
| `train/model/*/config.yaml` | Training config snapshot for each case | — |
| `rom/rom/checkpoint/latest.pt` | Trained ROM regressor weights | 12 MB |
| `rom/rom/checkpoint/config.yaml` | ROM training config snapshot | — |

---

## Installation

Python 3.10 or later is required. Install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is listed without a CUDA suffix. To enable GPU training, install the appropriate CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

---

## Usage

All scripts are run from the **project root** directory.

### 1. (Optional) Regenerate DeepONet training data

Edit `datagen/data_generator.py` to adjust `n_samples`, `seed`, or boundary conditions, then run:

```bash
python datagen/data_generator.py
# Output: data/train.h5 (train + val split handled internally)
```

### 2. (Optional) Regenerate ROM full-field data

```bash
python datagen/data_generator_rom.py
# Output: data/rom_fields.h5  (~3.8 GB, 150 samples by default)
```

### 3. Train DeepONet

Configuration is managed via Hydra. Edit `train/conf/config.yaml` (or use one of the variant configs in `train/conf/`) to select the model architecture. See `MODEL_CONFIGS.md` for a full guide.

```bash
python -m train.train
# Checkpoints saved to: train/model/<run_name>/
```

To use a specific config:

```bash
python -m train.train --config-name config_vanilla_ff
```

### 4. Evaluate DeepONet

```bash
# Full-field comparison against the reference FDM solver
python eval/evaluate_full_field.py

# Time-specific slice evaluation
python eval/evaluate_time_specific.py

# MSE statistics across train/val/test sets
python eval/measure_mse_stats.py

# Runtime benchmark (DeepONet vs FDM solver)
python eval/compare_runtime.py
```

Edit the `case` variable at the top of each script to select a trained model.

### 5. ROM pipeline

```bash
# Step 1: Compute POD basis from full-field data
python rom/compute_basis.py
# Output: data/basis.npz

# Step 2: Train ROM regressor
python -m rom.train_rom
# Checkpoints saved to: rom/rom/checkpoint/

# Step 3: Evaluate ROM vs FDM solver
python rom/evaluate_rom.py
```

### 6. Uncertainty Quantification (UQ)

```bash
# Step 1: Monte Carlo simulation (mode: "deeponet" or "solver")
python uq/simulate_uq_consolidation.py
# Output: uq/<case>/<mode>/Uv_timeseries*.npz

# Step 2: Postprocess and plot
python uq/postprocess_uq_consolidation.py
# Output: uq/<case>/<mode>/post/  (CSV summaries, percentile plots)
```

---

## Trained Model Cases

| Case name | Description |
|---|---|
| `case3_data_v2_vanilla_ff_scaling` | Vanilla MLP + Fourier features + scaling (primary result) |
| `case3_data_v2_vanilla_ff` | Vanilla MLP + Fourier features |
| `case3_data_v3_vanilla_ff_scaling` | Vanilla MLP + FF scaling, updated BC dataset |
| `case3_vanilla_ff` | Vanilla MLP + Fourier features (v1 data) |
| `case3_vanilla_ff_scaling` | Vanilla MLP + FF scaling (v1 data) |
| `case3_vanilla_ff_sig02` | Vanilla MLP + FF, σ=0.2 |
| `case3_vanilla_lr1e4` | Vanilla MLP, lr=1e-4 |

---

## Citation

If you use this codebase, please cite:

* Choi, Y., Liu, C., & Macedo, J. (2026). Operator learning for consolidation: An architectural comparison for DeepONet variants. *Computers and Geotechnics, 194*, 108017. [https://doi.org/10.1016/j.compgeo.2026.108017](https://doi.org/10.1016/j.compgeo.2026.108017)
