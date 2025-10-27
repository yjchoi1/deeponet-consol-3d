# DeepONet Model Configuration Guide

This document explains how to select and configure different DeepONet architectures via the config.yaml file.

## Available Models

### 1. ResNet Branch + Fourier Features (Default)
**Current baseline model with residual blocks and Fourier feature encoding**

```yaml
model:
  use_conv_branch: false
  use_fourier_features: true
  branch:
    input_dim: 2601  # 51 * 51
    hidden_dim: 256
    output_dim: 128
    num_blocks: 3

data:
  flatten_branch: true  # Must be true for ResNet
```

### 2. Conv2d Branch + Fourier Features
**Convolutional branch network with Fourier feature trunk**

```yaml
model:
  use_conv_branch: true
  use_fourier_features: true
  branch:
    output_dim: 128  # Only output_dim is used; input_dim and hidden_dim ignored
    # Conv architecture: 1→32→64→128 channels

data:
  flatten_branch: false  # Must be false for Conv2d
```

### 3. Conv2d Branch + Standard Trunk (No Fourier Features)
**Convolutional branch network with standard MLP trunk**

```yaml
model:
  use_conv_branch: true
  use_fourier_features: false
  branch:
    output_dim: 128

data:
  flatten_branch: false  # Must be false for Conv2d
```

### 4. Vanilla DeepONet
**Simple MLP networks without residual blocks or Fourier features**

```yaml
model:
  use_conv_branch: false
  use_fourier_features: false
  branch:
    input_dim: 2601
    hidden_dim: 256
    output_dim: 128
    num_blocks: 3  # Becomes num_layers in vanilla

data:
  flatten_branch: true  # Must be true for vanilla
```

## Architecture Details

### ResNet Branch
- Initial linear layer: input_dim → hidden_dim
- N residual blocks (each with 2 linear layers + skip connection)
- Final linear layer: hidden_dim → output_dim

### Conv2d Branch
- Conv1: 1→32 channels, 3×3 kernel, padding=1, ReLU, MaxPool(2)
- Conv2: 32→64 channels, 3×3 kernel, padding=1, ReLU, MaxPool(2)
- Conv3: 64→128 channels, 3×3 kernel, padding=1, ReLU, MaxPool(2)
- Flatten: (batch, 128, 6, 6) → (batch, 4608)
- Linear: 4608 → output_dim
- Input shape: (batch, 1, 51, 51)

### Trunk with Fourier Features
- Random Fourier feature projection: (cv, t, x, y, z) → sin/cos features
- ResNet: Similar to branch with residual blocks
- Vanilla: Simple MLP layers

### Trunk without Fourier Features
- Direct input: (cv, t, x, y, z)
- Simple MLP layers with ReLU activations

## Important Notes

1. **Data Loader Compatibility**:
   - Conv2d models require `flatten_branch: false` to preserve 2D structure
   - ResNet and Vanilla models require `flatten_branch: true` for 1D input

2. **Input Dimensions**:
   - For flatten_branch=true: input is (batch, 2601)
   - For flatten_branch=false: input is (batch, 51, 51) → reshaped to (batch, 1, 51, 51) in Conv model

3. **Backward Compatibility**:
   - Old checkpoints from train/deeponet.py are NOT compatible
   - Retrain models with the new architecture

4. **Evaluation**:
   - The evaluate.py script automatically uses the model type from the training config
   - No changes needed to evaluation code when switching models

