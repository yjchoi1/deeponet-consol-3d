"""Simple test script to verify all model variants can train."""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from train.models import build_model


def create_dummy_batch(batch_size: int, flatten_branch: bool):
    """Create a dummy batch for testing."""
    if flatten_branch:
        u = torch.randn(batch_size, 2601)  # 51 * 51 flattened
    else:
        u = torch.randn(batch_size, 51, 51)  # 2D input for conv
    
    cv = torch.randn(batch_size, 1)
    coord = torch.randn(batch_size, 4)  # t, x, y, z
    target = torch.randn(batch_size, 1)
    
    return u, cv, coord, target


def test_model(model_name: str, use_conv: bool, use_ff: bool, num_steps: int = 5):
    """Test a model variant for a few training steps."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"  use_conv_branch={use_conv}, use_fourier_features={use_ff}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model config
    config = {
        "use_conv_branch": use_conv,
        "use_fourier_features": use_ff,
        "branch": {
            "input_dim": 2601,
            "hidden_dim": 128,
            "output_dim": 64,
            "num_blocks": 2,
        },
        "trunk": {
            "hidden_dim": 128,
            "output_dim": 64,
            "num_blocks": 2,
        },
        "ff_features": 64,
        "ff_sigma": 1.0,
        "dtype": "float32",
        "device": device,
    }
    
    # Build model
    try:
        model = build_model(config)
        print(f"✓ Model built successfully")
    except Exception as e:
        print(f"✗ Model build failed: {e}")
        return False
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Determine flatten_branch for data
    flatten_branch = not use_conv
    
    # Run a few training steps
    model.train()
    for step in range(num_steps):
        try:
            # Create dummy batch
            u, cv, coord, target = create_dummy_batch(32, flatten_branch)
            u = u.to(device)
            cv = cv.to(device)
            coord = coord.to(device)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(u, cv, coord)
            
            # Check output shape
            assert output.shape == target.shape, f"Output shape {output.shape} != target shape {target.shape}"
            
            # Backward pass
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if step == 0:
                print(f"  Step {step+1}: loss={loss.item():.6f}, output_shape={output.shape}")
        except Exception as e:
            print(f"✗ Training step {step+1} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"✓ All {num_steps} training steps completed successfully")
    print(f"  Final loss: {loss.item():.6f}")
    
    # Test evaluation mode
    model.eval()
    with torch.no_grad():
        try:
            u, cv, coord, target = create_dummy_batch(16, flatten_branch)
            u = u.to(device)
            cv = cv.to(device)
            coord = coord.to(device)
            output = model(u, cv, coord)
            print(f"✓ Evaluation mode works, output_shape={output.shape}")
        except Exception as e:
            print(f"✗ Evaluation failed: {e}")
            return False
    
    return True


def main():
    """Run tests for all model variants."""
    print("\n" + "="*60)
    print("DeepONet Model Variant Tests")
    print("="*60)
    
    test_configs = [
        ("ResNet + Fourier Features", False, True),
        ("Conv2d + Fourier Features", True, True),
        ("Conv2d + No Fourier Features", True, False),
        ("Vanilla (No ResNet, No FF)", False, False),
    ]
    
    results = {}
    for name, use_conv, use_ff in test_configs:
        success = test_model(name, use_conv, use_ff, num_steps=5)
        results[name] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

