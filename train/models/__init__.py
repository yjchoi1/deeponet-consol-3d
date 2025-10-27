from __future__ import annotations

from typing import Mapping

import torch.nn as nn


def build_model(cfg: Mapping[str, object]) -> nn.Module:
    """Build a DeepONet model based on configuration flags.
    
    Args:
        cfg: Model configuration dictionary with flags:
            - use_conv_branch: If True, use Conv2d branch network
            - use_fourier_features: If True, use Fourier feature encoding in trunk
    
    Returns:
        DeepONet model instance
    """
    use_conv = bool(cfg.get("use_conv_branch", False))
    use_ff = bool(cfg.get("use_fourier_features", True))
    
    if use_conv and use_ff:
        from .deeponet_conv import build_conv_model
        return build_conv_model(cfg, use_fourier=True)
    elif use_conv and not use_ff:
        from .deeponet_conv import build_conv_model
        return build_conv_model(cfg, use_fourier=False)
    elif not use_conv and not use_ff:
        from .deeponet_vanilla import build_vanilla_model
        return build_vanilla_model(cfg)
    else:  # ResNet + FF (default)
        from .deeponet_resnet import build_resnet_model
        return build_resnet_model(cfg)


__all__ = ["build_model"]

