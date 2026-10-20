# losses/loss_2d/registry.py
from .losses import (
    loss_mse, loss_mae,
    loss_fal, loss_fcl, loss_facl,
    loss_fcl_ll, loss_fcl_high, loss_fcl_all,
    loss_fal_ll, loss_fal_high, loss_fal_all,
    loss_ll_only, loss_high_only, loss_wavelet_all, loss_wavelet_mixed,
    loss_facl_dtcwt_lv1,
)

_LOSSES_2D = {
    "mse": loss_mse,
    "mae": loss_mae,
    "fal": loss_fal,
    "fcl": loss_fcl,
    "facl": loss_facl,
    "fcl_ll": loss_fcl_ll,
    "fcl_high": loss_fcl_high,
    "fcl_all": loss_fcl_all,
    "fal_ll": loss_fal_ll,
    "fal_high": loss_fal_high,
    "fal_all": loss_fal_all,
    "ll_only": loss_ll_only,
    "high_only": loss_high_only,
    "wavelet_all": loss_wavelet_all,
    "wavelet_mixed": loss_wavelet_mixed,
    "facl_dtcwt_lv1": loss_facl_dtcwt_lv1,
}

def get_loss_2d(name: str):
    name = str(name).lower()
    if name not in _LOSSES_2D:
        raise KeyError(f"[2D] Unknown loss: {name}")
    return _LOSSES_2D[name]
