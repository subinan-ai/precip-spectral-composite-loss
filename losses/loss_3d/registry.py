# losses/loss_3d/registry.py
from .losses import (
    loss_mse, loss_mae, loss_fal, loss_fcl, loss_facl,
    loss_facl_dwt,
    loss_facl_dtcwt_lv1, loss_facl_dtcwt_lv2, loss_facl_dtcwt_lv3,
    loss_wacl_mse_lv2, loss_wacl_mae_lv2,
    loss_fcl_ll, loss_fcl_high, loss_fcl_all,
    loss_fal_ll, loss_fal_high, loss_fal_all,
    loss_ll_only, loss_high_only, loss_wavelet_all, loss_wavelet_mixed,
    loss_facl_dtcwt_lv2_weighted,
    loss_wacl_lv1, loss_wacl_lv2, loss_wacl_lv3,
    loss_facl_wacl_lv1, loss_facl_wacl_lv2, loss_facl_wacl_lv3,
    loss_facl_wavelet, loss_facl_wavelet_ll, loss_facl_wavelet_high,
    loss_wacl_mse_lv2_pixel, loss_facl_wacl_lv3_mae, loss_facl_wacl_lv3_alpha,
    loss_facl_wacl_lv3_phase_left,
    loss_facl_wacl_lv3_w1, loss_facl_wacl_lv3_w2, loss_facl_wacl_lv3_w3,
    loss_wacl_lv3_low
)

_LOSSES_3D = {
    "mse": loss_mse,
    "mae": loss_mae,
    "fal": loss_fal,
    "fcl": loss_fcl,
    "facl": loss_facl,
    "facl_dwt": loss_facl_dwt,
    "facl_dtcwt_lv1": loss_facl_dtcwt_lv1,
    "facl_dtcwt_lv2": loss_facl_dtcwt_lv2,
    "facl_dtcwt_lv2_weighted": loss_facl_dtcwt_lv2_weighted,
    "wacl_lv1": loss_wacl_lv1,
    "wacl_lv2": loss_wacl_lv2,
    "wacl_lv3": loss_wacl_lv3,
    "facl_wacl_lv1": loss_facl_wacl_lv1,
    "facl_wacl_lv2": loss_facl_wacl_lv2,
    "facl_wacl_lv3": loss_facl_wacl_lv3,
    "wacl_mse_lv2": loss_wacl_mse_lv2,
    "wacl_mse_lv2_pixel": loss_wacl_mse_lv2_pixel,
    "wacl_mae_lv2": loss_wacl_mae_lv2,
    "facl_dtcwt_lv3": loss_facl_dtcwt_lv3,
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
    "facl_wavelet": loss_facl_wavelet,
    "facl_wavelet_ll": loss_facl_wavelet_ll,
    "facl_wavelet_high": loss_facl_wavelet_high,
    "facl_wacl_lv3_mae": loss_facl_wacl_lv3_mae,
    "facl_wacl_lv3_alpha": loss_facl_wacl_lv3_alpha,
    "facl_wacl_lv3_phase_left": loss_facl_wacl_lv3_phase_left,
    "wacl_lv3_low": loss_wacl_lv3_low,
    "facl_wacl_lv3_w1": loss_facl_wacl_lv3_w1,
    "facl_wacl_lv3_w2": loss_facl_wacl_lv3_w2,
    "facl_wacl_lv3_w3": loss_facl_wacl_lv3_w3,
}

def get_loss_3d(name: str):
    name = str(name).lower()
    if name not in _LOSSES_3D:
        raise KeyError(f"[3D] Unknown loss: {name}")
    return _LOSSES_3D[name]
