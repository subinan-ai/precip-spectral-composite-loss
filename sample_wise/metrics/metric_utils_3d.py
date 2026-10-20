# metrics/metric_utils_3d.py
import numpy as np
import math
import torch
import torch.nn.functional as F
from functools import lru_cache
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from piq import multi_scale_ssim
# from cpbd import compute
# ========== 공통 유틸 ==========

def _ensure_5d(x):
    # (B,1,T,H,W) 강제
    if x.dim() == 5:
        return x
    if x.dim() == 4:  # (B,T,H,W) -> (B,1,T,H,W)
        return x.unsqueeze(1)
    raise ValueError(f"Expected 4D or 5D tensor, got {x.shape}")

def _to_bchw(x):
    """
    입력이 (B,T,H,W)면 (B*T,1,H,W)로, (B,1,T,H,W)면 (B*T,1,H,W)로 변환.
    """
    if x.dim() == 4:  # (B,1,H,W) or (B,C,H,W) but assume C=3
        if x.size(1) != 1:
            B, T, H, W = x.shape
            x = x.reshape(B*T,1,H,W)
        return x
    if x.dim() == 5:  # (B,1,T,H,W)
        B, C, T, H, W = x.shape
        if C != 1:
            raise ValueError(f"Expected channel=1 for 2D metrics, got {C}")
        return x.permute(0,2,1,3,4).reshape(B*T, 1, H, W)
    raise ValueError(f"Expected 4D or 5D tensor, got {tuple(x.shape)}")

def _nan_guard_scalar(v: float) -> float:
    t = torch.tensor(v)
    return float(v) if torch.isfinite(t) else 0.0

# ========== 기본 픽셀 지표 ==========

def mae(y_pred, y_true, **_):
    return _nan_guard_scalar(F.l1_loss(y_pred, y_true).item())

def mse(y_pred, y_true, **_):
    return _nan_guard_scalar(F.mse_loss(y_pred, y_true).item())

# ========== MS_SSIM / PSNR ==========

@torch.no_grad()
def ms_ssim(y_pred, y_true):
    y_pred_bchw = _to_bchw(y_pred)
    y_true_bchw = _to_bchw(y_true)

    if y_pred_bchw.numel() == 0 or y_true_bchw.numel() == 0:
        return 0.0
    y_pred_normalized=_log_normalize(y_pred_bchw)
    y_true_normalized=_log_normalize(y_true_bchw)
    data_range=1.0
    
    ms_ssim_val = multi_scale_ssim(y_pred_normalized, y_true_normalized, data_range=data_range, reduction='mean')

    return _nan_guard_scalar(ms_ssim_val.item())

@torch.no_grad()
def psnr(pred, target, data_range=None, **_):
    """
    data_range가 None이면 두 텐서의 합으로 퍼센타일 기반 범위 추정 가능.
    (여기서는 간단히 1.0 기본값 유지; 필요시 params로 넘겨 세밀 제어)
    """
    pred, target = _ensure_5d(pred), _ensure_5d(target)
    mse_val = F.mse_loss(pred, target)
    if mse_val <= 0:
        return float("inf")
    if data_range is None:
        data_range = 1.0
    return _nan_guard_scalar((20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse_val)).item())

# ========== 강수 특화: CSI / FSS ==========
# ※ threshold는 정규화 단위로 들어온다고 가정(외부에서 μ,σ 반영)

@torch.no_grad()
def csi(y_pred, y_true, threshold, window_size=5, **_):
    y_true_bchw = _to_bchw(y_true)
    y_pred_bchw = _to_bchw(y_pred)
    y_true_bin = (y_true_bchw >= threshold).float()
    y_pred_bin = (y_pred_bchw >= threshold).float()

    kernel = torch.ones((1, 1, window_size, window_size), device=y_true_bchw.device)
    
    tp_sum = F.conv2d(y_true_bin * y_pred_bin, kernel, padding='valid')
    fp_sum = F.conv2d((1 - y_true_bin) * y_pred_bin, kernel, padding='valid')
    fn_sum = F.conv2d(y_true_bin * (1 - y_pred_bin), kernel, padding='valid')

    csi_map = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-8)
    
    return _nan_guard_scalar(csi_map.mean().item())

@torch.no_grad()
def fss(y_pred, y_true, threshold, window_size=5, **_):
    y_true_bchw = _to_bchw(y_true)
    y_pred_bchw = _to_bchw(y_pred)

    y_true_bin = (y_true_bchw >= threshold).float()
    y_pred_bin = (y_pred_bchw >= threshold).float()
    kernel = torch.ones((1, 1, window_size, window_size), device=y_true_bchw.device)
    
    y_true_sum = F.conv2d(y_true_bin, kernel, padding=window_size // 2)
    y_pred_sum = F.conv2d(y_pred_bin, kernel, padding=window_size // 2)

    num_pixels_in_window = window_size * window_size
    y_true_frac = y_true_sum / num_pixels_in_window
    y_pred_frac = y_pred_sum / num_pixels_in_window

    mse_frac = ((y_true_frac - y_pred_frac) ** 2).mean()
    ref_mse = ((y_true_frac ** 2).mean() + (y_pred_frac ** 2).mean())
    fss_score = 1 - mse_frac / (ref_mse + 1e-8)

    return _nan_guard_scalar(fss_score.item())

# ========== RHD (Regional Histogram Divergence) ==========

@torch.no_grad()
def rhd(y_pred, y_true, patch_size=15, num_bins=50, epsilon=1e-6,
        robust=True,  # True면 백분위수(샘플링) 사용
        q_low=0.01, q_high=0.99, quant_sample=200_000):
    """
    KL(hist_true || hist_pred)를 패치별로 구해 평균.
    완전 벡터화: 루프 없음, torch.bincount 사용.
    """
    y_true_bchw = _to_bchw(y_true)  # (B,1,H,W)
    y_pred_bchw = _to_bchw(y_pred)
    B, _, H, W = y_true_bchw.shape
    device, dtype = y_true_bchw.device, y_true_bchw.dtype
    P = patch_size * patch_size

    # (B, P, Np) -> (Np_tot, P)
    t_unf = F.unfold(y_true_bchw, kernel_size=patch_size, stride=patch_size) \
              .transpose(1, 2).reshape(-1, P)
    p_unf = F.unfold(y_pred_bchw, kernel_size=patch_size, stride=patch_size) \
              .transpose(1, 2).reshape(-1, P)
    N = t_unf.shape[0]                      # 전체 패치 수
    num_patches_per_img = N // B            

    # ---- min/max 결정 (빠른 전역 or 샘플링 백분위수) ----
    if robust:
        # 큰 텐서에서 일부만 샘플링해 백분위수 근사
        both = torch.cat([t_unf, p_unf], dim=1).reshape(-1)
        K = min(quant_sample, both.numel())
        idx = torch.randperm(both.numel(), device=device)[:K]
        sample = both[idx]
        min_val = torch.quantile(sample, q_low)
        max_val = torch.quantile(sample, q_high)
    else:
        min_val = torch.minimum(t_unf.min(), p_unf.min())
        max_val = torch.maximum(t_unf.max(), p_unf.max())

    span = (max_val - min_val).clamp_min(1e-12)

    def per_row_hist(x):
        # x: (N, P) → 각 row(패치)마다 히스토그램 (N, num_bins)
        bins = ((x - min_val) / span * num_bins).floor().to(torch.int64)
        bins = bins.clamp_(0, num_bins - 1)                 # (N, P)
        row_ids = torch.arange(x.shape[0], device=device).unsqueeze(1).expand_as(bins)
        flat_idx = row_ids * num_bins + bins                # (N, P)
        counts = torch.bincount(flat_idx.reshape(-1),
                                minlength=x.shape[0] * num_bins)
        return counts.view(x.shape[0], num_bins)

    true_h = per_row_hist(t_unf)
    pred_h = per_row_hist(p_unf)

    # 확률화 + KL
    true_prob = (true_h + epsilon) / (true_h.sum(dim=1, keepdim=True) + num_bins * epsilon)
    pred_prob = (pred_h + epsilon) / (pred_h.sum(dim=1, keepdim=True) + num_bins * epsilon)
    kl = (true_prob * (true_prob / pred_prob).log()).sum(dim=1)   # (N,)

    # 배치 평균
    rhd_val = kl.view(B, num_patches_per_img).mean(dim=1).mean()
    return rhd_val.item()


# ========== LPIPS ==========

def get_lpips_model(net_type="alex", reduction="mean", normalize=True, device="cuda"):
    return LPIPS(net_type=net_type, reduction=reduction, normalize=normalize).to(device)

@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    # x: (B,1,H,W) -> (B,n,H,W)
    return x.expand(-1, n, -1, -1)

# @torch.no_grad()
# def _log_normalize2(x, shift=3.0, high_pct=99):
#     # x: (N,1,H,W)
#     # return : (N,1,H,W)
#     out = []
#     for idx in range(x.size(0)):
#         xi = x[idx] + shift
#         flat = xi.flatten()
#         flat = flat[~torch.isnan(flat)]
#         vmax = torch.quantile(flat, high_pct/100.0) if flat.numel() > 0 else torch.tensor(1.0, device=x.device)
#         xi = torch.log1p(xi) / torch.log1p(vmax)
#         xi = xi.clamp(0,1)
#         xi = torch.nan_to_num(xi, nan=0.0)
#         out.append(xi)
#     return torch.stack(out, dim=0)

@torch.no_grad()
def _log_normalize(data, c=-0.4, d=103.0, a=0.0, b=1.0):
        """
        Apply logarithmic transformation (log +1) to the linear norm of a tensor/array. The linear normalisation is:

        data_lin_norm = (data_in - c) * (b-a) / (d-c) + a

        where a and b are the lower and upper limits of the
        resulting range and c and d are the lower and upper
        values of the input range. By default, the resulting range of
        values [a, b] is chosen to be between 0 and 1.

        arr_log_lin_norm = np.log(arr_lin_norm + 1)

        Parameters
        ----------
        data : DataType
            Input data, either a NumPy array or a PyTorch tensor.
        c : float
            The lower bound of the input range.
        d : float
            The upper bound of the input range.
        a : float, optional
            The lower bound of the output range. Defaults to 0.
        b : float, optional
            The upper bound of the output range. Defaults to 1.

        Returns
        -------
        DataType
            The transformed data with a logarithmic scale applied after linear normalization.
        """
        scale = (b - a) / (d - c)
        z = ((data - c) * scale) + a
        z = z.clamp(min=0.0, max=1.0)
        return torch.log1p(z) / math.log(2.0)

@torch.no_grad()
def lpips(pred, target, model, **_):
    pred, target = _ensure_5d(pred), _ensure_5d(target)
    device = pred.device
    model = model.to(device)

    B, C, T, H, W = pred.shape
    scores = []
    for t in range(T):
        p = pred[:, :, t]  # (B,1,H,W)
        y = target[:, :, t]

        p = _log_normalize(p)
        y = _log_normalize(y)

        p = _expand_gray_to_rgb(p, 3)
        y = _expand_gray_to_rgb(y, 3)

        s = model(p, y).detach().mean()  
        scores.append(float(s.item()))
    return _nan_guard_scalar(sum(scores) / max(1, len(scores)))

# ========== 스펙트럴 (위상/진폭) ==========

@torch.no_grad()
def phase(pred, target, **_):
    pred, target = _ensure_5d(pred), _ensure_5d(target)
    # 공간 축(H,W)에 대해 2D FFT; T 축은 유지
    ph_p = torch.angle(torch.fft.fft2(pred, dim=(-2, -1), norm="ortho"))
    ph_t = torch.angle(torch.fft.fft2(target, dim=(-2, -1), norm="ortho"))
    # 코사인 위상 차 평균 ([-1,1], 클수록 좋음)
    val = torch.cos(ph_p - ph_t).mean()
    return _nan_guard_scalar(val.item())

@torch.no_grad()
def amp(pred, target, **_):
    pred, target = _ensure_5d(pred), _ensure_5d(target)
    ap = torch.abs(torch.fft.fft2(pred, dim=(-2, -1), norm="ortho"))
    at = torch.abs(torch.fft.fft2(target, dim=(-2, -1), norm="ortho"))
    return _nan_guard_scalar(F.mse_loss(ap, at).item())

# ========== CPBD ==========
# ========== CPBD ==========

import numpy as np
import torch
from scipy import ndimage
from typing import Tuple, Optional, Union, List
from typing import Union, Tuple, Optional

def _normalize_block_size(block_size: Union[int, float, list, tuple, torch.Tensor, np.ndarray]) -> Tuple[int, int]:
    """block_size를 (bh, bw)로 정규화"""
    if isinstance(block_size, (int, np.integer)):
        b = int(block_size); return (b, b)
    if isinstance(block_size, (list, tuple)):
        if len(block_size) == 1: b = int(block_size[0]); return (b, b)
        if len(block_size) == 2: return (int(block_size[0]), int(block_size[1]))
        raise TypeError(f"block_size sequence must have len 1 or 2, got {len(block_size)}")
    if isinstance(block_size, torch.Tensor):
        n = block_size.numel()
        if n == 1: b = int(block_size.reshape(()).item()); return (b, b)
        if n == 2:
            bh, bw = block_size.reshape(-1).tolist()
            return (int(bh), int(bw))
        raise TypeError(f"block_size tensor must be scalar or len-2, got {tuple(block_size.shape)}")
    if isinstance(block_size, np.ndarray):
        n = block_size.size
        if n == 1: b = int(block_size.reshape(()).item()); return (b, b)
        if n == 2:
            bh, bw = block_size.reshape(-1).tolist()
            return (int(bh), int(bw))
        raise TypeError(f"block_size array must be scalar or len-2, got {block_size.shape}")
    b = int(block_size); return (b, b)


def _normalize_to_u8(img: np.ndarray, lo_percent: float = 0.5, hi_percent: float = 99.5) -> Optional[np.ndarray]:
    """퍼센타일 대비 [0,255] float64로 정규화 (희소 영상에 안전한 기본값)"""
    vmin = np.percentile(img, lo_percent); vmax = np.percentile(img, hi_percent)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        return None
    img01 = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    return (img01 * 255.0).astype(np.float64, copy=False)


def _get_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """부울 마스크에서 (y0, y1, x0, x1) 반환"""
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    y_min, y_max = np.where(rows)[0][[0, -1]]; x_min, x_max = np.where(cols)[0][[0, -1]]
    return y_min, y_max + 1, x_min, x_max + 1


# ---------------------------
# Core (CPBD v2)
# ---------------------------

def compute_cpbd(
    image: np.ndarray,
    block_size: int = 15,
    edge_ratio_threshold: float = 0.08,     # 블록 내 강한 에지 최소 비율
    jnb_parameters: Tuple[float, float, float, float] = (0.5, 0.05, 10.0, 0.1),
    edge_gate_percentile: float = 90.0,     # 블록 내 '강한 에지' 경계
    *,
    edge_strength_gain: float = 32.0,       # 수동 gain(autogain이 켜지면 덮어씀)
    p_threshold: float = 0.5,               # 누적 확률 임계값
    output_scale: str = "percent",          # "unit"(0~1) | "percent"(0~100)
    crop_instead_of_pad: bool = False,      # True→패딩 대신 크롭
    use_canny: bool = False,                # 기본 False로 시작(과필터링 방지)
    auto_gain: bool = True,                 # 이미지별 auto-gain
) -> float:
    """
    p_blur = 1 - exp( - w_eff / es )   (★ 방향 수정: es↑ → p_blur↓)
    w_eff = w_jnb * gain
    CPBD = P(p_blur ≤ p_threshold)  (경험적 CDF)
    """
    if image.ndim != 2:
        raise ValueError(f"compute_cpbd_v2 expects 2D, got {image.shape}")

    img = image.astype(np.float64, copy=False)
    bh, bw = _normalize_block_size(block_size)
    if bh <= 0 or bw <= 0:
        raise ValueError("block_size must be positive")

    h, w = map(int, img.shape[:2])
    pad_h = (bh - (h % bh)) % bh; pad_w = (bw - (w % bw)) % bw
    if pad_h or pad_w:
        if crop_instead_of_pad:
            padded = img[:h - (h % bh), :w - (w % bw)]
        else:
            padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        padded = img

    # 1) 에지 강도 (Sobel)
    sobel_h = ndimage.sobel(padded, axis=0)
    sobel_v = ndimage.sobel(padded, axis=1)
    grad_mag = np.hypot(sobel_h, sobel_v)

    # 선택: Canny 성향의 강한 에지만 남김(간이)
    if use_canny:
        sm = ndimage.gaussian_filter(padded, sigma=1.0)
        gh = ndimage.sobel(sm, axis=0); gv = ndimage.sobel(sm, axis=1)
        g = np.hypot(gh, gv)
        hi = np.percentile(g, 90.0)
        canny_mask = (g >= hi)  # 강한 에지만
        grad_mag = grad_mag * canny_mask

    a, b, c, d = jnb_parameters
    H, W = padded.shape
    p_vals = []

    # 2) Auto-gain: 참조 에지에서 p_blur=0.5가 되도록 (ratio=ln 2)
    if auto_gain:
        es_pos = grad_mag[grad_mag > 0]
        es_ref = np.percentile(es_pos, 50.0) if es_pos.size else 1.0
        # contrast=0.5에서의 JNB 폭
        exp1 = -np.clip(0.5 / a, a_min=None, a_max=709)
        exp2 = -np.clip(0.5 / c, a_min=None, a_max=709)
        w_ref = b * np.exp(exp1) + d * (1.0 - np.exp(exp2))  # ~0.02~0.15
        target = np.log(2.0)  # ≈ 0.693  → 1 - exp(-target) = 0.5
        edge_strength_gain = max(1e-6, (target * es_ref) / (w_ref + 1e-12))

    # 3) 블록 순회
    for i in range(0, H, bh):
        for j in range(0, W, bw):
            blk = padded[i:i+bh, j:j+bw]
            es  = grad_mag[i:i+bh, j:j+bw]
            if es.size == 0: 
                continue

            # 블록 내 '강한 에지' 선택
            es_gate = np.percentile(es, edge_gate_percentile)
            edge_ratio = float((es > es_gate).mean())
            if edge_ratio < edge_ratio_threshold:
                continue

            # 4) 대비(두 방식 평균) → JNB 폭
            vmax, vmin = blk.max(), blk.min()
            denom = vmax + vmin
            michelson = 0.0 if denom < 1e-6 else (vmax - vmin) / denom
            rms = blk.std() / (blk.mean() + 1e-6)
            contrast = float(np.clip(0.5 * michelson + 0.5 * rms, 0.0, 1.0))

            exp1 = -np.clip(contrast / a, a_min=None, a_max=709)
            exp2 = -np.clip(contrast / c, a_min=None, a_max=709)
            w_jnb = b * np.exp(exp1) + d * (1.0 - np.exp(exp2))
            w_eff = w_jnb * edge_strength_gain

            sel = es[es > es_gate]
            if sel.size == 0:
                continue

            # ★ 방향 수정: es ↑ → p_blur ↓ (선명할수록 블러 확률 작아짐)
            ratio  = np.clip(w_eff / (sel + 1e-12), 0.0, 1e6)
            p_blur = 1.0 - np.exp(-ratio)  # (0,1)
            p_vals.extend(p_blur.tolist())

    if not p_vals:
        return 0.0 if output_scale == "unit" else 0.0

    p = np.asarray(p_vals, dtype=np.float64)

    # 5) 경험적 CDF: CPBD = P(p_blur ≤ p_threshold)
    cdf_val = float((p <= p_threshold).mean())

    if output_scale == "unit":
        return float(np.clip(cdf_val, 0.0, 1.0))
    score = cdf_val * 100.0
    return float(np.clip(score, 0.0, 100.0))

@torch.no_grad()
def _cpbd_from_tensor(x: torch.Tensor,
                      *,
                      block_size: int = 15,
                      edge_ratio_threshold: float = 0.08,
                      jnb_parameters: Tuple[float, float, float, float] = (0.5, 0.05, 10.0, 0.1),
                      edge_gate_percentile: float = 90.0,
                      lo_percent: float = 0.5,
                      hi_percent: float = 99.5,
                      masks: Optional[torch.Tensor] = None,
                      edge_strength_gain: float = 32.0,
                      p_threshold: float = 0.5,
                      output_scale: str = "percent",
                      crop_instead_of_pad: bool = False,
                      use_canny: bool = False,
                      auto_gain: bool = True) -> float:
    """입력 텐서 x만을 대상으로 CPBD 계산 (배치 평균 반환)."""
    t = x.detach()
    # (B,C,T,H,W)/(B,C,H,W)/(B,H,W) 모두 허용
    if t.ndim == 3:
        t = t.unsqueeze(1).unsqueeze(2)  # (B,1,1,H,W)
    elif t.ndim == 4:
        t = t.unsqueeze(2)               # (B,C,1,H,W)
    elif t.ndim != 5:
        raise ValueError(f"expect 3D/4D/5D, got {tuple(t.shape)}")

    B, C, T, H, W = t.shape
    arr = t.cpu().numpy()

    masks_np = None
    if masks is not None:
        masks_np = masks.detach().cpu().numpy().astype(bool)
        masks_np = np.broadcast_to(masks_np, (B, C, T, H, W))

    scores = []
    for b in range(B):
        for c in range(C):
            for tt in range(T):
                img = arr[b, c, tt]  # (H,W)
                img_u8 = _normalize_to_u8(img, lo_percent, hi_percent)
                if img_u8 is None:
                    continue
                if masks_np is not None:
                    m = masks_np[b, c, tt]
                    bb = _get_bounding_box(m)
                    if bb is None: 
                        continue
                    y0, y1, x0, x1 = bb
                    img_u8 = img_u8[y0:y1, x0:x1]
                s = compute_cpbd(
                    image=img_u8,
                    block_size=block_size,
                    edge_ratio_threshold=edge_ratio_threshold,
                    jnb_parameters=jnb_parameters,
                    edge_gate_percentile=edge_gate_percentile,
                    edge_strength_gain=edge_strength_gain,
                    p_threshold=p_threshold,
                    output_scale=output_scale,
                    crop_instead_of_pad=crop_instead_of_pad,
                    use_canny=use_canny,
                    auto_gain=auto_gain,
                )
                scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def cpbd(pred: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
    """예측 영상만 사용."""
    return _cpbd_from_tensor(pred, **kwargs)

@torch.no_grad()
def cpbd_true(pred: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
    """정답 영상만 사용."""
    return _cpbd_from_tensor(y, **kwargs)


# ========== LPC ==========
def _odd_even_axis(n):
    """MATLAB 코드와 동일한 odd/even 축: [-0.5..0.5] 정규화."""
    if n % 2 == 1:
        arr = torch.arange(-(n-1)/2, (n-1)/2 + 1) / (n-1)
    else:
        arr = torch.arange(-n/2, n/2) / n
    return arr

def _meshgrid_like(rows, cols, device, dtype):
    x = _odd_even_axis(cols).to(device=device, dtype=dtype)
    y = _odd_even_axis(rows).to(device=device, dtype=dtype)
    xx, yy = torch.meshgrid(y, x, indexing="ij")
    return xx, yy

def lowpassfilter_torch(size, cutoff, n,
                        device=None, dtype=torch.float32):
    """
    MATLAB lowpassfilter.m 포팅 (torch).
    반환 필터는 ifftshift가 적용된 상태(=원점이 코너).
    """
    rows, cols = int(size[0]), int(size[1])
    if not (0 <= cutoff <= 0.5):
        raise ValueError("cutoff must be in [0,0.5]")
    if int(n) != n or n < 1:
        raise ValueError("n must be integer >= 1")

    y, x = _meshgrid_like(rows, cols, device, dtype)
    radius = torch.sqrt(x**2 + y**2)
    f = 1.0 / (1.0 + (radius / cutoff)**(2*int(n)))
    f = torch.fft.ifftshift(f)  # MATLAB과 동일
    return f

@lru_cache(maxsize=64)
def _cached_loggabor_key(rows, cols,
                         norient,
                         scales_tuple,
                         sigma_on_f,
                         dtype_str):
    """
    필터 캐시 키 전용 헬퍼. 실제 텐서는 생성 함수에서 dtype/device로 변환.
    """
    # CPU float64로 저장(정밀/재사용); 호출 시 to(device,dtype)로 옮김
    device = torch.device("cpu")
    dtype = torch.float64

    # 좌표
    y, x = _meshgrid_like(rows, cols, device, dtype)
    radius = torch.sqrt(x**2 + y**2)
    theta  = torch.atan2(-y, x)
    radius = torch.fft.ifftshift(radius)
    theta  = torch.fft.ifftshift(theta)
    radius[0, 0] = 1.0

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    # LP 필터
    lp = lowpassfilter_torch((rows, cols), 0.45, 15, device=device, dtype=dtype)

    # 스케일별 radial 파트
    minWaveLength = 4.0
    logGabor = []
    for s_val in scales_tuple:
        wavelength = minWaveLength * s_val
        fo = 1.0 / wavelength
        lg = torch.exp(- (torch.log(radius/fo)**2) / (2 * (math.log(sigma_on_f)**2)))
        lg = lg * lp
        lg[0, 0] = 0.0
        logGabor.append(lg)

    # 방향별 angular 파트
    dThetaOnSigma = 1.5
    thetaSigma = math.pi / norient / dThetaOnSigma
    spread = []
    for o in range(norient):
        angl = o * math.pi / norient
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread.append(torch.exp(-(dtheta**2) / (2 * (thetaSigma**2))))

    # 최종 필터 bank(logGabor[s] * spread[o])
    out = []
    for o in range(norient):
        for s in range(len(scales_tuple)):
            out.append((logGabor[s] * spread[o]).clone())
    # 튜플로 캐시
    return tuple(out)

def loggabor_2d_torch(rows, cols,
                      norient,
                      scales,
                      sigma_on_f = 0.33,
                      device=None, dtype=torch.float32):
    """
    반환: (norient, nscale, H, W) 실수 필터(FFT 도메인).
    """
    if isinstance(scales, float) or isinstance(scales, int):
        scales = [float(scales)**s for s in range(0, 1)]  # 안전장치
    scales = tuple(float(v) for v in scales)
    key = (rows, cols, norient, scales, sigma_on_f, str(torch.float64))
    bank = _cached_loggabor_key(*key)  # CPU float64 캐시

    nscale = len(scales)
    # (norient, nscale, H, W)로 재배치 후 to(device,dtype)
    filt = torch.stack([b for b in bank], dim=0).reshape(norient, nscale, rows, cols)
    return filt.to(device=device, dtype=dtype)

@torch.no_grad()
def lpc_si_tensor(im, C=2.0, Beta_k=1e-4,
        scales=None, norient=8, B=None, weights=None):
    """
    im: (H,W) float tensor (0~1 or 0~255) on any device.
    반환: (si, lpc_map[H,W], torch.float64)
    """
    if im.ndim != 2:
        raise ValueError("im should be 2D (H,W)")

    H, W = im.shape
    device = im.device
    cdtype = torch.complex64 if im.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128
    rdtype = torch.float32 if im.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float64

    # 기본 scales/weights
    if scales is None:
        scales = [1.0, 1.5, 2.0]
        w = torch.tensor([1.0, -3.0, 2.0], dtype=rdtype, device=device)
    else:
        scales = [float(v) for v in scales]
        if weights is None:
            # 원본 m파일의 주석에 맞춘 휴리스틱
            if len(scales) == 3 and all(abs(a-b) < 1e-8 for a, b in zip(scales, [1, 2, 3])):
                w = torch.tensor([1.0, -4.0, 3.0], dtype=rdtype, device=device)
            elif len(scales) == 3 and all(abs(a-b) < 1e-8 for a, b in zip(scales, [1, 2, 4])):
                w = torch.tensor([1.0, -3.0, 2.0], dtype=rdtype, device=device)
            elif len(scales) == 3 and all(abs(a-b) < 1e-8 for a, b in zip(scales, [1, 1.5, 2.0])):
                w = torch.tensor([1.0, -3.0, 2.0], dtype=rdtype, device=device)
            else:
                raise ValueError("Custom scales 사용 시 weights도 명시하세요 (예: [1,-3,2])")
        else:
            w = torch.tensor(list(weights), dtype=rdtype, device=device)
            if len(w) != len(scales):
                raise ValueError("len(weights) must equal len(scales)")

    nscale = len(scales)
    if B is None:
        B = int(round(min(H, W) / 16.0))

    # 필터 뱅크 (FFT domain)
    filt = loggabor_2d_torch(H, W, norient, scales, sigma_on_f=0.33, device=device, dtype=rdtype)  # (O,S,H,W)

    # FFT(im)
    X = torch.fft.fft2(im.to(rdtype))

    # 누적 텐서
    s_lpc = torch.ones((H, W, norient), dtype=cdtype, device=device)
    energy = torch.zeros((H, W, norient), dtype=rdtype, device=device)

    # 방향 루프
    for o in range(norient):
        M_first = None
        for s in range(nscale):
            Hso = filt[o, s]  # (H,W), real
            M = torch.fft.ifft2(X * Hso)  # complex
            if s == 0:
                M_first = M
            # 복소 거듭제곱
            s_lpc[..., o] = s_lpc[..., o] * (M ** w[s])

        # energy: |M_first|에서 T=mean+2*std, max(0, e-T)
        e = torch.abs(M_first)
        if B > 0 and (2*B < H) and (2*B < W):
            e_center = e[B:H-B, B:W-B]
        else:
            e_center = e
        T = e_center.mean() + 2.0 * e_center.std(unbiased=False)
        e = torch.clamp(e - T, min=0.0)
        energy[..., o] = e

    # cos(angle(s_lpc)), 음수 컷
    s_lpc_map = torch.cos(torch.angle(s_lpc))
    s_lpc_map = torch.where(s_lpc_map < 0, torch.zeros_like(s_lpc_map), s_lpc_map)

    num = (s_lpc_map * energy).sum(dim=2)
    den = energy.sum(dim=2) + C
    lpc_map = num / den  # (H,W)

    # SI 계산(중앙 crop 정렬→가중평균)
    if B > 0 and (2*B < H) and (2*B < W):
        center = lpc_map[B:H-B, B:W-B]
    else:
        center = lpc_map

    sorted_vals = torch.sort(center.reshape(-1), descending=True).values
    N = sorted_vals.numel()
    if N <= 1:
        u = torch.ones(1, dtype=rdtype, device=device)
    else:
        idx = torch.arange(N, dtype=rdtype, device=device)
        u = torch.exp(- (idx / (N - 1.0)) / Beta_k)

    si = float((sorted_vals * u).sum().item() / (u.sum().item()))
    return si

@torch.no_grad()
def lpc_si(pred, target):
    """
    pred: torch.tensor (B, 1, H, W) or (B, T, H, W)
    target: torch.tensor (only for function interface)
    """
    # tensor reshape
    imgs = _to_bchw(pred) # (B*T, 1, H, W)

    # log normalize to [0, 1]
    imgs = _log_normalize(imgs) # (B*T, 1, H, W)

    scores = []
    for i in range(imgs.shape[0]):
        scores.append(lpc_si_tensor(imgs[i, 0]))

    return _nan_guard_scalar(sum(scores) / max(1, len(scores)))

@torch.no_grad()
def lpc_si_true(pred, target):
    """
    pred: torch.tensor (B, 1, H, W) or (B, T, H, W)
    target: torch.tensor (only for function interface)
    """
    # tensor reshape
    imgs = _to_bchw(target) # (B*T, 1, H, W)

    # log normalize to [0, 1]
    imgs = _log_normalize(imgs) # (B*T, 1, H, W)

    scores = []
    for i in range(imgs.shape[0]):
        scores.append(lpc_si_tensor(imgs[i, 0]))

    return _nan_guard_scalar(sum(scores) / max(1, len(scores)))

ALL_METRICS = {
    "mae": mae,
    "mse": mse,
    "ms_ssim": ms_ssim,
    "csi": csi,
    "fss": fss,
    "rhd": rhd,
    "phase": phase,
    "amp": amp,
    "cpbd": cpbd,
    "cpbd_true": cpbd_true,
    "lpips": lpips,
    "lpc_si": lpc_si,
    "lpc_si_true": lpc_si_true
}