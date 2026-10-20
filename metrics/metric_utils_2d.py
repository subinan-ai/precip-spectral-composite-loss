# metrics/metric_utils_2d.py
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from scipy import ndimage
from piq import multi_scale_ssim

# ===== 공통 유틸 =====

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

# ===== 기본 픽셀 지표 =====

def mae(y_pred, y_true, **_):
    p, t = _to_bchw(y_pred), _to_bchw(y_true)
    return _nan_guard_scalar(F.l1_loss(p, t).item())

def mse(y_pred, y_true, **_):
    p, t = _to_bchw(y_pred), _to_bchw(y_true)
    return _nan_guard_scalar(F.mse_loss(p, t).item())

# ===== MS_SSIM / PSNR =====

@torch.no_grad()
def _log_normalize(x, shift=3.0, high_pct=99):
    # x: (B,1,H,W)
    out = []
    for idx in range(x.size(0)):
        xi = x[idx] + shift
        flat = xi.flatten()
        flat = flat[~torch.isnan(flat)]
        vmax = torch.quantile(flat, high_pct/100.0) if flat.numel() > 0 else torch.tensor(1.0, device=x.device)
        xi = torch.log1p(xi) / torch.log1p(vmax)
        xi = xi.clamp(0,1)
        xi = torch.nan_to_num(xi, nan=0.0)
        out.append(xi)
    return torch.stack(out, dim=0)

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
    

# ===== 강수 특화: CSI / FSS =====
# threshold는 정규화 단위(μ,σ 반영 후)로 들어온다고 가정

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

# ===== RHD (Regional Histogram Divergence) =====

@torch.no_grad()
def rhd(y_true, y_pred, patch_size=15, num_bins=50, epsilon=1e-6,
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

# ===== LPIPS =====

def get_lpips_model(net_type="alex", reduction="mean", normalize=True, device="cuda"):
    return LPIPS(net_type=net_type, reduction=reduction, normalize=normalize).to(device)

@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    return x.expand(-1, n, -1, -1)


@torch.no_grad()
def lpips(pred, target, model, **_):
    device = pred.device
    model = model.to(device)

    B, T, H, W = pred.shape
    scores = []
    for t in range(T):

        p = pred[:,t:t+1,:,:] # (B, 1, H, W)
        y = target[:,t:t+1,:,:]

        p = _expand_gray_to_rgb(_log_normalize(p), 3)
        y = _expand_gray_to_rgb(_log_normalize(y), 3)

        s = model(p, y).detach().mean()
        scores.append(float(s.item()))

    return _nan_guard_scalar(sum(scores) / max(1, len(scores)))

# ===== 스펙트럴 (위상/진폭) =====

@torch.no_grad()
def phase(y_pred, y_true, **_):
    x, y = _to_bchw(y_pred), _to_bchw(y_true)
    ph_x = torch.angle(torch.fft.fft2(x, dim=(-2,-1), norm="ortho"))
    ph_y = torch.angle(torch.fft.fft2(y, dim=(-2,-1), norm="ortho"))
    val = torch.cos(ph_x - ph_y).mean()
    return _nan_guard_scalar(val.item())

@torch.no_grad()
def amp(y_pred, y_true, **_):
    x, y = _to_bchw(y_pred), _to_bchw(y_true)
    ax = torch.abs(torch.fft.fft2(x, dim=(-2,-1), norm="ortho"))
    ay = torch.abs(torch.fft.fft2(y, dim=(-2,-1), norm="ortho"))
    return _nan_guard_scalar(F.mse_loss(ax, ay).item())

# ========== CPBD ==========
def _normalize_block_size(block_size):
    """
    block_size를 (bh, bw) 튜플로 정규화.
    허용: int/np.int | 길이2 시퀀스 | 0-dim 텐서 | 길이2 텐서/배열
    """
    if isinstance(block_size, (int, np.integer)):
        b = int(block_size)
        return (b, b)

    if isinstance(block_size, (list, tuple)):
        if len(block_size) == 1:
            b = int(block_size[0])
            return (b, b)
        elif len(block_size) == 2:
            return (int(block_size[0]), int(block_size[1]))
        else:
            raise TypeError(f"block_size sequence must have len 1 or 2, got {len(block_size)}")

    if isinstance(block_size, torch.Tensor):
        n = block_size.numel()
        if n == 1:
            b = int(block_size.reshape(()).item())
            return (b, b)
        elif n == 2:
            bh, bw = block_size.reshape(-1).tolist()
            return (int(bh), int(bw))
        else:
            raise TypeError(f"block_size tensor must be scalar or len-2, got shape={tuple(block_size.shape)}")

    if isinstance(block_size, np.ndarray):
        n = block_size.size
        if n == 1:
            b = int(block_size.reshape(()).item())
            return (b, b)
        elif n == 2:
            bh, bw = block_size.reshape(-1).tolist()
            return (int(bh), int(bw))
        else:
            raise TypeError(f"block_size array must be scalar or len-2, got shape={block_size.shape}")

    # 그 외 스칼라로 간주
    return (int(block_size), int(block_size))


def compute_cpbd(image, block_size=15, edge_threshold=0.002,
                 jnb_parameters=(0.5, 0.05, 10.0, 0.1)):

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy() 

    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)  # CHW -> HWC
        image = image.mean(axis=2)
    elif image.ndim > 3:
        raise ValueError(f"compute_cpbd expects 2D or 3D image, got shape={image.shape}")

    image = image.astype(np.float64, copy=False)

    bh, bw = _normalize_block_size(block_size)
    if bh <= 0 or bw <= 0:
        raise ValueError(f"block_size must be positive, got (bh,bw)=({bh},{bw})")

    h, w = map(int, image.shape[:2])

    pad_h = (bh - (h % bh)) % bh
    pad_w = (bw - (w % bw)) % bw

    if pad_h or pad_w:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        padded_image = image

    # 4) 에지 강도 맵 (Sobel)
    sobel_h = ndimage.sobel(padded_image, axis=0)
    sobel_v = ndimage.sobel(padded_image, axis=1)
    edge_strength_map = np.hypot(sobel_h, sobel_v)

    max_edge_strength = edge_strength_map.max()
    if max_edge_strength > 0:
        edge_strength_map = edge_strength_map / max_edge_strength

    a, b, c, d = jnb_parameters
    all_p_blur_values = []

    H, W = padded_image.shape
    for i in range(0, H, bh):
        for j in range(0, W, bw):
            block = padded_image[i:i+bh, j:j+bw]
            block_es = edge_strength_map[i:i+bh, j:j+bw]
            if block_es.size == 0:
                continue

            # 저텍스처 블록 스킵
            edge_ratio = (block_es > 0).mean()
            if edge_ratio < edge_threshold:
                continue

            # Michelson 대비
            max_val = block.max()
            min_val = block.min()
            denom = max_val + min_val
            contrast = 0.0 if denom == 0 else (max_val - min_val) / denom
            contrast = np.clip(contrast, 0.0, 1.0)

            # JNB 폭 (오버플로 방지)
            exp1 = -np.clip(contrast / a, -np.inf, 709)
            exp2 = -np.clip(contrast / c, -np.inf, 709)
            w_jnb = b * np.exp(exp1) + d * (1.0 - np.exp(exp2))
            if w_jnb <= 1e-12:
                continue

            # 블록 내 에지 픽셀들에 대한 blur 확률
            es_vals = block_es[block_es > 0]
            if es_vals.size:
                p_blur = 1.0 - np.exp(-(es_vals / w_jnb))
                all_p_blur_values.extend(p_blur.tolist())

    if not all_p_blur_values:
        return 0.0

    # 5) 히스토그램 누적 (CPBD)
    p = np.asarray(all_p_blur_values, dtype=np.float64)
    hist, bins = np.histogram(p, bins=100, range=(0.0, 1.0), density=True)
    p_jnb_threshold = 0.632
    mask = (bins[:-1] <= p_jnb_threshold)
    cpbd = float((hist[mask] * np.diff(bins)[mask]).sum() * 100.0)
    return cpbd

@torch.no_grad()
def cpbd(y_pred, block_size=15):
    """
    y_pred: (B,T,H,W) 또는 (B,1,H,W) 등의 텐서. 내부에서 _to_bchw로 정규화한다고 가정.
    """
    y_pred_bchw = _to_bchw(y_pred)  
    arr = y_pred_bchw.detach().cpu().numpy()   # (N,1,H,W)
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(f"_to_bchw must return (N,1,H,W), got {arr.shape}")

    amin, amax = arr.min(), arr.max()
    if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) < 1e-12:
        return 0.0

    scores = []
    for i in range(arr.shape[0]):
        img = arr[i, 0]  # (H,W)
        img_u8 = (img - amin) / (amax - amin) * 255.0
        s = compute_cpbd(image=img_u8, block_size=15, edge_threshold=0.002, jnb_parameters=(0.5,0.05,10,0.1))
        scores.append(s)

    return float(np.mean(scores)) if scores else 0.0

ALL_METRICS = {
    "mae": mae,
    "mse": mse,
    "ms_ssim": ms_ssim,
    "csi": csi,
    "fss": fss,
    "rhd": rhd,
    "phase": phase,
    "amp": amp,
    "lpips": lpips,
    "cpbd": cpbd
}