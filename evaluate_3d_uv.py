# evaluate_3d.py
import os
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from utils import set_seed
from utils.save_utils import load_checkpoint_for_eval
from models.get_model import build_model
from metrics import get_metric_fns
from datasets.dataloader import get_test_loader
# ===== CPBD pair 시각화 유틸 =====
import torch, numpy as np, torch.nn.functional as F
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

# 네가 이미 정의한 compute_cpbd를 사용
# from your_module import compute_cpbd
import numpy as np
import torch
from scipy import ndimage
from typing import Tuple, Optional, Union, List
from typing import Union, Tuple, Optional
##########폰트, 글자 크기 통일############
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
})
#####################################
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
def _cpbd_heatmap(
    field_2d: np.ndarray,
    patch=32, stride=16,
    *, cpbd_fn=compute_cpbd, cpbd_kwargs=None,
    lo_fixed=None, hi_fixed=None   # ← 새로 추가
):
    """단일 2D 필드 → 슬라이딩 윈도우 CPBD 히트맵"""
    if cpbd_kwargs is None:
        cpbd_kwargs = {}
    H, W = field_2d.shape

    # 정규화 범위 선택 (GT 기준 lo/hi를 외부에서 고정 가능)
    if lo_fixed is not None and hi_fixed is not None:
        lo, hi = lo_fixed, hi_fixed
    else:
        lo, hi = np.percentile(field_2d, 0.5), np.percentile(field_2d, 99.5)

    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        img_u8 = np.zeros_like(field_2d, dtype=np.float64)
    else:
        img01 = np.clip((field_2d - lo) / (hi - lo), 0, 1)
        img_u8 = (img01 * 255.0).astype(np.float64, copy=False)

    Hp = 1 + max(0, (H - patch)//stride)
    Wp = 1 + max(0, (W - patch)//stride)
    heat = np.zeros((Hp, Wp), dtype=np.float32)

    r = 0
    for i in range(0, H - patch + 1, stride):
        c = 0
        for j in range(0, W - patch + 1, stride):
            tile = img_u8[i:i+patch, j:j+patch]
            heat[r, c] = float(cpbd_fn(tile, **cpbd_kwargs))
            c += 1
        r += 1

    heat_t = torch.from_numpy(heat)[None, None]
    heat_up = torch.nn.functional.interpolate(
        heat_t, size=(H, W), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return heat_up


@torch.no_grad()
def plot_cpbd_pair_from_tensors(
    gt, pred, b=0, c=0, t=0,
    extent=(0, 360, -90, 90),
    central_longitude=180,
    patch=32, stride=16,
    cpbd_kwargs=None,
    save_path=None,
    # --- 폰트 크기: 이 두 가지만 함수 내부에서 강제 적용 ---
    title_fontsize=19,       # 패널 제목 크기
    tick_labelsize=13,       # x/y 축 눈금 크기
    # --- 나머지는 rcParams에 따르게 둠 ---
    dpi=300,
    cmap="viridis",
    # ===== 아래 3개는 '형식 유지'는 하되, 대비 개선을 위한 순수 옵션 =====
    scale_mode="power",      # "power"(기본) | "linear" | "fixed"
    p_lo=5.0, p_hi=95.0,     # 퍼센타일 클리핑(선형/파워 공통)
    gamma=0.7,               # PowerNorm 감마 (작을수록 중간대 대비↑)
    fixed_vmin=None, fixed_vmax=None,  # scale_mode="fixed"일 때 사용
):
    """
    CPBD 기반 sharpness (GT vs Pred) 비교 시각화.
    - GT intensity 범위를 기준으로 정규화
    - auto_gain=False (고정)
    - 제목/축 눈금만 함수 내부에서 크게 설정(나머지는 rcParams)
    - 스케일/정규화 옵션:
        * scale_mode="power" : PowerNorm(γ) + 퍼센타일 클리핑 (기본)
        * scale_mode="linear": 선형 + 퍼센타일 클리핑
        * scale_mode="fixed" : 선형 + 고정 스케일(fixed_vmin/fixed_vmax)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.ticker as mticker
    import matplotlib.colors as mcolors

    # ---- 데이터 ----
    gt_map   = gt[b, c, t].detach().cpu().numpy()
    pred_map = pred[b, c, t].detach().cpu().numpy()

    # ---- GT 기준 밝기 구간 ----
    lo = np.percentile(gt_map, 0.5)
    hi = np.percentile(gt_map, 99.5)

    # ---- CPBD heatmap (auto_gain 끔) ----
    common_kwargs = dict(auto_gain=False, edge_strength_gain=32.0)
    if cpbd_kwargs is not None:
        common_kwargs.update(cpbd_kwargs)

    hm_gt = _cpbd_heatmap(
        gt_map, patch=patch, stride=stride,
        cpbd_fn=compute_cpbd, cpbd_kwargs=common_kwargs,
        lo_fixed=lo, hi_fixed=hi
    )
    hm_pred = _cpbd_heatmap(
        pred_map, patch=patch, stride=stride,
        cpbd_fn=compute_cpbd, cpbd_kwargs=common_kwargs,
        lo_fixed=lo, hi_fixed=hi
    )

    # ===== 공통 컬러 스케일/정규화 =====
    flat = np.concatenate([hm_gt.ravel(), hm_pred.ravel()])

    norm = None
    vmin = None
    vmax = None

    if scale_mode == "fixed":
        # 고정 스케일 (데이터셋 간 절대 비교용)
        if fixed_vmin is None or fixed_vmax is None:
            raise ValueError("scale_mode='fixed'인 경우 fixed_vmin, fixed_vmax를 지정하세요.")
        vmin, vmax = float(fixed_vmin), float(fixed_vmax)

    elif scale_mode == "linear":
        # 선형 + 퍼센타일 클리핑
        vmin = float(np.nanpercentile(flat, p_lo))
        vmax = float(np.nanpercentile(flat, p_hi))

    else:
        # "power" (기본): PowerNorm(감마) + 퍼센타일 클리핑
        vmin = float(np.nanpercentile(flat, p_lo))
        vmax = float(np.nanpercentile(flat, p_hi))
        # 감마 < 1 → 중간대 대비 강화
        norm = mcolors.PowerNorm(gamma=max(1e-6, float(gamma)), vmin=vmin, vmax=vmax)

    # ---- Figure & Axes ----
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5),
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )

    # ---- 축 장식(크기 지정은 바깥에서 처리) ----
    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.5, linestyle="dashed",
            color="gray", alpha=0.6, zorder=6
        )
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

    # ---- 도색 helper (norm이 있으면 norm 우선) ----
    def _pcolor(ax, data):
        X = np.linspace(extent[0], extent[1], data.shape[1])
        Y = np.linspace(extent[2], extent[3], data.shape[0])
        if norm is not None:
            return ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm,
                                transform=ccrs.PlateCarree(), shading="auto")
        else:
            return ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                                transform=ccrs.PlateCarree(), shading="auto")

    # ---- GT ----
    im0 = _pcolor(axes[0], hm_gt)
    axes[0].set_title("(a) GT", fontsize=title_fontsize)
    _decorate(axes[0])
    axes[0].tick_params(axis="both", which="major", labelsize=tick_labelsize)

    # ---- Pred ----
    im1 = _pcolor(axes[1], hm_pred)
    axes[1].set_title("(c) FACL", fontsize=title_fontsize)
    _decorate(axes[1])
    axes[1].tick_params(axis="both", which="major", labelsize=tick_labelsize)

    # ---- 공유 컬러바(폰트는 rcParams에 따름) ----
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical",
                        shrink=0.9, fraction=0.046, pad=0.04)
    cbar.set_label("CPBD", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # ---- 저장/표시 ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

        plt.show()

@torch.no_grad()
def plot_cpbd_diff_pair_from_tensors(
    gt, pred_a, pred_b,
    label_a="Model-A", label_b="Model-B",
    b=0, c=0, t=0,
    extent=(0, 360, -90, 90),
    central_longitude=180,
    patch=32, stride=16,
    cpbd_kwargs=None,
    save_path=None,
    # fonts (match your style)
    title_fontsize=19,
    tick_labelsize=13,
    dpi=300,
    cmap="viridis",
    diff_cmap="coolwarm",
    abs_diff=False,
):
    """
    두 loss(mod el) 간 CPBD 차이 시각화.
    - 상단: 각 모델의 CPBD 히트맵 (공통 컬러스케일)
    - 하단: ∆CPBD = CPBD(label_b) - CPBD(label_a) (기본 signed, abs_diff=True면 |∆|)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.ticker as mticker

    # ---- 데이터 ----
    gt_map   = gt[b, c, t].detach().cpu().numpy()
    a_map    = pred_a[b, c, t].detach().cpu().numpy()
    b_map    = pred_b[b, c, t].detach().cpu().numpy()

    # ---- GT 기준 밝기 구간 ----
    lo = np.percentile(gt_map, 0.5)
    hi = np.percentile(gt_map, 99.5)

    # ---- CPBD heatmap (auto_gain 끔) ----
    common_kwargs = dict(auto_gain=False, edge_strength_gain=32.0)
    if cpbd_kwargs is not None:
        common_kwargs.update(cpbd_kwargs)

    hm_a = _cpbd_heatmap(
        a_map, patch=patch, stride=stride,
        cpbd_fn=compute_cpbd, cpbd_kwargs=common_kwargs,
        lo_fixed=lo, hi_fixed=hi
    )
    hm_b = _cpbd_heatmap(
        b_map, patch=patch, stride=stride,
        cpbd_fn=compute_cpbd, cpbd_kwargs=common_kwargs,
        lo_fixed=lo, hi_fixed=hi
    )

    # ---- 공통 컬러 스케일(히트맵) ----
    vmin_hm = 0.0
    vmax_hm = float(np.nanpercentile(
        np.concatenate([hm_a.ravel(), hm_b.ravel()]), 99.5
    ))

    # ---- ∆CPBD ----
    diff = hm_b - hm_a
    if abs_diff:
        diff = np.abs(diff)
        vmin_df, vmax_df = 0.0, float(np.nanpercentile(diff, 99.5))
    else:
        vmax_df = float(np.nanpercentile(np.abs(diff), 99.5))
        vmin_df = -vmax_df

    # ---- Figure ----
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8),
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )

    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.5, linestyle="dashed",
            color="gray", alpha=0.6, zorder=6
        )
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

    # 좌표
    xs = np.linspace(extent[0], extent[1], hm_a.shape[1])
    ys = np.linspace(extent[2], extent[3], hm_a.shape[0])

    # (a) A
    im00 = axes[0,0].pcolormesh(xs, ys, hm_a, cmap=cmap, vmin=vmin_hm, vmax=vmax_hm,
                                transform=ccrs.PlateCarree(), shading="auto")
    axes[0,0].set_title(f"(a) {label_a}", fontsize=title_fontsize); _decorate(axes[0,0])
    axes[0,0].tick_params(axis="both", which="major", labelsize=tick_labelsize)

    # (b) B
    im01 = axes[0,1].pcolormesh(xs, ys, hm_b, cmap=cmap, vmin=vmin_hm, vmax=vmax_hm,
                                transform=ccrs.PlateCarree(), shading="auto")
    axes[0,1].set_title(f"(b) {label_b}", fontsize=title_fontsize); _decorate(axes[0,1])
    axes[0,1].tick_params(axis="both", which="major", labelsize=tick_labelsize)

    # (c) ∆CPBD
    im10 = axes[1,0].pcolormesh(xs, ys, diff, cmap=diff_cmap, vmin=vmin_df, vmax=vmax_df,
                                transform=ccrs.PlateCarree(), shading="auto")
    axes[1,0].set_title(f"(c) FACL" + (" (abs)" if abs_diff else ""), fontsize=title_fontsize)
    _decorate(axes[1,0]); axes[1,0].tick_params(axis="both", which="major", labelsize=tick_labelsize)

    # (d) 기준선(0) 강조를 위한 빈 패널 또는 GT-범례
    axes[1,1].axis("off")

    # 컬러바
    cbar0 = fig.colorbar(im01, ax=[axes[0,0], axes[0,1]], orientation="horizontal", fraction=0.05, pad=0.08)
    cbar0.set_label("CPBD", fontsize=16); cbar0.ax.tick_params(labelsize=12)
    cbar1 = fig.colorbar(im10, ax=[axes[1,0]], orientation="horizontal", fraction=0.05, pad=0.08)
    cbar1.set_label("∆CPBD" + (" (abs)" if abs_diff else " (signed)"), fontsize=16); cbar1.ax.tick_params(labelsize=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


@torch.no_grad()
def plot_cpbd_diffmap_from_tensors(
    *,
    gt, pred, baseline,               # (B,1,T,H,W)
    b=0, c=0, t=0,
    extent=(0, 360, -90, 90),
    central_longitude=180,
    patch=32, stride=16,
    cpbd_kwargs=None,
    save_path=None,
    # style
    title="ΔCPBD (pred − baseline)",
    title_fontsize=19, tick_labelsize=13,
    dpi=300, diff_cmap="coolwarm", abs_diff=False,
):
    import os, numpy as np, matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.ticker as mticker

    gt_map  = gt[b, c, t].detach().cpu().numpy()
    p_map   = pred[b, c, t].detach().cpu().numpy()
    b_map   = baseline[b, c, t].detach().cpu().numpy()

    # GT 기준 밝기 구간 고정
    lo = np.percentile(gt_map, 0.5)
    hi = np.percentile(gt_map, 99.5)

    common_kwargs = dict(auto_gain=False, edge_strength_gain=32.0)
    if cpbd_kwargs is not None:
        common_kwargs.update(cpbd_kwargs)

    # 두 히트맵을 같은 lo/hi 로 계산
    hm_p = _cpbd_heatmap(
        p_map, patch=patch, stride=stride,
        cpbd_fn=compute_cpbd, cpbd_kwargs=common_kwargs,
        lo_fixed=lo, hi_fixed=hi
    )
    hm_b = _cpbd_heatmap(
        b_map, patch=patch, stride=stride,
        cpbd_fn=compute_cpbd, cpbd_kwargs=common_kwargs,
        lo_fixed=lo, hi_fixed=hi
    )

    # ∆CPBD
    diff = hm_p - hm_b
    if abs_diff:
        diff = np.abs(diff)
        vmin, vmax = 0.0, float(np.nanpercentile(diff, 99.5))
    else:
        vmax = float(np.nanpercentile(np.abs(diff), 99.5))
        vmin = -vmax

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    xs = np.linspace(extent[0], extent[1], diff.shape[1])
    ys = np.linspace(extent[2], extent[3], diff.shape[0])

    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.5, linestyle="dashed",
            color="gray", alpha=0.6, zorder=6
        )
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

        # ✅ 축/지도 배경 흰색
        ax.set_facecolor("white")
        if hasattr(ax, "background_patch"):
            ax.background_patch.set_facecolor("white")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(
        1, 1, figsize=(6.4, 4.6),
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )

    # ✅ 전체 Figure 배경도 흰색
    fig.patch.set_facecolor("white")

    im = ax.pcolormesh(
        xs, ys, diff,
        cmap=diff_cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), shading="auto"
    )
    ax.set_title(title + (" (abs)" if abs_diff else ""), fontsize=title_fontsize)
    _decorate(ax)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("ΔCPBD", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    # (선택) 컬러바 배경도 흰색으로
    cbar.ax.set_facecolor("white")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
#---------------------------------------
#CPBD 밴드별 그림 (bar plot 포함)
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from dtcwt.numpy import Transform2d
from scipy.ndimage import zoom

# ---------------------------------------------------
# (네가 제공한 cpbd 구현을 import 혹은 동일 파일에 둔다고 가정)
# from your_module import _cpbd_from_tensor   # 여기선 바로 사용
# ---------------------------------------------------

def to_5d_BCTHW(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5: return x
    if x.ndim == 4: return x.unsqueeze(2)     # (B,C,H,W)->(B,C,1,H,W)
    if x.ndim == 3: return x.unsqueeze(1).unsqueeze(2)  # (B,H,W)->(B,1,1,H,W)
    raise ValueError(f"Unsupported shape {tuple(x.shape)}")

def pct_norm(a: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-8):
    lo, hi = np.percentile(a, [pmin, pmax])
    return (np.clip(a, lo, hi) - lo) / (hi - lo + eps)

def upsample_to(a: np.ndarray, H: int, W: int, order=1):
    if a.shape == (H, W): return a
    zy, zx = H / a.shape[0], W / a.shape[1]
    return zoom(a, (zy, zx), order=order)

def imshow_geo(
    ax,
    data2d,
    extent=(0,360,-90,90),
    central_longitude=180,
    vmin=0,
    vmax=1,
    cmap="viridis",
    title="",
    tick_labelsize=12,
):
    import numpy as np
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # 흰 배경
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['savefig.edgecolor']   = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax.set_global()
    ax.coastlines(linewidth=0.6, color="dimgray")

    ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
    ax.set_yticks(np.arange(-90,   91, 30), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=tick_labelsize)

    cmap_obj = mpl.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad('white', 1.0)
    cmap_obj.set_under('white', 1.0)

    img = ax.imshow(
        data2d,
        origin="lower",                 # 위아래 반전 방지
        transform=ccrs.PlateCarree(),
        extent=extent,
        vmin=vmin, vmax=vmax,
        cmap=cmap_obj,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=13)
    return img





def dtcwt_decompose(arr2d: np.ndarray, levels: int = 3):
    tfm = Transform2d()
    coeffs = tfm.forward(arr2d, nlevels=levels)
    ll = coeffs.lowpass                     # (H/2^L, W/2^L)
    highs = list(coeffs.highpasses)         # len=levels, each (h,w,6) complex
    return ll, highs

def _cpbd_numpy_2d(arr2d: np.ndarray, **cpbd_kwargs) -> float:
    """
    네 cpbd 파이프라인에 맞추어 (1,1,1,H,W) 텐서로 포장하여 계산.
    arr2d: 2D numpy array (H,W), dtype float
    """
    ten = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return _cpbd_from_tensor(ten, **cpbd_kwargs)

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from dtcwt.numpy import Transform2d
#from dtcwt.pyramid import Pyramid
from scipy.ndimage import zoom

from dtcwt.numpy import Transform2d

import numpy as np
from dtcwt.numpy.transform2d import Transform2d  # 경로가 다르면 try/except로 호환 임포트

def dtcwt_forward(arr2d: np.ndarray, levels: int = 3):
    tfm = Transform2d()
    pyr = tfm.forward(arr2d, nlevels=levels)  # pyr.lowpass (float), pyr.highpasses (list of complex)
    return tfm, pyr

def reconstruct_single_high_band(arr2d, levels, level, orient):
    tfm, pyr = dtcwt_forward(arr2d, levels=levels)
    PyrClass = pyr.__class__
    lows  = np.zeros_like(pyr.lowpass)
    highs = [np.zeros_like(H) for H in pyr.highpasses]
    highs[level-1][..., orient] = pyr.highpasses[level-1][..., orient]
    rec = tfm.inverse(PyrClass(lows, highs))
    return rec.astype(np.float32, copy=False)

def reconstruct_lowpass_only(arr2d, levels):
    """모든 highpass=0, lowpass만으로 역복원 → LL-only 영상."""
    tfm, pyr = dtcwt_forward(arr2d, levels=levels)
    PyrClass = pyr.__class__
    highs = [np.zeros_like(H) for H in pyr.highpasses]
    rec = tfm.inverse(PyrClass(pyr.lowpass, highs))
    return rec.astype(np.float32, copy=False)


# ---------------------------
# 너의 CPBD 함수(이미 같은 파일에 있다고 가정)
# from your_module import _cpbd_from_tensor
# ---------------------------

def to_5d_BCTHW(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5: return x
    if x.ndim == 4: return x.unsqueeze(2)                 # (B,C,H,W)->(B,C,1,H,W)
    if x.ndim == 3: return x.unsqueeze(1).unsqueeze(2)    # (B,H,W)->(B,1,1,H,W)
    raise ValueError(f"Unsupported shape {tuple(x.shape)}")

def pct_norm(a: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-8):
    lo, hi = np.percentile(a, [pmin, pmax])
    return (np.clip(a, lo, hi) - lo) / (hi - lo + eps)

def upsample_to(a: np.ndarray, H: int, W: int, order=3):
    if a.shape == (H, W): return a
    zy, zx = H / a.shape[0], W / a.shape[1]
    return zoom(a, (zy, zx), order=order)

def imshow_geo(
    ax,
    data2d,
    extent=(0,360,-90,90),
    central_longitude=180,
    vmin=0,
    vmax=1,
    cmap="viridis",
    title="",
    tick_labelsize=12,
):
    import numpy as np
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # 흰 배경
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['savefig.edgecolor']   = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax.set_global()
    ax.coastlines(linewidth=0.6, color="dimgray")

    ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
    ax.set_yticks(np.arange(-90,   91, 30), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=tick_labelsize)

    cmap_obj = mpl.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad('white', 1.0)
    cmap_obj.set_under('white', 1.0)

    img = ax.imshow(
        data2d,
        origin="lower",                 # 위아래 반전 방지
        transform=ccrs.PlateCarree(),
        extent=extent,
        vmin=vmin, vmax=vmax,
        cmap=cmap_obj,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=13)
    return img



def _cpbd_numpy_2d(arr2d: np.ndarray, **cpbd_kwargs) -> float:
    ten = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return _cpbd_from_tensor(ten, **cpbd_kwargs)

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from dtcwt.numpy import Transform2d
from scipy.ndimage import zoom, gaussian_filter

# ----------------------------------
# 유틸
# ----------------------------------
def to_5d_BCTHW(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5: return x
    if x.ndim == 4: return x.unsqueeze(2)               # (B,C,H,W)->(B,C,1,H,W)
    if x.ndim == 3: return x.unsqueeze(1).unsqueeze(2)  # (B,H,W)->(B,1,1,H,W)
    raise ValueError(f"Unsupported shape {tuple(x.shape)}")

def pct_norm(a: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-8):
    lo, hi = np.percentile(a, [pmin, pmax])
    return (np.clip(a, lo, hi) - lo) / (hi - lo + eps)

def resample_to(a: np.ndarray, out_h: int, out_w: int, order=3, antialias=True):
    """고해상도→저해상도 시 aliasing 방지용 가우시안 프리필터 옵션"""
    in_h, in_w = a.shape
    if antialias and (out_h < in_h or out_w < in_w):
        # Nyquist 근사: sigma ≈ 0.5 * sqrt(max(scale^-2 - 1, 0))
        sy = in_h / out_h; sx = in_w / out_w
        sigy = max(0.0, 0.5 * np.sqrt(max(sy**2 - 1.0, 0.0)))
        sigx = max(0.0, 0.5 * np.sqrt(max(sx**2 - 1.0, 0.0)))
        if sigx > 0 or sigy > 0:
            a = gaussian_filter(a, sigma=(sigy, sigx), mode="nearest")
    return zoom(a, (out_h / a.shape[0], out_w / a.shape[1]), order=order)

'''def imshow_geo(ax, data2d, extent=(0,360,-90,90), central_longitude=180,
               vmin=0, vmax=1, cmap="viridis", title="", tick_labelsize=12):
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.4)
    gl.top_labels = False; gl.right_labels = False
    ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
    ax.set_yticks(np.arange(-90,   91, 30), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=tick_labelsize)

    img = ax.imshow(
        data2d, origin="upper", transform=ccrs.PlateCarree(),
        extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest",
    )
    ax.set_title(title, fontsize=13)
    return img'''

# ----------------------------------
# LPIPS 보조
# ----------------------------------
def _to_lpips_input(x: np.ndarray) -> torch.Tensor:
    """
    LPIPS 입력 형식으로 변환:
      - x: 2D numpy, [0,1] 권장 (이미 pct_norm 등으로 정규화 후)
      - 반환: (1,3,H,W) float32, 범위 [-1,1]
    """
    x01 = np.clip(x, 0.0, 1.0).astype(np.float32)
    x_t = torch.from_numpy(x01)[None, None, ...]       # (1,1,H,W)
    x_t = x_t.repeat(1, 3, 1, 1)                       # grayscale→RGB
    x_t = x_t * 2.0 - 1.0                              # [0,1] → [-1,1]
    return x_t

@torch.no_grad()
def lpips_score_2d(pred_img: np.ndarray, gt_img: np.ndarray, lpips_model) -> float:
    """
    pred_img / gt_img: 2D numpy, 이미 같은 해상도.
    lpips_model: e.g., lpips.LPIPS(net='alex').eval().to(device)
    """
    device = next(lpips_model.parameters()).device
    x0 = _to_lpips_input(pred_img).to(device)
    y0 = _to_lpips_input(gt_img).to(device)
    s  = lpips_model(x0, y0).item()
    return float(s)

# ----------------------------------
# 메인: Lv1 해상도로 통일 + LPIPS bar
# ----------------------------------
import os, math, numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from matplotlib import cm
from dtcwt.numpy import Transform2d
from scipy.ndimage import zoom, gaussian_filter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as TM_LPIPS

# =========================
# LPIPS 준비/헬퍼 ([0,1] 입력)
# =========================
def get_lpips_model(net_type="alex", reduction="mean", normalize=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return TM_LPIPS(net_type=net_type, reduction=reduction, normalize=normalize).to(device).eval()

@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    # x: (...,1,H,W) -> (...,n,H,W)  (expand로 복제)
    return x.expand(*x.shape[:-3], n, x.shape[-2], x.shape[-1])

@torch.no_grad()
def _log_normalize(data: torch.Tensor, c=-0.4, d=103.0, a=0.0, b=1.0):
    """
    선형 정규화→log1p: [c,d]→[a,b]→log1p / log(2) => [0,1] 근사
    (네 기존 lpips 메트릭과 동일한 스케일링)
    """
    scale = (b - a) / (d - c)
    z = ((data - c) * scale) + a
    z = torch.clamp(z, 0.0, 1.0)
    return torch.log1p(z) / math.log(2.0)

@torch.no_grad()
def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
    # (B,1,T,H,W) 보장
    if x.dim() == 5: return x
    if x.dim() == 4: return x.unsqueeze(2)
    if x.dim() == 3: return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"expect 3D/4D/5D, got {tuple(x.shape)}")

# =========================
# 공통 유틸
# =========================
# ---- DTCWT 호환 임포트 슈림 ----
# (에디터/버전 차이로 경로가 달라도 동작하도록)

def to_5d_BCTHW(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5: return x
    if x.ndim == 4: return x.unsqueeze(2)
    if x.ndim == 3: return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported shape {tuple(x.shape)}")

def pct_norm(a: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-8):
    lo, hi = np.percentile(a, [pmin, pmax])
    return (np.clip(a, lo, hi) - lo) / (hi - lo + eps)

def resample_to(a: np.ndarray, out_h: int, out_w: int, order=3, antialias=True):
    in_h, in_w = a.shape
    if antialias and (out_h < in_h or out_w < in_w):
        sy = in_h / out_h; sx = in_w / out_w
        sigy = max(0.0, 0.5 * np.sqrt(max(sy**2 - 1.0, 0.0)))
        sigx = max(0.0, 0.5 * np.sqrt(max(sx**2 - 1.0, 0.0)))
        if sigx > 0 or sigy > 0:
            a = gaussian_filter(a, sigma=(sigy, sigx), mode="nearest")
    return zoom(a, (out_h / a.shape[0], out_w / a.shape[1]), order=order)

'''def imshow_geo(ax, data2d, extent=(0,360,-90,90), central_longitude=180,
               vmin=0, vmax=1, cmap="viridis", title="", tick_labelsize=12,
               flip_vertical=False):
    import numpy as np
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax.set_global()
    ax.coastlines(linewidth=0.6)

    ax.set_xticks(np.arange(-180, 181, 60), crs=proj)
    ax.set_yticks(np.arange(-90,   91, 30), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=tick_labelsize)

    # ★ 여기서 강제로 위아래 뒤집기
    img_data = np.flipud(data2d) if flip_vertical else data2d

    img = ax.imshow(
        img_data, origin="upper",                 # origin 고정
        transform=ccrs.PlateCarree(),
        extent=extent, vmin=vmin, vmax=vmax, cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=13)
    return img
'''



# =========================
# LPIPS 스코어(2D, [0,1] 입력)
# =========================
import torch.nn.functional as F

import math
import torch
import torch.nn.functional as F
import numpy as np

def _ceil_to(v, m):
    return int(math.ceil(v / float(m)) * m)

@torch.no_grad()
def lpips_score_2d_01(pred_2d, gt_2d, lpips_model, percentiles=(1, 99)):
    return lpips_score_global_01(pred_2d, gt_2d, lpips_model, percentiles=percentiles)


def to_5d_BCTHW(x: torch.Tensor):
    if x.dim() == 4:  # (B,C,H,W) -> (B,C,1,H,W)
        return x.unsqueeze(2)
    return x


# =========================
# ① Lv1 해상도 통일 + LPIPS bar
# =========================
import os, math
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from dtcwt.numpy import Transform2d  # 환경에 맞게 import 경로 유지

# ===================== DTCWT 유틸 =====================
def dtcwt_decompose_numpy(arr2d: np.ndarray, levels: int):
    tfm = Transform2d()
    return tfm.forward(arr2d, nlevels=levels)   # pyr.lowpass, pyr.highpasses(list of (H,W,6) complex)

def level_energy_maps(pyr, reduce="l2"):
    Es = []
    for H in pyr.highpasses:
        mag = np.abs(H)
        if reduce == "l2":
            E = np.sqrt((mag**2).sum(axis=-1))
        else:
            E = mag.sum(axis=-1)
        Es.append(E.astype(np.float32, copy=False))
    return Es

def resample_to(arr: np.ndarray, H: int, W: int, order=3, antialias=True):
    import skimage.transform as skt
    return skt.resize(arr, (H, W), order=order, mode="reflect",
                      anti_aliasing=antialias, preserve_range=True).astype(np.float32, copy=False)

def pct_norm(arr: np.ndarray, p_lo=1, p_hi=99):
    lo, hi = np.nanpercentile(arr, [p_lo, p_hi])
    den = max(1e-9, hi - lo)
    return np.clip((arr - lo) / den, 0.0, 1.0).astype(np.float32, copy=False)

# ===================== LPIPS 유틸 =====================
def to_5d_BCTHW(x: torch.Tensor):
    return x.unsqueeze(2) if x.dim() == 4 else x

@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    # x: (...,1,H,W) -> (...,n,H,W)
    return x.expand(*x.shape[:-3], n, x.shape[-2], x.shape[-1])

@torch.no_grad()
def _log_normalize(data: torch.Tensor, c=-0.4, d=103.0, a=0.0, b=1.0):
    scale = (b - a) / (d - c)
    z = ((data - c) * scale) + a
    z = torch.clamp(z, 0.0, 1.0)
    return torch.log1p(z) / math.log(2.0)

def _ceil_to(v, m):
    return int(math.ceil(v / float(m)) * m)

def pick_orient_per_level(pyr_gt) -> list:
    """각 레벨에서 GT 에너지가 최대인 방향 인덱스(0..5)를 반환."""
    orients = []
    for H in pyr_gt.highpasses:  # (H,W,6)
        Es = [np.sum(np.abs(H[..., k])**2) for k in range(6)]
        orients.append(int(np.argmax(Es)))
    return orients  # 예: [3,5,2]

def crop_border(x: np.ndarray, n=12):
    """가장자리 n픽셀 크롭(너무 작으면 원본 반환)."""
    return x[n:-n, n:-n] if (x.shape[0] > 2*n and x.shape[1] > 2*n) else x

# ===================== 메인 함수 =====================
import os, numpy as np, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# --- 필요한 유틸들이 이미 파일에 있다면 재정의 없이 그대로 사용하세요. ---
# pct_norm, crop_border, resample_to, to_5d_BCTHW
# dtcwt_decompose_numpy, pick_orient_per_level, reconstruct_single_high_band

# =========================
# LPIPS (전역, 한 번에 계산)
# =========================
@torch.no_grad()
def _ensure_min_side_for_lpips(img: torch.Tensor, min_side: int = 64):
    """
    img: (N,3,H,W) [0,1]
    Alex/VGG LPIPS는 다운샘플링을 반복하므로 H 또는 W가 너무 작으면 터짐.
    짧은 변이 min_side 미만이면 비율 유지 업샘플.
    """
    _, _, H, W = img.shape
    m = min(H, W)
    if m >= min_side:
        return img
    scale = float(min_side) / float(m)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    return F.interpolate(img, size=(newH, newW), mode="bilinear", align_corners=False)

# ---------- LPIPS 스코어 (전역) ----------

@torch.no_grad()
def lpips_score_global_01(pred_2d, gt_2d, lpips_model, percentiles=(1, 99)):
    """
    pred_2d, gt_2d: (H,W) numpy or torch
    percentiles: (lo, hi)로 [0,1] 정규화 후 LPIPS 계산.
    """
    import torch.nn.functional as F

    # to tensor (float32, no NaN)
    p = torch.from_numpy(pred_2d) if isinstance(pred_2d, np.ndarray) else pred_2d
    g = torch.from_numpy(gt_2d)   if isinstance(gt_2d,   np.ndarray) else gt_2d
    p = torch.nan_to_num(p, nan=0.0).detach().float()
    g = torch.nan_to_num(g, nan=0.0).detach().float()

    # [0,1] 정규화 (퍼센타일)
    def _pct_norm_t(t, lo, hi):
        x_np = t.cpu().numpy()
        lo_v, hi_v = np.nanpercentile(x_np, [lo, hi])
        den = max(1e-9, hi_v - lo_v)
        x = torch.clamp((t - lo_v) / den, 0.0, 1.0)
        return x

    lo, hi = percentiles
    p = _pct_norm_t(p, lo, hi)
    g = _pct_norm_t(g, lo, hi)

    # BCHW + RGB
    p = _expand_gray_to_rgb(p.unsqueeze(0).unsqueeze(0), 3)
    g = _expand_gray_to_rgb(g.unsqueeze(0).unsqueeze(0), 3)

    # 모델 디바이스 정렬
    dev = _model_device(lpips_model)
    lpips_model = lpips_model.to(dev)
    p = p.to(dev, dtype=torch.float32)
    g = g.to(dev, dtype=torch.float32)

    # 최소 해상도 가드(알렉스넷 백본: 대략 32 이상 권장)
    H, W = p.shape[-2], p.shape[-1]
    MIN_HW = 32
    if H < MIN_HW or W < MIN_HW:
        p = F.interpolate(p, size=(max(MIN_HW, H), max(MIN_HW, W)),
                          mode="bilinear", align_corners=False)
        g = F.interpolate(g, size=(max(MIN_HW, H), max(MIN_HW, W)),
                          mode="bilinear", align_corners=False)

    s = lpips_model(p, g)  # (1,) or scalar
    return float(s.detach().mean().item())



# ===========================================
# 그림 1: GT/LL/밴드 + 전역 LPIPS 막대 포함 버전
# ===========================================
@torch.no_grad()
def plot_ll_lv123_with_lpipsbar_same_res(
    gt: torch.Tensor,
    pred: torch.Tensor,
    *,
    lpips_model,
    b=0, c=0, t=0,
    levels=3,
    extent=(0,360,-90,90),
    central_longitude=180,
    ll_percentiles=(1,99),
    high_percentiles=(1,99),
    figsize=(16, 8),
    save_path=None,
    close_after_save=False,
    resample_order=3,
    antialias=True,
    crop_n=12,
    **kwargs,
):
    """
    - 각 레벨별: GT에서 '최대 에너지 방향' 1개를 골라 단일 밴드 역복원
    - **LPIPS는 전역(프레임 한 번)에 대해 계산**하며, 계산 전 경계 crop 적용
    - LL도 동일 규칙으로 전역 LPIPS를 계산해 막대에 포함
    - 시각화는 Lv1 해상도로 업샘플, 배경 흰색
    """
    # ---- 렌더링(흰 배경) ----
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'
    magma_w = cm.get_cmap('magma').copy(); magma_w.set_bad('white',1.0); magma_w.set_under('white',1.0)
    plasma  = cm.get_cmap('plasma').copy(); plasma.set_bad('white',1.0); plasma.set_under('white',1.0)
    viridis = cm.get_cmap('viridis').copy(); viridis.set_bad('white',1.0); viridis.set_under('white',1.0)

    # ---- 데이터 ----
    gt = to_5d_BCTHW(gt); pred = to_5d_BCTHW(pred)
    gt2d   = torch.nan_to_num(gt[b, c, t],   nan=0.0).detach().cpu().numpy()
    pred2d = torch.nan_to_num(pred[b, c, t], nan=0.0).detach().cpu().numpy()
    H, W = pred2d.shape
    H1, W1 = H//2, W//2

    # ---- DTCWT ----
    pyr_gt   = dtcwt_decompose_numpy(gt2d,   levels)
    pyr_pred = dtcwt_decompose_numpy(pred2d, levels)

    # ---- LL (참고용 + LPIPS 계산용) ----
    ll_gt_full   = pyr_gt.lowpass.real.astype(np.float32, copy=False)
    ll_pred_full = pyr_pred.lowpass.real.astype(np.float32, copy=False)

    # ---- 레벨별 단일 밴드(최대 에너지 방향) 역복원 ----
    or_lv = pick_orient_per_level(pyr_gt)  # 길이=levels, 각 원소 0..5
    recon_gt, recon_pr = [], []
    for L in (1,2,3):
        if L > levels:
            recon_gt.append(None); recon_pr.append(None); continue
        recon_gt.append(reconstruct_single_high_band(gt2d,   levels, L, or_lv[L-1]))
        recon_pr.append(reconstruct_single_high_band(pred2d, levels, L, or_lv[L-1]))

    # ---- 전역 LPIPS (경계 crop 후 한 번 계산) ----
    # 1) LL
    ll_lpips = lpips_score_global_01(
        pred_2d=crop_border(ll_pred_full, crop_n),
        gt_2d  =crop_border(ll_gt_full,   crop_n),
        lpips_model=lpips_model,
        percentiles=ll_percentiles
    )

    # 2) Lv1~Lv3
    level_names, lpips_vals = [], []
    for i, L in enumerate((1,2,3)):
        level_names.append(f"Lv{L}")
        if recon_pr[i] is None:
            lpips_vals.append(np.nan); continue
        lp = lpips_score_global_01(
            pred_2d=crop_border(recon_pr[i], crop_n),
            gt_2d  =crop_border(recon_gt[i], crop_n),
            lpips_model=lpips_model,
            percentiles=high_percentiles
        )
        lpips_vals.append(float(lp))

    # ---- 시각화용 업샘플/공통 스케일 ----
    up = lambda x: resample_to(x, H1, W1, order=resample_order, antialias=antialias)
    ll_vis  = pct_norm(resample_to(ll_pred_full, H1, W1, resample_order, antialias), *ll_percentiles)
    gt_vis_full = pct_norm(gt2d, *ll_percentiles)

    vis_h = []
    for i in range(3):
        if recon_pr[i] is None: vis_h.append(None); continue
        vis_h.append(pct_norm(up(recon_pr[i]), *high_percentiles))

    joint = np.concatenate([v.ravel() for v in vis_h if v is not None]) if any(v is not None for v in vis_h) else np.array([0,1])
    vmin, vmax = np.nanpercentile(joint, [1, 99]) if joint.size>1 else (0.0, 1.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(joint)), float(np.nanmax(joint) + 1e-9)

    # ---- Figure ----
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig = plt.figure(figsize=figsize, dpi=150)

    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray")
        lon = np.arange(-180, 181, 60); lat = np.arange(-90, 91, 30)
        ax.set_xticks(lon, crs=ccrs.PlateCarree()); ax.set_yticks(lat, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, linestyle="dashed", color="gray", alpha=0.6)
        gl.xlocator = mticker.FixedLocator(lon); gl.ylocator = mticker.FixedLocator(lat)

    # (a) GT
    ax_a = plt.subplot(2, 3, 1, projection=proj)
    im_a = ax_a.imshow(np.ma.masked_invalid(gt_vis_full), origin="lower", extent=extent,
                       transform=ccrs.PlateCarree(), cmap=plasma, vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    ax_a.set_title("(a) GT"); _decorate(ax_a)

    # (b) LL (Pred)
    ax_b = plt.subplot(2, 3, 2, projection=proj)
    im_b = ax_b.imshow(np.ma.masked_invalid(ll_vis), origin="lower", extent=extent,
                       transform=ccrs.PlateCarree(), cmap=viridis, vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    ax_b.set_title("(b) LL"); _decorate(ax_b)

    # (c) LPIPS bar (LL + Lv1/2/3)
    ax_c = plt.subplot(2, 3, 3)
    bar_labels = ["LL"] + level_names
    bar_vals   = [ll_lpips] + lpips_vals
    x = np.arange(len(bar_labels))
    ax_c.bar(x, bar_vals, width=0.45)
    ax_c.set_xticks(x); ax_c.set_xticklabels(bar_labels)
    ax_c.set_title(f"(c) LPIPS (↓)  |  global, crop={crop_n}px")
    ax_c.set_ylabel("LPIPS"); ax_c.grid(axis="y", linestyle=":", linewidth=0.5)

    # (d)(e)(f) Lv1/Lv2/Lv3 단일 밴드 (Pred)
    titles = [f"(d) Lv1 (single band, ori={or_lv[0] if levels>=1 else 'N/A'})",
              f"(e) Lv2 (single band, ori={or_lv[1] if levels>=2 else 'N/A'})",
              f"(f) Lv3 (single band, ori={or_lv[2] if levels>=3 else 'N/A'})"]
    for i in range(3):
        ax = plt.subplot(2, 3, 4+i, projection=proj)
        if vis_h[i] is None:
            ax.axis("off"); ax.set_title(titles[i] + " (N/A)"); continue
        im = ax.imshow(np.ma.masked_invalid(vis_h[i]), origin="lower", extent=extent,
                       transform=ccrs.PlateCarree(), cmap=magma_w, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(titles[i]); _decorate(ax)

    fig.suptitle("GT / LL / LPIPS bar  |  Per-level max-energy orient (GLOBAL LPIPS with border crop)")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight"); print(f"[Saved] {save_path}")
        if close_after_save:
            plt.close(fig)

    return fig, {
        "labels": ["LL"] + level_names,
        "lpips":  [ll_lpips] + lpips_vals,
        "orients": or_lv
    }


#============LPIPS hit map & CDF plot====================================
import os, math
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from dtcwt.numpy.transform2d import Transform2d

def resample_to(arr, H, W, order=3, antialias=True):
    import skimage.transform as skt
    return skt.resize(arr, (H, W), order=order, mode="reflect",
                      anti_aliasing=antialias, preserve_range=True).astype(np.float32, copy=False)

def pct_norm(arr, p_lo=1, p_hi=99):
    lo, hi = np.nanpercentile(arr, [p_lo, p_hi])
    den = max(1e-9, hi - lo)
    return np.clip((arr - lo) / den, 0.0, 1.0).astype(np.float32, copy=False)

def crop_border(x: np.ndarray, n=12):
    return x[n:-n, n:-n] if (x.shape[0] > 2*n and x.shape[1] > 2*n) else x

# =========================
# LPIPS helpers (질문자 파이프라인 그대로)
# =========================
@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    # x: (...,1,H,W) -> (...,n,H,W)
    return x.expand(*x.shape[:-3], n, x.shape[-2], x.shape[-1])

@torch.no_grad()
def _log_normalize(data: torch.Tensor, c=-0.4, d=103.0, a=0.0, b=1.0):
    scale = (b - a) / (d - c)
    z = ((data - c) * scale) + a
    z = torch.clamp(z, 0.0, 1.0)
    return torch.log1p(z) / math.log(2.0)


@torch.no_grad()
def lpips_scalar_from_maps(lp_map: np.ndarray, mask_zero=False) -> float:
    """LPIPS 히트맵의 평균을 스칼라로. mask_zero=True면 0(스캔 외곽) 무시."""
    x = lp_map
    if mask_zero:
        x = x[x > 0]
    if x.size == 0:
        return 0.0
    return float(np.mean(x))

# =========================
# 지도 데코(폰트/크기: rcParams 따름 → visualize_prediction_map과 동일 스타일)
# =========================
def _decorate_geo_axes(ax, add_gridlines=True):
    ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
    lon_ticks = np.arange(-180, 181, 60)
    lat_ticks = np.arange(-90, 91, 30)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", which="major")
    if add_gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, linestyle="dashed",
                          color="gray", alpha=0.6, zorder=6)
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
    ax.set_global()

@torch.no_grad()
def _scale01_pair(P2: torch.Tensor, G2: torch.Tensor,
                  p_lo: float = 0.5, p_hi: float = 99.5, mode: str = "joint") -> tuple:
    """
    P2, G2: (H,W) torch.Tensor (float32), 같은 device
    mode = "joint"  : GT+Pred 합친 분포의 퍼센타일로 스케일 (권장)
         = "gt_only": GT만으로 스케일(보수적)
    반환: (P01, G01) in [0,1]
    """
    import torch
    eps = 1e-8
    P2 = torch.nan_to_num(P2, nan=0.0)
    G2 = torch.nan_to_num(G2, nan=0.0)

    if mode == "joint":
        both = torch.stack([P2.reshape(-1), G2.reshape(-1)], dim=0).reshape(-1)
        lo = torch.quantile(both, p_lo / 100.0)
        hi = torch.quantile(both, p_hi / 100.0)
    else:  # "gt_only"
        lo = torch.quantile(G2.reshape(-1), p_lo / 100.0)
        hi = torch.quantile(G2.reshape(-1), p_hi / 100.0)

    if (hi - lo) < eps:
        # 평탄한 경우: 전부 0으로
        return torch.zeros_like(P2), torch.zeros_like(G2)

    def _mm(x):  # min-max to [0,1]
        x = (x - lo) / (hi - lo + eps)
        return x.clamp(0.0, 1.0)

    return _mm(P2), _mm(G2)


@torch.no_grad()
def _lpips_scalar_global(pred_2d, gt_2d, lpips_model, crop_n=0,
                         p_lo: float = 0.5, p_hi: float = 99.5, mode: str = "joint",
                         min_hw: tuple = None):
    """
    pred_2d, gt_2d: (H,W) np.ndarray 또는 torch.Tensor
    LPIPS(torchmetrics) normalize=True → 입력은 [0,1]이어야 함
    min_hw: (min_h, min_w). None이면 백본에 맞춰 자동 결정.
    """
    import torch, numpy as np
    import torch.nn.functional as F

    # 모델 디바이스
    try:
        device = next(lpips_model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 백본에 따른 최소 해상도
    if min_hw is None:
        # torchmetrics의 LPIPS는 .net에 백본 이름이 없습니다. 사용자가 net_type을 알고 있다고 가정.
        # 안전한 기본값: alex → 64x64, vgg → 96x96
        # 필요 시 호출부에서 명시적으로 min_hw=(64,64) 또는 (96,96) 전달하세요.
        min_hw = (64, 64)

    def to_tensor_2d(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            raise TypeError(f"expected np.ndarray or torch.Tensor, got {type(x)}")
        x = x.squeeze()
        if x.dim() != 2:
            raise ValueError(f"expected (H,W), got {tuple(x.shape)}")
        return x.to(device=device, dtype=torch.float32)

    P2 = to_tensor_2d(pred_2d)
    G2 = to_tensor_2d(gt_2d)

    # 경계 크롭(너무 크면 예외)
    if crop_n and crop_n > 0:
        if (P2.size(-2) <= 2*crop_n) or (P2.size(-1) <= 2*crop_n):
            raise ValueError(f"crop_n={crop_n} too large for {tuple(P2.shape)}")
        P2 = P2[crop_n:-crop_n, crop_n:-crop_n]
        G2 = G2[crop_n:-crop_n, crop_n:-crop_n]

    # [0,1] 정규화 (joint-percentile)
    def _scale01_pair(P2, G2, p_lo, p_hi, mode):
        eps = 1e-8
        P2 = torch.nan_to_num(P2, nan=0.0)
        G2 = torch.nan_to_num(G2, nan=0.0)
        if mode == "joint":
            both = torch.stack([P2.reshape(-1), G2.reshape(-1)], dim=0).reshape(-1)
            lo = torch.quantile(both, p_lo / 100.0)
            hi = torch.quantile(both, p_hi / 100.0)
        else:
            lo = torch.quantile(G2.reshape(-1), p_lo / 100.0)
            hi = torch.quantile(G2.reshape(-1), p_hi / 100.0)
        if (hi - lo) < eps:
            return torch.zeros_like(P2), torch.zeros_like(G2)
        def mm(x): return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
        return mm(P2), mm(G2)

    P01, G01 = _scale01_pair(P2, G2, p_lo, p_hi, mode)

    # ★ 최소 해상도 업샘플(양 변이 min_h/min_w 이상이 되도록) ★
    def upsample_min(x2, min_h, min_w):
        H, W = int(x2.shape[-2]), int(x2.shape[-1])
        H2, W2 = max(H, min_h), max(W, min_w)
        if (H2 != H) or (W2 != W):
            x2 = F.interpolate(x2[None, None], size=(H2, W2),
                               mode="bilinear", align_corners=False)[0, 0]
        return x2

    P01 = upsample_min(P01, min_hw[0], min_hw[1])
    G01 = upsample_min(G01, min_hw[0], min_hw[1])

    # 1x3xHxW로 확장
    def to3(x2): return x2.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).contiguous()
    P3, G3 = to3(P01), to3(G01)

    # LPIPS 계산
    score = lpips_model(P3, G3)
    return float(score.reshape(()).item())





# ====== 새 시각화 함수 (막대 폭만 축소) ======
@torch.no_grad()
def show_lpips_band_evidence_ll_compact_bar(
    gt: torch.Tensor, pred: torch.Tensor, lpips_model,
    *,
    b=0, c=0, t=0,
    levels=3,
    extent=(0,360,-90,90),
    central_longitude=180,
    crop_n=12,
    ll_percentiles=(1,99), high_percentiles=(1,99),
    figsize=(16, 6.5),
    save_path=None, dpi=300,
    add_gridlines=True,
    bar_width=0.45,                 # 👈 막대 가로 폭만 조절 (기본: 0.45)
    bar_color="steelblue"
):
    # 흰 배경 유지
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'
    magma_r_white = cm.get_cmap('magma_r').copy()
    magma_r_white.set_bad('white', 1.0); magma_r_white.set_under('white', 1.0)

    # 입력 정리
    def to5d(x): return x.unsqueeze(2) if x.dim()==4 else x
    gt   = to5d(gt); pred = to5d(pred)
    gt2d   = torch.nan_to_num(gt[b, c, t],   nan=0.0).float().cpu().numpy()
    pred2d = torch.nan_to_num(pred[b, c, t], nan=0.0).float().cpu().numpy()

    # DTCWT 및 방향 선택
    _, pyr_gt = dtcwt_forward(gt2d, levels=levels)
    or_lv = pick_orient_per_level(pyr_gt)

    # 레벨별 single-band 역복원 (GT, Pred)
    recon_gt, recon_pr = [], []
    for L in (1,2,3):
        if L > levels:
            recon_gt.append(None); recon_pr.append(None); continue
        recon_gt.append(reconstruct_single_high_band(gt2d,   levels, L, or_lv[L-1]))
        recon_pr.append(reconstruct_single_high_band(pred2d, levels, L, or_lv[L-1]))

    # LL-only (lowpass)
    rg_ll = pyr_gt.lowpass.real.astype(np.float32, copy=False)
    rp_ll = dtcwt_forward(pred2d, levels=levels)[1].lowpass.real.astype(np.float32, copy=False)

    # LPIPS 스칼라 (crop → patch-map 평균)
    bar_vals = [
        _lpips_scalar_global(rp_ll,           rg_ll,           lpips_model, crop_n=crop_n),
        _lpips_scalar_global(recon_pr[0],     recon_gt[0],     lpips_model, crop_n=crop_n) if recon_pr[0] is not None else np.nan,
        _lpips_scalar_global(recon_pr[1],     recon_gt[1],     lpips_model, crop_n=crop_n) if recon_pr[1] is not None else np.nan,
        _lpips_scalar_global(recon_pr[2],     recon_gt[2],     lpips_model, crop_n=crop_n) if recon_pr[2] is not None else np.nan,
    ]

    # 시각화용 맵 (퍼센타일 정규화)
    maps_for_plot = [
        pct_norm(rp_ll, *ll_percentiles),
        pct_norm(recon_pr[0], *high_percentiles) if recon_pr[0] is not None else None,
        pct_norm(recon_pr[1], *high_percentiles) if recon_pr[1] is not None else None,
        pct_norm(recon_pr[2], *high_percentiles) if recon_pr[2] is not None else None,
    ]
    titles = ["(a) LL", f"(b) Lv1 (ori={or_lv[0]})", f"(c) Lv2 (ori={or_lv[1]})", f"(d) Lv3 (ori={or_lv[2]})"]

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon = np.arange(-180, 181, 60); lat = np.arange(-90, 91, 30)
        ax.set_xticks(lon, crs=ccrs.PlateCarree()); ax.set_yticks(lat, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
        if add_gridlines:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.5, linestyle="dashed", color="gray", alpha=0.6)
            gl.xlocator = mticker.FixedLocator(lon); gl.ylocator = mticker.FixedLocator(lat)

    # 상단 4개 지도
    for i in range(4):
        ax = plt.subplot(2, 4, 1+i, projection=proj)
        if maps_for_plot[i] is None:
            ax.axis("off"); ax.set_title(titles[i] + " (N/A)"); continue
        im = ax.imshow(maps_for_plot[i], origin="lower", extent=extent,
                       transform=ccrs.PlateCarree(), cmap=magma_r_white, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(titles[i])
        _decorate(ax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("normed value")

    # 하단 바그래프 (막대 너비만 축소)
    axb = plt.subplot(2, 4, 5)
    labels = ["LL", "Lv1", "Lv2", "Lv3"]; x = np.arange(len(labels))
    axb.bar(x, bar_vals, width=bar_width, color=bar_color)  # 👈 폭만 줄임
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set_title("(e) LPIPS bar (mean over map, border-cropped)")
    axb.set_ylabel("LPIPS")
    axb.grid(axis="y", linestyle=":", linewidth=0.5)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    return {
        "orients": or_lv,
        "bar_vals": dict(zip(["LL","Lv1","Lv2","Lv3"], bar_vals))
    }
# =========================
# 메인: 레벨별 LPIPS 맵 + LL LPIPS + 막대그래프
# =========================
@torch.no_grad()
def show_lpips_band_evidence_ll(
    gt: torch.Tensor, pred: torch.Tensor, lpips_model,
    *,
    b=0, c=0, t=0,
    levels=3,
    extent=(0,360,-90,90),
    central_longitude=180,
    crop_n=12,
    ll_percentiles=(1,99),
    high_percentiles=(1,99),
    patch=64, stride=32,
    figsize=(16, 6.5),
    save_path=None, dpi=300,
    add_gridlines=True,
    # ▼ NEW: bar 레이아웃/스타일
    bar_width=0.35,
    bar_color="steelblue",
    bar_span_cols=2,      # 아래줄에서 몇 칸을 차지할지 (1~4)
    bar_start_col=1,      # 몇 번째 칸부터 시작할지 (0~3, 범위 벗어나면 자동 보정)
):
    import os, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import cartopy.crs as ccrs
    import torch

    # ---- 스타일 ----
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'
    magma_r_white = cm.get_cmap('magma_r').copy()
    magma_r_white.set_bad('white', 1.0); magma_r_white.set_under('white', 1.0)

    def to5d(x):  # (B,1,H,W) -> (B,1,1,H,W)
        return x.unsqueeze(2) if x.dim()==4 else x

    def _decorate_geo_axes(ax, add_gridlines=True):
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        if add_gridlines:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.5, linestyle="dashed", color="gray", alpha=0.6)
            gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 181, 60))
            gl.ylocator = mpl.ticker.FixedLocator(np.arange(-90, 91, 30))

    # ---- 데이터/디바이스 ----
    gt   = to5d(gt); pred = to5d(pred)
    gt2d   = torch.nan_to_num(gt[b, c, t],   nan=0.0).float()
    pred2d = torch.nan_to_num(pred[b, c, t], nan=0.0).float()
    lpips_model = lpips_model.to(pred2d.device)

    # ---- 분해/방향 선택 ----
    g_np = gt2d.detach().cpu().numpy()
    p_np = pred2d.detach().cpu().numpy()
    _, pyr_gt = dtcwt_forward(g_np, levels=levels)
    or_lv = pick_orient_per_level(pyr_gt)

    # ---- 밴드별 역복원 ----
    recon_gt, recon_pr = [], []
    for L in (1,2,3):
        if L > levels:
            recon_gt.append(None); recon_pr.append(None); continue
        recon_gt.append(reconstruct_single_high_band(g_np, levels, L, or_lv[L-1]))
        recon_pr.append(reconstruct_single_high_band(p_np, levels, L, or_lv[L-1]))

    # ---- LL-only ----
    rg_ll = reconstruct_lowpass_only(g_np, levels).astype(np.float32, copy=False)
    rp_ll = reconstruct_lowpass_only(p_np, levels).astype(np.float32, copy=False)

    # ---- 패치 LPIPS 맵: LL + Lv1~3 ----
    maps, titles = [], ["(a) LL"]
    # LL
    gt_ll_t = torch.from_numpy(crop_border(rg_ll, crop_n)).to(pred2d.device)
    pr_ll_t = torch.from_numpy(crop_border(rp_ll, crop_n)).to(pred2d.device)
    lp_ll_map = lpips_patch_map_single_frame(pr_ll_t, gt_ll_t, lpips_model,
                                             patch=patch, stride=stride, lognorm=True)
    maps.append(lp_ll_map)
    # Highs
    for i in range(3):
        if recon_pr[i] is None:
            maps.append(None); titles.append(f"({chr(ord('b')+i)}) Lv{i+1} (ori=N/A)")
            continue
        gt_crop = crop_border(recon_gt[i], crop_n)
        pr_crop = crop_border(recon_pr[i], crop_n)
        gt_t = torch.from_numpy(gt_crop).to(pred2d.device)
        pr_t = torch.from_numpy(pr_crop).to(pred2d.device)
        lp_map = lpips_patch_map_single_frame(pr_t, gt_t, lpips_model,
                                              patch=patch, stride=stride, lognorm=True)
        maps.append(lp_map)
        titles.append(f"({chr(ord('b')+i)}) Lv{i+1} (ori={or_lv[i]})")

    # ---- 바 값(맵 평균; 0 제외) ----
    bar_vals = [lpips_scalar_from_maps(m, mask_zero=True) if m is not None else np.nan
                for m in maps]

    # ---- 컬러 스케일(공통) ----
    joint_vals = []
    for m in maps:
        if m is None: continue
        v = m[m > 0]
        if v.size: joint_vals.append(v.ravel())
    joint_vals = np.concatenate(joint_vals) if joint_vals else np.array([1.0])
    vmin_lp = max(float(np.nanpercentile(joint_vals, 1.0)), 1e-12)
    vmax_lp = float(np.nanpercentile(joint_vals, 99.5))

    # ---- Figure & GridSpec ----
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        height_ratios=[3.2, 1.2],   # 윗줄:맵, 아랫줄:바플롯
        wspace=0.25, hspace=0.35
    )

    # 윗줄 4개 지도
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    for i in range(4):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        if maps[i] is not None:
            im = ax.imshow(np.ma.masked_less_equal(maps[i], vmin_lp),
                           origin="lower", extent=extent, transform=proj,
                           cmap=magma_r_white, vmin=vmin_lp, vmax=vmax_lp, interpolation="nearest")
            _decorate_geo_axes(ax, add_gridlines=add_gridlines)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            _decorate_geo_axes(ax, add_gridlines=add_gridlines)
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(titles[i])

    # 아랫줄: 바플롯(가운데만 좁게)
    # 범위 보정
    bar_span_cols = max(1, min(4, int(bar_span_cols)))
    bar_start_col = max(0, min(4 - bar_span_cols, int(bar_start_col)))
    # ✅ 한 칸만 사용할 때는 단일 인덱스, 그 외에는 슬라이스
    if bar_span_cols == 1:
        axb = fig.add_subplot(gs[1, bar_start_col])
    else:
        axb = fig.add_subplot(gs[1, bar_start_col:bar_start_col + bar_span_cols])

    labels = ["LL", "Lv1", "Lv2", "Lv3"]
    axb.bar(labels, bar_vals, width=bar_width, color=bar_color)
    axb.set_title("(e) LPIPS bar (Patch means)")
    axb.set_ylabel("LPIPS")
    axb.grid(axis="y", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return {
        "orients": or_lv,
        "bar_vals": {"LL_Patch":bar_vals[0], "Lv1_Patch":bar_vals[1],
                     "Lv2_Patch":bar_vals[2], "Lv3_Patch":bar_vals[3]},
        "vmin_vmax_lp_patch": (vmin_lp, vmax_lp),
    }





#==========LPIPS map===============
# ===================== LPIPS MAP VISUALIZER =====================
# 요구: pip install lpips cartopy  (cartopy는 이전 CPBD 시각화와 동일)
# pred/target: (B,1,T,H,W) 또는 (B,1,H,W)
# ===============================================================
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


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

# ----- 보조 유틸: 혹시 없을 수 있으니 최소 버전 제공 -----
@torch.no_grad()
def _ensure_5d(x: torch.Tensor):
    # (B,1,T,H,W) OK / (B,1,H,W) -> (B,1,1,H,W)
    if x.dim() == 4:
        return x.unsqueeze(2)
    return x

def _nan_guard_scalar(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    try:
        xv = float(x)
        if math.isnan(xv) or math.isinf(xv):
            return 0.0
        return xv
    except Exception:
        return 0.0


# ===================== 시각화(단일 예측 vs GT) =====================
@torch.no_grad()
def plot_lpips_pair_from_tensors(
    gt, pred, model,
    b=0, c=0, t=0,
    extent=(0, 360, -90, 90),
    central_longitude=180,
    patch=32, stride=16,
    pred_title="MSE",          # 우측 패널 제목 라벨
    save_path=None,
    title_fontsize=19,
    tick_labelsize=13,
    dpi=300,
    cmap="viridis",
    # ▼ LPIPS 패널 전용: 이 값 이하(<=)는 흰색으로 처리
    lpips_floor_percentile=1.0,   # 하위 p%를 배경 취급(예: 1.0). 0으로 두면 끔.
    lpips_floor_abs=None,         # 절대 임계치(예: 1e-3). 지정 시 percentile보다 우선.
):
    """
    CPBD 2패널 형식에 맞춘 LPIPS 시각화:
      (a) GT (0.5–99.5% 정규화 원장)
      (b) Pred vs GT LPIPS 패치 heatmap (작은 값은 흰색으로 배경 처리)

    gt/pred : (B,1,T,H,W) 또는 (B,1,H,W)
    model   : LPIPS 인스턴스
    """
    import os, numpy as np, matplotlib.pyplot as plt, torch, math
    import matplotlib as mpl
    from matplotlib import cm
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.ticker as mticker

    # ----- 스타일: 저장/그림 배경 흰색(데이터 컬러맵은 기존 유지) -----
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'

    def _ensure_5d(x):
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError("empty list passed")
            if torch.is_tensor(x[0]):
                x = torch.stack(x, dim=0)
            else:
                x = torch.from_numpy(np.asarray(x))
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not torch.is_tensor(x):
            raise TypeError(f"unsupported type: {type(x)}")
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B,1,H,W) → (B,1,1,H,W)
        return x

    def _expand_gray_to_rgb(x, n=3):
        return x.expand(-1, n, -1, -1)

    def _log_normalize(data, c=-0.4, d=103.0, a=0.0, b=1.0):
        scale = (b - a) / (d - c)
        z = ((data - c) * scale) + a
        z = z.clamp(min=0.0, max=1.0)
        return torch.log1p(z) / math.log(2.0)

    @torch.no_grad()
    def _lpips_patch_map_single(pr_1ch, gt_1ch, model, patch=32, stride=16):
        device = pr_1ch.device
        H, W = pr_1ch.shape
        acc = torch.zeros((H, W), device=device, dtype=torch.float32)
        cnt = torch.zeros((H, W), device=device, dtype=torch.float32)

        # NaN 안전
        pr_1ch = torch.nan_to_num(pr_1ch, nan=0.0)
        gt_1ch = torch.nan_to_num(gt_1ch, nan=0.0)

        for y0 in range(0, H - patch + 1, stride):
            for x0 in range(0, W - patch + 1, stride):
                y1, x1 = y0 + patch, x0 + patch
                p = pr_1ch[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0)
                g = gt_1ch[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0)
                p = _expand_gray_to_rgb(_log_normalize(p), 3)
                g = _expand_gray_to_rgb(_log_normalize(g), 3)
                s = model(p, g).detach().mean().item()
                acc[y0:y1, x0:x1] += float(s)
                cnt[y0:y1, x0:x1] += 1.0

        lp_map = acc / torch.clamp_min(cnt, 1.0)
        if (cnt == 0).any():
            fillv = float(torch.nan_to_num(lp_map, nan=0.0).mean().item())
            lp_map[cnt == 0] = fillv
        return torch.nan_to_num(lp_map, nan=0.0).detach().cpu().numpy()

    # ---------- 데이터 ----------
    gt   = _ensure_5d(gt)
    pred = _ensure_5d(pred)
    dev = pred[b, c, t].device
    model = model.to(dev)

    gt_np   = torch.nan_to_num(gt[b, c, t],   nan=0.0).detach().cpu().numpy()
    pred_t  = torch.nan_to_num(pred[b, c, t], nan=0.0).detach()

    # GT 정규화 (0.5–99.5%)
    lo = float(np.nanpercentile(gt_np, 0.5))
    hi = float(np.nanpercentile(gt_np, 99.5))
    gt_vis = np.clip((gt_np - lo) / max(1e-9, (hi - lo)), 0.0, 1.0)

    # LPIPS 패치 맵
    lp_map = _lpips_patch_map_single(pred_t, torch.from_numpy(gt_np).to(pred_t.device), model,
                                     patch=patch, stride=stride)

    # LPIPS 패널의 상단 범위
    vmin_raw = 0.0
    vmax = float(np.nanpercentile(lp_map.ravel(), 99.5))
    if np.isclose(vmin_raw, vmax):
        vmax = vmin_raw + 1e-6

    # ---------- 여기부터 LPIPS 패널만 "흰색 바탕" 처리 ----------
    # 1) 컬러맵 복제 + 작은 값/NaN을 흰색으로
    magma_white = cm.get_cmap('magma').copy()
    magma_white.set_bad('white', 1.0)     # 마스크/NaN → 흰색
    magma_white.set_under('white', 1.0)   # vmin 이하 → 흰색

    # 2) 임계치 tau 결정: 절대값 우선, 없으면 하위 백분위로
    if lpips_floor_abs is not None:
        tau = float(lpips_floor_abs)
    elif (lpips_floor_percentile is not None) and (lpips_floor_percentile > 0.0):
        tau = float(np.nanpercentile(lp_map, lpips_floor_percentile))
    else:
        tau = vmin_raw  # 비활성화

    # 3) 시각화용 마스크 배열: tau 이하는 마스크 → 흰색으로 렌더링
    if tau > vmin_raw:
        lp_vis = np.ma.masked_less_equal(lp_map, tau)
        vmin_for_plot = tau
    else:
        lp_vis = lp_map
        vmin_for_plot = vmin_raw
    # ---------- LPIPS 패널 전용 처리 끝 ----------

    # ---------- Figure ----------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             subplot_kw=dict(projection=proj),
                             constrained_layout=True,
                             facecolor='white')

    # 축 배경은 흰색(데이터의 낮은 값은 위에서 흰색 처리됨)
    for ax in axes:
        ax.set_facecolor("white")

    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, linestyle="dashed",
                          color="gray", alpha=0.6, zorder=6)
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
        ax.tick_params(axis="both", labelsize=tick_labelsize)

    Hh, Ww = gt_vis.shape
    lon = np.linspace(extent[0], extent[1], Ww)
    lat = np.linspace(extent[2], extent[3], Hh)

    # (a) GT  — 기존 스타일 유지
    axes[0].pcolormesh(
        lon, lat, gt_vis, cmap="plasma", vmin=0.0, vmax=1.0,
        transform=ccrs.PlateCarree(), shading="auto"
    )
    axes[0].set_title("(a) GT", fontsize=title_fontsize)
    _decorate(axes[0])

    # (b) LPIPS — 작은 값 흰색/마스크 적용
    im1 = axes[1].pcolormesh(
        lon, lat, lp_vis, cmap=magma_white,
        vmin=vmin_for_plot, vmax=vmax,
        transform=ccrs.PlateCarree(), shading="auto"
    )
    axes[1].set_title(f"(a) {pred_title}", fontsize=title_fontsize)
    _decorate(axes[1])

    cbar = fig.colorbar(im1, ax=axes, orientation="vertical",
                        shrink=0.9, fraction=0.046, pad=0.04)
    cbar.set_label("LPIPS", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi,
                    facecolor="white", edgecolor="white", transparent=False)
        plt.close(fig)
    else:
        plt.show()




##################LPIPS 차이맵
# lpips_delta_runner.py
import os, math, numpy as np, torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

# --- 네 프로젝트 유틸을 그대로 사용 ---
from utils.config_utils import load_config
from datasets.dataloader import get_test_loader
from utils.save_utils import load_checkpoint_for_eval

@torch.no_grad()
def make_lpips_delta_map(
    cfg_mse_path: str,
    cfg_facl_path: str,
    ckpt_mse_path: str,
    ckpt_facl_path: str,
    out_png: str,
    sample_idx: int = 0,   # b
    ch_idx: int = 0,       # c
    t_idx: int = 0,        # t
    central_longitude: float = 180.0,
    extent = (0, 360, -90, 90),
    patch: int = 32,
    stride: int = 16,
    title_a: str = "FACL",
    title_b: str = "MSE",
    add_gridlines: bool = False,
    title_fontsize: int = 19,
    tick_labelsize: int = 13,
    dpi: int = 300,
):
    """
    (a) GT, (b) LPIPS(FACL,GT), (c) Δ=LPIPS(FACL,GT)−LPIPS(MSE,GT) 3-패널 저장.
    - 배치 키가 {'input','target'} 또는 {'X1','X2','Y','time'} 모두 동작.
    - 두 입력 모델(model(x1,x2))도 처리.
    - 흰 배경/NaN/투명/다크스타일 이슈 방어 포함.
    """
    import os, math, numpy as np, torch
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import matplotlib.ticker as mticker
    import lpips

    # ---- 프로젝트 유틸/모델 ----
    from utils.config_utils import load_config
    from utils.save_utils    import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model    import build_model

    # ---------- 전역 렌더링(흰 배경 강제) ----------
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'

    # NaN/bad 색을 흰색으로 지정한 colormap 복제
    plasma = cm.get_cmap('plasma').copy(); plasma.set_bad(color='white', alpha=1.0)
    magma  = cm.get_cmap('magma').copy();  magma.set_bad(color='white',  alpha=1.0)
    bwr    = cm.get_cmap('bwr').copy();    bwr.set_bad(color='white',    alpha=1.0)

    # ---------- 헬퍼 ----------
    def _extract_xy(batch, device):
        """
        배치에서 (inputs, target)을 반환.
        - dict: 'input/target' 또는 'X1','X2','Y','time' 모두 지원
        - inputs는 텐서 또는 (x1, x2) 튜플로 반환
        """
        if isinstance(batch, dict):
            # (A) 일반 단일 입력
            if any(k in batch for k in ("input","inputs","x","src","data","image","images")) and \
               any(k in batch for k in ("target","targets","y","label","labels")):
                for k_in in ("input","inputs","x","src","data","image","images"):
                    if k_in in batch: x = batch[k_in]; break
                for k_tg in ("target","targets","y","label","labels"):
                    if k_tg in batch: y = batch[k_tg]; break
                return x.to(device), y.to(device)

            # (B) 두 입력 케이스: X1, X2, Y, time
            if all(k in batch for k in ("X1","X2","Y")):
                x1 = batch["X1"].to(device)
                x2 = batch["X2"].to(device)
                y  = batch["Y"].to(device)
                return (x1, x2), y

            keys = list(batch.keys())
            raise KeyError(f"Batch dict에서 입력/정답 키를 찾지 못했습니다. keys={keys}")

        if isinstance(batch, (list, tuple)):
            if len(batch) != 2:
                raise ValueError(f"Batch sequence length must be 2, got {len(batch)}")
            x, y = batch
            if isinstance(x, (list, tuple)):
                x = tuple(_x.to(device) for _x in x)
            else:
                x = x.to(device)
            return x, y.to(device)

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _forward_model(model, x):
        """입력이 텐서면 model(x), (x1,x2) 튜플이면 model(x1,x2)."""
        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError(f"Expected two inputs (x1,x2), got {len(x)}")
            x1, x2 = x
            return model(x1, x2)
        return model(x)

    def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
        """(B,1,T,H,W) 형태 보장 (4D면 T 차원 추가)."""
        if x.dim() == 5: return x
        if x.dim() == 4: return x.unsqueeze(2)
        raise ValueError(f"expect 4D/5D, got {x.shape}")

    def _norm01_by_ref(x2d: torch.Tensor, ref2d: torch.Tensor, lo=0.5, hi=99.5) -> torch.Tensor:
        r = ref2d.detach().cpu().numpy()
        lo_v, hi_v = np.percentile(r, [lo, hi]); den = max(1e-9, hi_v - lo_v)
        z = (x2d - lo_v) / den
        return z.clamp_(0.0, 1.0)

    def _to_rgb_m11(x01_2d: torch.Tensor) -> torch.Tensor:
        x01 = x01_2d.unsqueeze(0).unsqueeze(0)     # (1,1,H,W)
        x01 = x01.expand(-1, 3, -1, -1)            # (1,3,H,W)
        return x01.mul(2.0).sub(1.0)               # [0,1]→[-1,1]

    def _lpips_patch_map(lpips_model, pr_1ch: torch.Tensor, gt_1ch: torch.Tensor,
                         patch=32, stride=16) -> np.ndarray:
        """패치 단위 LPIPS를 stride로 스캔 후 평균 맵 생성."""
        Hh, Ww = pr_1ch.shape
        acc = torch.zeros((Hh, Ww), device=pr_1ch.device)
        cnt = torch.zeros((Hh, Ww), device=pr_1ch.device)

        pr01_full = _norm01_by_ref(pr_1ch, gt_1ch)
        gt01_full = _norm01_by_ref(gt_1ch, gt_1ch)

        for y0 in range(0, Hh - patch + 1, stride):
            for x0 in range(0, Ww - patch + 1, stride):
                y1, x1 = y0 + patch, x0 + patch
                p = _to_rgb_m11(pr01_full[y0:y1, x0:x1])
                g = _to_rgb_m11(gt01_full[y0:y1, x0:x1])
                s = lpips_model(p, g).mean()
                acc[y0:y1, x0:x1] += s
                cnt[y0:y1, x0:x1] += 1.0

        lp_map = acc / torch.clamp_min(cnt, 1.0)
        if (cnt == 0).any():
            fill = torch.nan_to_num(lp_map, nan=0.0).mean()
            lp_map = torch.where(cnt == 0, fill, lp_map)
        return torch.nan_to_num(lp_map, nan=0.0).detach().cpu().numpy()

    # ---------- device / cfg / model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_mse  = load_config(cfg_mse_path)
    cfg_facl = load_config(cfg_facl_path)

    model_mse  = build_model(cfg_mse).to(device)
    model_facl = build_model(cfg_facl).to(device)
    _ = load_checkpoint_for_eval(ckpt_mse_path,  model_mse,  device=device)
    _ = load_checkpoint_for_eval(ckpt_facl_path, model_facl, device=device)
    model_mse.eval(); model_facl.eval()

    # 동일 테스트 셋
    test_loader = get_test_loader(cfg_mse)

    # ---------- 첫/지정 샘플만 인퍼런스 ----------
    preds_facl, preds_mse, targets = [], [], []
    processed = 0
    for batch in test_loader:
        x, y = _extract_xy(batch, device)
        out_facl = _forward_model(model_facl, x)
        out_mse  = _forward_model(model_mse,  x)

        preds_facl.append(out_facl.detach().cpu())
        preds_mse.append(out_mse.detach().cpu())
        targets.append(y.detach().cpu())

        bs = x[0].size(0) if isinstance(x, (list, tuple)) else x.size(0)
        processed += bs
        if processed > sample_idx:
            break

    if len(preds_facl) == 0:
        raise RuntimeError("test_loader가 비어있습니다. cfg_mse의 데이터 경로나 split을 확인하세요.")

    preds_facl = _ensure_5d(torch.cat(preds_facl, dim=0))
    preds_mse  = _ensure_5d(torch.cat(preds_mse,  dim=0))
    targets    = _ensure_5d(torch.cat(targets,    dim=0))

    B, C, T, H, W = targets.shape
    assert 0 <= sample_idx < B and 0 <= ch_idx < C and 0 <= t_idx < T, \
        f"indices out of range: BCT=({B},{C},{T}) asked ({sample_idx},{ch_idx},{t_idx})"

    gt = targets[sample_idx, ch_idx, t_idx].float().to(device)      # (H,W)
    pa = preds_facl[sample_idx, ch_idx, t_idx].float().to(device)   # (H,W)
    pb = preds_mse [sample_idx, ch_idx, t_idx].float().to(device)   # (H,W)

    # ---------- LPIPS ----------
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    lp_a = _lpips_patch_map(lpips_model, pa, gt, patch, stride)   # LPIPS(FACL, GT)
    lp_b = _lpips_patch_map(lpips_model, pb, gt, patch, stride)   # LPIPS(MSE,  GT)
    lp_d = lp_a - lp_b                                            # Δ = A − B

    # ---------- GT 시각화용 0..1 ----------
    gtn = torch.nan_to_num(gt, nan=0.0).detach().cpu().numpy()
    lo_v, hi_v = np.percentile(gtn, [0.5, 99.5]); den = max(1e-9, hi_v - lo_v)
    gt_vis = np.clip((gtn - lo_v) / den, 0.0, 1.0)
    gt_vis = np.nan_to_num(gt_vis, nan=0.0)

    vmax_a = float(np.nanpercentile(lp_a.ravel(), 99.5))
    vmax_d = float(np.nanpercentile(np.abs(lp_d.ravel()), 99.5))
    lp_a = np.nan_to_num(lp_a, nan=0.0)
    lp_b = np.nan_to_num(lp_b, nan=0.0)
    lp_d = np.nan_to_num(lp_d, nan=0.0)

    # ---------- Figure ----------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.8),
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )
    fig.patch.set_facecolor("white"); fig.patch.set_alpha(1.0)
    for ax in axes:
        ax.set_facecolor("white"); ax.patch.set_alpha(1.0)
        if hasattr(ax, "background_patch"):
            ax.background_patch.set_facecolor("white")
            ax.background_patch.set_alpha(1.0)

    def _decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        if add_gridlines:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.5, linestyle="dashed", color="gray", alpha=0.6)
            gl.xlocator = mticker.FixedLocator(lon_ticks)
            gl.ylocator = mticker.FixedLocator(lat_ticks)

    Hh, Ww = gt_vis.shape
    lon = np.linspace(extent[0], extent[1], Ww)
    lat = np.linspace(extent[2], extent[3], Hh)

    # (a) GT
    im0 = axes[0].pcolormesh(lon, lat, gt_vis, cmap=plasma, vmin=0, vmax=1,
                             transform=ccrs.PlateCarree(), shading="auto")
    axes[0].set_title("(a) GT", fontsize=title_fontsize); _decorate(axes[0])

    # (b) LPIPS(FACL,GT)
    im1 = axes[1].pcolormesh(lon, lat, lp_a, cmap=magma, vmin=0, vmax=vmax_a,
                             transform=ccrs.PlateCarree(), shading="auto")
    axes[1].set_title(f"(b) LPIPS({title_a}, GT)", fontsize=title_fontsize); _decorate(axes[1])

    # (c) ΔLPIPS = LPIPS(FACL,GT) − LPIPS(MSE,GT)
    im2 = axes[2].pcolormesh(lon, lat, lp_d, cmap=bwr,
                             vmin=-vmax_d, vmax=vmax_d,
                             transform=ccrs.PlateCarree(), shading="auto")
    axes[2].set_title(f"(c) ΔLPIPS ({title_a} − {title_b})", fontsize=title_fontsize); _decorate(axes[2])

    # colorbar는 Δ 기준(가장 해석 용이)
    cbar = fig.colorbar(im2, ax=axes, orientation="vertical",
                        shrink=0.9, fraction=0.046, pad=0.04)
    cbar.set_label("ΔLPIPS", fontsize=16)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    cbar.ax.set_facecolor('white'); cbar.ax.patch.set_alpha(1.0)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor='white', edgecolor='white', transparent=False)
    plt.close(fig)
    print(f"[OK] saved → {out_png}")

# ===================== LPIPS 비교
# ----------------- Common helpers -----------------

@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    # x: (...,1,H,W) -> (...,n,H,W)
    return x.expand(*x.shape[:-3], n, x.shape[-2], x.shape[-1])

@torch.no_grad()
def _log_normalize(data: torch.Tensor, c=-0.4, d=103.0, a=0.0, b=1.0):
    # [c,d]→[0,1] 선형정규화 후 log1p/ln2 (네가 쓰던 스케일 그대로)
    scale = (b - a) / (d - c)
    z = ((data - c) * scale) + a
    z = torch.clamp(z, 0.0, 1.0)
    return torch.log1p(z) / math.log(2.0)

def _percentile_minmax(x: np.ndarray, p_lo=0.5, p_hi=99.5):
    vmin = float(np.nanpercentile(x, p_lo))
    vmax = float(np.nanpercentile(x, p_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
        if vmin == vmax: vmax = vmin + 1e-9
    return vmin, vmax


# ----------------- Main: A/B LPIPS + Δ (imshow 사용) -----------------
import math, os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, ticker as mticker
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ----------------- 이미 보유한 헬퍼 (그대로 사용) -----------------
@torch.no_grad()
def _expand_gray_to_rgb(x, n=3):
    # x: (...,1,H,W) -> (...,n,H,W)
    return x.expand(*x.shape[:-3], n, x.shape[-2], x.shape[-1])

@torch.no_grad()
def _log_normalize(data: torch.Tensor, c=-0.4, d=103.0, a=0.0, b=1.0):
    # [c,d]→[0,1] 선형정규화 후 log1p/ln2
    scale = (b - a) / (d - c)
    z = ((data - c) * scale) + a
    z = torch.clamp(z, 0.0, 1.0)
    return torch.log1p(z) / math.log(2.0)

def _percentile_minmax(x: np.ndarray, p_lo=0.5, p_hi=99.5):
    vmin = float(np.nanpercentile(x, p_lo))
    vmax = float(np.nanpercentile(x, p_hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
        if vmin == vmax: vmax = vmin + 1e-9
    return vmin, vmax

# --------------------------------------------------------------------
#  lpips_patch_map_single_frame: 사용자가 준 방식 그대로 재현
#   - 패치 그리드: range(0, H - patch + 1, stride)
#   - 각 패치별 log 정규화 → RGB 확장 → LPIPS
#   - 겹침은 평균, 닿지 않은 픽셀은 전체 평균으로 채움
# --------------------------------------------------------------------
def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        # 파라미터가 없는 모듈일 경우 버퍼에서 가져오거나 CPU 기본
        for b in model.buffers():
            return b.device
        return torch.device("cpu")

def _to_model_device(x: torch.Tensor, model: torch.nn.Module, dtype=torch.float32):
    dev = _model_device(model)
    return x.to(device=dev, dtype=dtype)

# ---- 1) 패치맵 함수: p/g & 누적 버퍼 모두 "모델 디바이스"에 맞추기 ----
@torch.no_grad()
def lpips_patch_map_single_frame(pred_1ch, target_1ch, model,
                                 patch=32, stride=16, lognorm=True):
    """
    pred_1ch, target_1ch: (H,W) torch.Tensor (device 무관)
    반환: (H,W) float32 ndarray. 스캔 안된 외곽은 0(under→white).
    """
    # 모델 기준 디바이스로 정렬
    model_device = _model_device(model)
    model.to(model_device)  # shift/scale 버퍼까지 이동

    p_full = torch.nan_to_num(pred_1ch, nan=0.0).detach().to(model_device, dtype=torch.float32)
    g_full = torch.nan_to_num(target_1ch, nan=0.0).detach().to(model_device, dtype=torch.float32)

    H, W = p_full.shape
    acc = torch.zeros((H, W), device=model_device, dtype=torch.float32)
    cnt = torch.zeros((H, W), device=model_device, dtype=torch.float32)

    # 경계 포함 안 되는 영역 최소화: 마지막 패치를 가장자리에 정렬
    ys = list(range(0, max(1, H - patch + 1), stride))
    xs = list(range(0, max(1, W - patch + 1), stride))
    if ys[-1] != H - patch: ys.append(max(0, H - patch))
    if xs[-1] != W - patch: xs.append(max(0, W - patch))

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + patch, x0 + patch
            p = p_full[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0)  # (1,1,P,P)
            g = g_full[y0:y1, x0:x1].unsqueeze(0).unsqueeze(0)

            if lognorm:
                p = _log_normalize(p)
                g = _log_normalize(g)

            p = _expand_gray_to_rgb(p, 3)  # (1,3,P,P)
            g = _expand_gray_to_rgb(g, 3)

            # torchmetrics LPIPS는 모델 버퍼와 같은 디바이스여야 함
            s = model(p, g).detach().mean()
            acc[y0:y1, x0:x1] += float(s.item())
            cnt[y0:y1, x0:x1] += 1.0

    lp_map = torch.div(acc, torch.clamp_min(cnt, 1.0))
    return torch.nan_to_num(lp_map, nan=0.0).detach().cpu().numpy().astype(np.float32, copy=False)




# --------------------------------------------------------------------
#  Main: A/B LPIPS + Δ (위 함수 방식으로 계산, 배경 흰색 유지)
# --------------------------------------------------------------------
@torch.no_grad()
def make_lpips_ab_and_delta_map(
    cfg_a_path: str, cfg_b_path: str,
    ckpt_a_path: str, ckpt_b_path: str,
    out_png: str,
    sample_idx: int = 0, ch_idx: int = 0, t_idx: int = 0,
    central_longitude: float = 180.0,
    extent = (0, 360, -90, 90),
    patch: int = 32, stride: int = 16,
    title_a: str = "Model-A", title_b: str = "Model-B",
    add_gridlines: bool = False,
    title_fontsize: int = 19,      # ← 더 이상 사용하지 않지만 시그니처는 유지
    tick_labelsize: int = 13,      # ← 더 이상 사용하지 않지만 시그니처는 유지
    dpi: int = 300,
    abs_diff: bool = False, vmax_percentile: float = 99.5,
):
    """
    (a) GT, (b) LPIPS(A,GT), (c) LPIPS(B,GT), (d) Δ(A−B)
    - LPIPS 맵 계산은 lpips_patch_map_single_frame 방식
    - A/B 패널: 0/NaN → 흰색 (magma_r + set_under/Bad=white, vmin>0)
    - Δ 패널: 기본은 center=0 흰색(bwr), abs_diff=True면 0/NaN 흰색(magma_r)
    - 제목/축/컬러바 폰트는 모두 rcParams를 따름(visualize_prediction_map과 동일)
    """
    import os, numpy as np, torch, matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm, ticker as mticker
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # ---- 프로젝트 유틸/모델 ----
    from utils.config_utils import load_config
    from utils.save_utils    import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model    import build_model

    # ---- 저장/그림 배경 흰색 확정 ----
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['savefig.edgecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'

    # ---- 컬러맵: NaN/under → white ----
    plasma = cm.get_cmap('plasma').copy()
    plasma.set_bad('white', 1.0); plasma.set_under('white', 1.0)

    magma_r_white = cm.get_cmap('magma_r').copy()
    magma_r_white.set_bad('white', 1.0)   # NaN → white
    magma_r_white.set_under('white', 1.0) # vmin 미만(=0 포함) → white

    bwr = cm.get_cmap('bwr').copy()
    bwr.set_bad('white', 1.0)

    # ---- LPIPS 모델 (사용자가 제공한 함수 사용) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = get_lpips_model(device=device).eval()

    # ---- cfg & 모델 & 데이터 ----
    cfg_a = load_config(cfg_a_path)
    cfg_b = load_config(cfg_b_path)
    model_a = build_model(cfg_a).to(device).eval()
    model_b = build_model(cfg_b).to(device).eval()
    _ = load_checkpoint_for_eval(ckpt_a_path, model_a, device=device)
    _ = load_checkpoint_for_eval(ckpt_b_path, model_b, device=device)
    test_loader = get_test_loader(cfg_a)

    # ---- 지정 샘플만 추론 ----
    preds_a, preds_b, targets = [], [], []
    processed = 0
    for batch in test_loader:
        if isinstance(batch, dict):
            if all(k in batch for k in ("X1","X2","Y")):
                x = (batch["X1"].to(device), batch["X2"].to(device))
                y = batch["Y"].to(device)
            else:
                x = next((batch[k].to(device) for k in ("input","inputs","x","src","data","image","images") if k in batch), None)
                y = next((batch[k].to(device) for k in ("target","targets","y","label","labels") if k in batch), None)
        else:
            x, y = batch
            if isinstance(x, (list, tuple)):
                x = tuple(_x.to(device) for _x in x)
            else:
                x = x.to(device)
            y = y.to(device)

        def _forward(model, inp):
            if isinstance(inp, (list, tuple)): return model(inp[0], inp[1])
            return model(inp)

        with torch.no_grad():
            out_a = _forward(model_a, x)
            out_b = _forward(model_b, x)

        preds_a.append(out_a.detach().cpu())
        preds_b.append(out_b.detach().cpu())
        targets.append(y.detach().cpu())

        bs = x[0].size(0) if isinstance(x, (list, tuple)) else x.size(0)
        processed += bs
        if processed > sample_idx:
            break

    def _ensure_5d(t):
        if t.dim() == 5: return t
        if t.dim() == 4: return t.unsqueeze(2)
        raise ValueError(f"expect 4D/5D, got {tuple(t.shape)}")

    preds_a = _ensure_5d(torch.cat(preds_a, dim=0))
    preds_b = _ensure_5d(torch.cat(preds_b, dim=0))
    targets = _ensure_5d(torch.cat(targets, dim=0))

    gt = targets[sample_idx, ch_idx, t_idx].float().to(device)
    pa = preds_a [sample_idx, ch_idx, t_idx].float().to(device)
    pb = preds_b [sample_idx, ch_idx, t_idx].float().to(device)

    # ---- LPIPS 패치 맵 (가장자리 0 유지) ----
    lp_a = lpips_patch_map_single_frame(pa, gt, lpips_model, patch=patch, stride=stride)
    lp_b = lpips_patch_map_single_frame(pb, gt, lpips_model, patch=patch, stride=stride)
    lp_d = lp_a - lp_b

    # ---- GT 시각화용 [0,1] ----
    gtn = torch.nan_to_num(gt, nan=0.0).cpu().numpy()
    def _percentile_minmax(x: np.ndarray, p_lo=0.5, p_hi=99.5):
        vmin = float(np.nanpercentile(x, p_lo))
        vmax = float(np.nanpercentile(x, p_hi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
            if vmin == vmax: vmax = vmin + 1e-9
        return vmin, vmax

    vmin_gt, vmax_gt = _percentile_minmax(gtn, 0.5, 99.5)
    gt_vis = np.clip((gtn - vmin_gt) / (vmax_gt - vmin_gt + 1e-9), 0.0, 1.0)
    gt_vis = np.ma.masked_invalid(gt_vis)

    # ---- LPIPS 색상 범위(A/B 공통, 0 제외) ----
    def _pos_percentile_minmax(arrs, p_lo=0.5, p_hi=99.5):
        vals = np.concatenate([a.ravel() for a in arrs])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return 1e-12, 1.0
        vmin = float(np.nanpercentile(vals, p_lo))
        vmax = float(np.nanpercentile(vals, p_hi))
        if vmin == vmax: vmax = vmin + 1e-9
        return vmin, vmax

    vmin_lp, vmax_lp = _pos_percentile_minmax([lp_a, lp_b])
    vmin_lp_plot = max(vmin_lp, 1e-12)  # 0은 under → white

    # ---- Δ 스케일 ----
    if abs_diff:
        lp_d_plot = np.abs(lp_d)
        vmin_d, vmax_d = _pos_percentile_minmax([lp_d_plot])
        vmin_d_plot = max(vmin_d, 1e-12)
        norm_d = Normalize(vmin=vmin_d_plot, vmax=vmax_d, clip=False)
        cmap_d = magma_r_white
        lp_d_vis = np.ma.masked_invalid(lp_d_plot)
    else:
        vmax_d = float(np.nanpercentile(np.abs(lp_d.ravel()), 99.5))
        norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0.0, vmax=vmax_d)  # 0이 흰색
        cmap_d = bwr
        lp_d_vis = np.ma.masked_invalid(lp_d)

    # ---- 축 꾸미기(visualize_prediction_map과 동일 스타일) ----
    def _decorate_geo_axes(ax, add_grid=True):
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
        if add_grid:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.5, linestyle="dashed",
                              color="gray", alpha=0.6, zorder=6)
            gl.xlocator = mticker.FixedLocator(lon_ticks)
            gl.ylocator = mticker.FixedLocator(lat_ticks)
        ax.set_global()

    # ---- Figure ----
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8),
                             subplot_kw=dict(projection=proj),
                             constrained_layout=True,
                             facecolor='white')

    # (a) GT
    im0 = axes[0].imshow(gt_vis, origin="lower", extent=extent,
                         transform=ccrs.PlateCarree(), cmap=plasma,
                         vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("(q) GT")                     # ← fontsize 지정 없음
    _decorate_geo_axes(axes[0], add_gridlines)

    # (b) LPIPS(A,GT) — 0/NaN 흰색(under/bad)
    im1 = axes[1].imshow(np.ma.masked_invalid(lp_a), origin="lower", extent=extent,
                         transform=ccrs.PlateCarree(), cmap=magma_r_white,
                         vmin=vmin_lp_plot, vmax=vmax_lp, interpolation="nearest")
    axes[1].set_title(f"(a) LPIPS({title_a}, GT)")
    _decorate_geo_axes(axes[1], add_gridlines)

    # (c) LPIPS(B,GT) — 0/NaN 흰색(under/bad)
    im2 = axes[2].imshow(np.ma.masked_invalid(lp_b), origin="lower", extent=extent,
                         transform=ccrs.PlateCarree(), cmap=magma_r_white,
                         vmin=vmin_lp_plot, vmax=vmax_lp, interpolation="nearest")
    axes[2].set_title(f"(b) LPIPS({title_b}, GT)")
    _decorate_geo_axes(axes[2], add_gridlines)

    # (d) Δ — abs_diff: 0/NaN 흰색 / 기본: center=0 흰색
    im3 = axes[3].imshow(lp_d_vis, origin="lower", extent=extent,
                         transform=ccrs.PlateCarree(), cmap=cmap_d,
                         norm=norm_d, interpolation="nearest")
    axes[3].set_title(f"(c) {'|ΔLPIPS|' if abs_diff else 'ΔLPIPS'} ({title_a}−{title_b})")
    _decorate_geo_axes(axes[3], add_gridlines)

    # ---- Colorbars (라벨/틱 크기 모두 rcParams 따름) ----
    fig.colorbar(im1, ax=[axes[1], axes[2]], orientation="vertical",
                 shrink=0.9, fraction=0.046, pad=0.02).set_label("LPIPS (↓)")
    fig.colorbar(im3, ax=[axes[0], axes[3]], orientation="vertical",
                 shrink=0.9, fraction=0.046, pad=0.02).set_label("|ΔLPIPS|" if abs_diff else "ΔLPIPS")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight",
                facecolor='white', edgecolor='white', transparent=False)
    plt.close(fig)
    print(f"[OK] saved → {out_png}")







#===========DTCWT kernels==============
import numpy as np
import matplotlib.pyplot as plt

# utils/dtcwt_vis.py
import numpy as np
import matplotlib.pyplot as plt

def visualize_dtcwt_kernels(
    size=(256, 256),
    nlevels=1,
    biort="near_sym_b",
    qshift="qshift_b",
    show_phase=True,
    cmap_mag="gray",
    cmap_phase="twilight",
    save_path=None,
    # ----- 새 옵션들 -----
    crop=64,                     # 중앙 크롭 반경(px). None이면 자동(min(H,W)//6)
    mag_pr=(1, 99),              # Magnitude 퍼센타일 스트레치
    real_pr=99,                  # Real 대칭 퍼센타일(|real|의 상위 p%)
    use_log_mag=True,            # Magnitude 로그 스케일(log1p)
    phase_from_mag=True,         # Magnitude 큰 곳만 phase를 보이도록 알파마스크 적용
    phase_mag_thresh=0.25,       # (정규화된) Magnitude 임계값(0~1)
    interp="nearest",            # imshow 보간 방식
):
    """
    DTCWT Level-1의 6방향 임펄스 응답(=실효 2D 커널)을 또렷하게 시각화한다.

    - 임펄스(δ)를 특정 서브밴드에만 주입 후 inverse → 공간영역 커널 복원
    - 중앙 크롭 + 퍼센타일 스트레치 + (선택) 로그 스케일로 가시성 향상
    - Phase는 Magnitude가 충분히 큰 위치에서만 반투명으로 표시(phase_from_mag)
    """
    # 지연 임포트 (패키지 없으면 여기서 에러)
    from dtcwt.numpy import Transform2d, Pyramid

    # -------------------- 변환 준비 --------------------
    H, W = map(int, size)
    t = Transform2d(biort=biort, qshift=qshift)

    # 임펄스 → forward (shape 파악)
    imp = np.zeros((H, W), dtype=np.float32)
    imp[H // 2, W // 2] = 1.0
    pyr = t.forward(imp, nlevels=nlevels)

    # Lowpass 임펄스 응답
    lp = np.zeros_like(pyr.lowpass)
    lp[lp.shape[0] // 2, lp.shape[1] // 2] = 1.0
    hp_zeros = [np.zeros_like(pyr.highpasses[i]) for i in range(nlevels)]
    lp_resp = t.inverse(Pyramid(lowpass=lp, highpasses=hp_zeros))  # (H,W)

    # Level-1 6방향 커널 복원
    hp = pyr.highpasses[0]                 # (H/2, W/2, 6) complex
    K = hp.shape[-1]                       # 보통 6
    kernels = []
    for k in range(K):
        sel = [np.zeros_like(hp) for _ in range(nlevels)]
        sel[0][hp.shape[0] // 2, hp.shape[1] // 2, k] = 1.0 + 0j
        rec = t.inverse(Pyramid(lowpass=np.zeros_like(pyr.lowpass), highpasses=sel))  # (H,W) complex(or real)
        kernels.append(rec)

    # -------------------- 보조 유틸 --------------------
    def _center_slice(h, w, c):
        if c is None:
            c = min(h, w) // 6
        c = int(max(8, min(c, min(h, w)//2 - 1)))
        cy, cx = h // 2, w // 2
        return (slice(cy - c, cy + c), slice(cx - c, cx + c))

    def _robust_norm(a, pr=(1, 99)):
        """퍼센타일 기반 0~1 정규화(강건)."""
        lo, hi = np.percentile(a, pr[0]), np.percentile(a, pr[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            lo, hi = a.min(), a.max() + 1e-6
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    # 중앙 크롭 영역
    cs = _center_slice(H, W, crop)

    # -------------------- 시각화 --------------------
    ncols = 3 if show_phase else 2
    fig, axes = plt.subplots(K + 1, ncols,
                             figsize=(4.6 * ncols, 2 + K * 2.2),
                             constrained_layout=True)

    # Lowpass (Magnitude만 표시)
    lp_crop = lp_resp[cs]
    mag_lp = np.abs(lp_crop)
    if use_log_mag:
        mag_lp = np.log1p(mag_lp)
    mag_lp = _robust_norm(mag_lp, mag_pr)

    axes[0, 0].imshow(mag_lp, cmap=cmap_mag, interpolation=interp)
    axes[0, 0].set_title("Lowpass (Level 1)")
    axes[0, 0].axis("off")
    if ncols > 2:
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")
    else:
        axes[0, 1].axis("off")

    # 각 방향 커널
    for i, rec in enumerate(kernels, start=1):
        rec_c = rec.astype(np.complex64) if np.iscomplexobj(rec) else rec + 0j
        crop_c = rec_c[cs]

        # Magnitude
        mag = np.abs(crop_c)
        if use_log_mag:
            mag = np.log1p(mag)
        mag_n = _robust_norm(mag, mag_pr)

        axes[i, 0].imshow(mag_n, cmap=cmap_mag, interpolation=interp)
        axes[i, 0].set_title(f"Orient {i-1} — Magnitude")
        axes[i, 0].axis("off")

        # Real (대칭 vmin/vmax)
        real = np.real(crop_c)
        vmax = np.percentile(np.abs(real), real_pr)
        vmax = float(max(vmax, 1e-8))
        axes[i, 1].imshow(real, cmap="seismic", vmin=-vmax, vmax=vmax, interpolation=interp)
        axes[i, 1].set_title(f"Orient {i-1} — Real")
        axes[i, 1].axis("off")

        # Phase (옵션, magnitude 기반 알파마스크)
        if show_phase:
            phase = np.angle(crop_c)
            if phase_from_mag:
                # mag_n ∈ [0,1] → thresh 이상만 보이도록 선형 알파
                a = np.clip((mag_n - phase_mag_thresh) / max(1e-6, 1 - phase_mag_thresh), 0, 1)
                axes[i, 2].imshow(phase, cmap=cmap_phase, vmin=-np.pi, vmax=np.pi,
                                  interpolation=interp, alpha=a)
            else:
                axes[i, 2].imshow(phase, cmap=cmap_phase, vmin=-np.pi, vmax=np.pi,
                                  interpolation=interp)
            axes[i, 2].set_title(f"Orient {i-1} — Phase")
            axes[i, 2].axis("off")

    fig.suptitle(f"DTCWT kernels (biort={biort}, qshift={qshift})", y=0.995)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

from utils.config_utils import load_config


def run(cfg, out_dir, logger):
    """
    main.py에서 task=eval/test/vis로 라우팅될 때 호출되는 엔트리포인트.
    - 체크포인트 로드
    - 테스트셋 평가
    - 평균 메트릭 로그 및 첫 샘플 시각화 저장
    """
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 모델 생성
    model = build_model(cfg).to(device).float()

    # 체크포인트 결정: monitor.key → best_<name>.pt, 없으면 last.pt
    mon_key = cfg.get("monitor", {}).get("key", "val/mse")
    mon_name = mon_key.split("/")[-1]
    ckpt_dir = cfg["paths"]["ckpt_dir"]
    default_ckpt = os.path.join(ckpt_dir, f"best_{mon_name}.pt")
    if not os.path.exists(default_ckpt):
        default_ckpt = os.path.join(ckpt_dir, "last.pt")
        
    # ckpt yaml에 지정하여 사용    
    ckpt_path = cfg.get("ckpt", default_ckpt)

    logger.info(f"[Eval-3D] Loading checkpoint: {ckpt_path}")
    checkpoint = load_checkpoint_for_eval(ckpt_path, model, device=device)
    epoch = checkpoint.get("epoch", -1)

    # 3) 데이터/메트릭 준비
    test_loader = get_test_loader(cfg)       
    metric_fns  = get_metric_fns(cfg)       

    # 4) 평가 수행
    avg_metrics, preds, targets, datetimes = evaluate_epoch(
        model=model,
        loader=test_loader,
        metric_fns=metric_fns,
        device=device,
        metric_params=cfg["metrics"].get("params", {}),
    )

    # 5) 로그 출력
    logger.info(f"[Eval] ckpt epoch={epoch}")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    # 5-1) 리드타임별 메트릭 계산 및 CSV 저장
    import copy, csv as _csv
    logger.info("[Eval-3D] Computing lead-time–specific metrics (lead_idx-based)")
    exp_name = cfg.get("experiment_name", cfg.get("exp_name", "experiment"))
    csv_dir  = cfg["paths"].get("log_dir", out_dir)
    os.makedirs(csv_dir, exist_ok=True)

    cfg_fixed = copy.deepcopy(cfg)
    cfg_fixed.setdefault("data", {})
    cfg_fixed["data"]["label_offsets"] = [1, 2, 3]
    test_loader_fixed = get_test_loader(cfg_fixed)

    leadtime_rows = []
    for lead_day in [1, 2, 3]:
        logger.info(f"[Eval-3D] Lead t+{lead_day}")
        lead_metrics, _, _, _ = evaluate_epoch(
            model=model,
            loader=test_loader_fixed,
            metric_fns=metric_fns,
            device=device,
            metric_params=cfg["metrics"].get("params", {}),
            lead_idx=lead_day - 1,
            keep_pred_shape=True,
        )
        row = {"model": exp_name, "lead_time": lead_day}
        row.update(lead_metrics)
        leadtime_rows.append(row)
        logger.info(
            f"[Lead t+{lead_day}] "
            + ", ".join(f"{k}: {v:.6f}" for k, v in lead_metrics.items())
        )
        test_loader_fixed = get_test_loader(cfg_fixed)

    csv_path = os.path.join(csv_dir, f"metrics_by_leadtime_{exp_name}.csv")
    with open(csv_path, "w", newline="") as _f:
        writer = _csv.DictWriter(_f, fieldnames=leadtime_rows[0].keys())
        writer.writeheader()
        for r in leadtime_rows:
            writer.writerow(r)
    logger.info(f"[Eval-3D] Saved lead-time metrics CSV → {csv_path}")

    # 6) 시각화 (첫 샘플)
    if len(preds) > 0:
        vis_dir = cfg["paths"]["eval_dir"]
        os.makedirs(vis_dir, exist_ok=True)

        exp = cfg.get("experiment_name", cfg.get("exp_name", "exp"))
        loss_name = cfg["loss"]["name"]
        base_time = str(datetimes[0]) if len(datetimes) else "unknown"
        save_path = os.path.join(vis_dir, f"{exp}_{loss_name}_{base_time}_ep{epoch}.png")

        ################## Map 1 ########################
        visualize_prediction(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,
            vmin_percentile=1.0,
            vmax_percentile=98.5,
            cmap="plasma",
        )
        logger.info(f"[Eval-3D] Saved Map 1 visualization → {save_path}")

        ################## Map 2 ########################
        save_path = os.path.join(vis_dir, f"Map2_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
        visualize_prediction_map(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,
            vmin_percentile=1.0,
            vmax_percentile=99.0,
            cmap="plasma",
            central_longitude=180.0,
            add_gridlines=True,
            dpi=300,
        )
        logger.info(f"[Eval-3D] Saved Map2 visualization → {save_path}")


        ################## LPC ########################
        save_path = os.path.join(vis_dir, f"LPC_{exp}_{loss_name}_{base_time}_ep{epoch}.png")

        visualize_lpc_maps(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,
            dpi=300,
        )



        ################## Freq Plot ########################
        save_path = os.path.join(vis_dir, f"Freq_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
        visualize_freq_analysis(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            nbins=128,
            unit="cyc",
            dx=1.0,
            dy=1.0,
            save_path=save_path,
            mask_percentile=5.0,
            amp_profile_ylim=(-10,5),
            show_wavelength=False,
            reducer="mean",
            phase_mode="cos",
            amp_mode="db",
            dpi=300,
            add_freq_guides=False
        )
        logger.info(f"[Eval-3D] Saved Freq visualization → {save_path}")
        ################## LPIPS Map ####################3
        lpips_model = get_lpips_model(net_type="alex", reduction="mean", normalize=True, device="cuda")
        save_path = os.path.join(vis_dir, f"LPIPS_map_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
        gt_tensor   = targets[0]   # shape 예: (16,1,3,180,360)
        pred_tensor = preds[0]
        # 2) 실행 (CPBD와 파라미터 형식 매칭)
        _ = plot_lpips_pair_from_tensors(
            gt=gt_tensor,          # (B,1,T,H,W) 전체 텐서로 전달!
            pred=pred_tensor,          # (B,1,T,H,W)
            model=lpips_model,
            b=0, c=0, t=0,       # t=0 → t₀+1
            extent=(0, 360, -90, 90),
            central_longitude=180,
            patch=32, stride=16,
            dpi=300,
            cmap="viridis",
            save_path=save_path
        )
        logger.info(f"[Eval-3D] Saved LPIPS heatmap → {save_path}")
        #############LPIPS 차이맵#####################
        # 1) LPIPS(A,GT), LPIPS(B,GT), Δ(A−B) 1×4 패널 저장
        save_path = os.path.join(
            vis_dir, f"LPIPS_AB_Delta_{exp}_{base_time}_ep{epoch}.png"
        )

        # (선택) 경로 sanity check
        for p in [
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/mse/mse.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl/facl.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_mse/checkpoints/best_val_lpips.pt",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl/checkpoints/best_val_lpips.pt",
        ]:
            assert isinstance(p, str) and os.path.exists(p), f"Not found: {p}"

        # ➜ 새 함수로 호출 (A=FACL, B=MSE)
        make_lpips_ab_and_delta_map(
            cfg_a_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl/facl.yaml",
            cfg_b_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/mse/mse.yaml",
            ckpt_a_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl/checkpoints/best_val_lpips.pt",
            ckpt_b_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_mse/checkpoints/best_val_lpips.pt",
            out_png=save_path,
            sample_idx=0, ch_idx=0, t_idx=0,
            title_a="FACL", title_b="MSE",
            central_longitude=180.0,
            extent=(0, 360, -90, 90),
            patch=32, stride=16,
            abs_diff=False,          # |Δ|로 보고 싶으면 True
            vmax_percentile=99.5,    # (b)(c) 공통 상한 퍼센타일
        )

        logger.info(f"[Eval-3D] Saved LPIPS A/B & Δ map → {save_path}")
        #===========LPIPS 차이맵 & bar plot=============
        # 0) LPIPS 모델 (이미 있으시면 그대로 사용)
        lpips_model = get_lpips_model(net_type="alex", reduction="mean", normalize=True, device="cuda").eval()

        # 1) Tensor 준비
        gt_tensor   = targets[0]   # (B,1,T,H,W) 혹은 (B,1,H,W)
        pred_tensor = preds[0]

        # 2) 저장 경로
        save_path = os.path.join(
            vis_dir, f"LPIPS_bands_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        # 3) 실행: 상단 4맵 + 하단 바(폭만 축소)
        out = show_lpips_band_evidence_ll_compact_bar(
            gt=gt_tensor,
            pred=pred_tensor,
            lpips_model=lpips_model,
            b=0, c=0, t=0,                 # t=0 → t₀+1
            levels=3,
            extent=(0, 360, -90, 90),
            central_longitude=180,
            crop_n=12,
            ll_percentiles=(1, 99),
            high_percentiles=(1, 99),
            figsize=(16, 6.5),
            dpi=300,
            save_path=save_path,
            bar_width=0.42                 # 👈 막대 가로 길이만 살짝 줄임 (0.35~0.5 권장)
        )

        logger.info(f"[Eval-3D] Saved LPIPS bands+bar → {save_path}")
        # 필요하면 반환값 사용:
        # print("chosen orientations:", out["orients"])
        # print("bar LPIPS:", out["bar_vals"])





        ################## CPBD Map (GT vs 현재 예측) ########################
        # ================= CPBD =================
        save_path = os.path.join(
            vis_dir, f"CPBD_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        plot_cpbd_pair_from_tensors(
            gt=targets[0],     # (B,1,T,H,W)
            pred=preds[0],
            b=0, c=0, t=0,     # t=0 → t₀+1 시점
            extent=(0, 360, -90, 90),
            central_longitude=180,
            patch=32, stride=16,
            cpbd_kwargs=dict(
                block_size=15,
                edge_ratio_threshold=0.08,
                jnb_parameters=(0.5, 0.05, 10.0, 0.1),
                edge_gate_percentile=90.0,
                p_threshold=0.5,
                output_scale="percent",
                crop_instead_of_pad=False,
                use_canny=False,
                auto_gain=True,
            ),
            # ---- 시각화 옵션 ----
            save_path=save_path,
            dpi=300,
            cmap="viridis",
            # ---- [추가] 대비 조정용 파라미터 ----
            scale_mode="power",   # "power"(기본) | "linear" | "fixed"
            p_lo=5.0,             # 하위 5% 클리핑
            p_hi=95.0,            # 상위 95% 클리핑
            gamma=0.7,            # 감마 <1 → 중간대 대비 강화
            # 고정스케일 예: scale_mode="fixed", fixed_vmin=20, fixed_vmax=90
        )

        logger.info(f"[Eval-3D] Saved CPBD visualization → {save_path}")

        save_path = os.path.join(vis_dir, f"CPBD_diff_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
        plot_cpbd_diffmap_from_tensors(
            gt=targets[0],
            pred=preds[0],
            baseline=targets[0],   # baseline = GT
            b=0, c=0, t=0,
            extent=(0, 360, -90, 90), central_longitude=180,
            patch=32, stride=16,
            cpbd_kwargs=dict(
                block_size=15,
                edge_ratio_threshold=0.08,
                jnb_parameters=(0.5, 0.05, 10.0, 0.1),
                edge_gate_percentile=90.0,
                p_threshold=0.5,
                output_scale="percent",
                crop_instead_of_pad=False,
                use_canny=False,
                auto_gain=True,
            ),
            save_path=save_path,
            dpi=300,
            diff_cmap="coolwarm",
            abs_diff=False,  # True면 |Δ|만
            title=f"(b) FACL",
        )
        logger.info(f"[Eval-3D] Saved ΔCPBD (vs GT) → {save_path}")
        # 💡 추가: DTCWT 기반 LL & 고주파 밴드 + LPIPS bar (모두 Lv1 해상도에서 비교)
        save_path_lv = os.path.join(
            vis_dir, f"LL_Lv123_LPIPS_Lv1res_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        fig, out = plot_ll_lv123_with_lpipsbar_same_res(
            gt=targets[0],          # (B,1,T,H,W)
            pred=preds[0],          # (B,1,T,H,W)
            lpips_model=lpips_model,  # ✅ 미리 준비한 LPIPS 인스턴스 (아래 참고)
            b=0, c=0, t=0,          # 첫 배치, t=0 → t+1
            levels=3,               # Lv1~Lv3
            orient=0,               # 0~5 중 택1 (모든 레벨 동일 방향)
            extent=(0, 360, -90, 90),
            central_longitude=180,
            save_path=save_path_lv,
            close_after_save=True,  # 저장 후 figure 닫기 (중복 그리기 방지)
            # ↓ 리샘플 파라미터(원해상도에서 복원 → 모두 Lv1 해상도로 다운샘플)
            resample_order=3,
            antialias=True,
        )

        logger.info(f"[Eval-3D] Saved LL/Lv1/Lv2/Lv3 LPIPS bar (Lv1 res) → {save_path_lv}")
        logger.info(f"LPIPS scores: {dict(zip(out['labels'], out['lpips']))}")
        #===========LPIPS heat map & CDF plot===============
        # (권장) 모델-입력 디바이스 일치
        lpips_model = lpips_model.to(gt_tensor.device)

        # 0) 저장 경로
        save_path_evi = os.path.join(
            vis_dir, f"LPIPS_evidence_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        # 1) 실행 (LL/Lv1/Lv2/Lv3 = 모두 Patch 기반)
        _ = show_lpips_band_evidence_ll(
            gt=gt_tensor,
            pred=pred_tensor,
            lpips_model=lpips_model,
            b=0, c=0, t=0,
            levels=3,
            extent=(0, 360, -90, 90),
            central_longitude=180,
            crop_n=12,
            ll_percentiles=(1, 99),
            high_percentiles=(1, 99),

            # 패치 설정 (alex 기준)
            patch=32,
            stride=16,

            # 바플롯: 아래 한 줄에 좁게 배치
            bar_span_cols=1,      # 아래줄 4칸 중 2칸만 사용
            bar_start_col=1,      # 가운데 정렬(1~2칸 사용)
            bar_width=0.35,       # 막대 더 얇게
            bar_color="steelblue",

            figsize=(16, 6.5),
            dpi=300,
            save_path=save_path_evi,
            add_gridlines=True,
        )

        logger.info(f"[Eval-3D] Saved LPIPS level-evidence → {save_path_evi}")









                ################## DTCWT Kernels ########################
        try:
            # 입력 크기 추정 (B,1,T,H,W 가정)
            if targets[0].ndim == 5:
                H, W = map(int, targets[0][0, 0, 0].shape[-2:])
            else:
                H, W = map(int, targets[0].shape[-2:])

            save_path = os.path.join(
                vis_dir, f"DTCWT_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
            )
            visualize_dtcwt_kernels(
                size=(H, W),
                nlevels=cfg.get("viz", {}).get("dtcwt_levels", 1),
                biort=cfg.get("viz", {}).get("dtcwt_biort", "near_sym_b"),
                qshift=cfg.get("viz", {}).get("dtcwt_qshift", "qshift_b"),
                show_phase=cfg.get("viz", {}).get("dtcwt_show_phase", True),
                save_path=save_path,
            )
            logger.info(f"[Eval-3D] Saved DTCWT kernel visualization → {save_path}")
        except ImportError:
            logger.warning("[Eval-3D] dtcwt not installed; skipping DTCWT kernel visualization.")
        except Exception as e:
            logger.exception(f"[Eval-3D] DTCWT kernel visualization failed: {e}")


    else:
        logger.warning("[Eval-3D] No predictions collected; skipping visualization.")

    return avg_metrics


@torch.no_grad()
def evaluate_epoch(model, loader, metric_fns, device, metric_params=None,
                   lead_idx=None, keep_pred_shape=True):
    """
    metric_fns: dict[str -> callable], (pred, y, **metric_params) -> float (or scalar tensor)
    lead_idx: 0/1/2 → t+1/t+2/t+3 해당 lead만 metric 계산
    반환: (평균 metric dict, preds list[tensor], targets list[tensor], datetimes list[str])
    """
    model.eval()
    metrics = defaultdict(float)
    count = 0
    preds, targets, datetimes = [], [], []

    def _select_lead(t: torch.Tensor, idx: int) -> torch.Tensor:
        if t.ndim != 5:
            return t
        idx = int(idx)
        if t.shape[2] <= 10:       # (B, C, T, H, W)
            T = t.shape[2]
            j = max(0, min(idx, T - 1))
            return t[:, :, j:j+1, :, :]
        if t.shape[1] <= 10:       # (B, T, C, H, W)
            T = t.shape[1]
            j = max(0, min(idx, T - 1))
            return t[:, j:j+1, :, :, :]
        return t

    for batch in loader:
        x1 = batch["X1"].to(device, non_blocking=True).float()
        x2 = batch["X2"].to(device, non_blocking=True).float()
        x3 = batch["X3"].to(device, non_blocking=True).float()
        x4 = batch["X4"].to(device, non_blocking=True).float()
        y  = batch["Y"].to(device,  non_blocking=True).float()
        dt = batch.get("time", None)

        pred = model(x1, x2, x3, x4)

        # metric 계산 (lead_idx 지정 시 해당 lead만 슬라이스)
        pred_m = pred
        y_m    = y
        if lead_idx is not None:
            pred_m = _select_lead(pred, lead_idx)
            y_m    = _select_lead(y,    lead_idx)

        for name, fn in metric_fns.items():
            try:
                val = fn(pred_m, y_m, **(metric_params or {}))
            except TypeError:
                val = fn(pred_m, y_m)
            val = float(val.item()) if hasattr(val, "item") else float(val)
            if not torch.isfinite(torch.tensor(val)):
                val = 0.0
            metrics[name] += val

        if keep_pred_shape:
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
        else:
            preds.append(pred_m.detach().cpu())
            targets.append(y_m.detach().cpu())

        if dt is not None:
            if isinstance(dt, (list, tuple)):
                datetimes.extend([str(d).split("T")[0] for d in dt])
            else:
                datetimes.append(str(dt).split("T")[0])

        count += 1

    avg_metrics = {k: v / max(1, count) for k, v in metrics.items()}
    return avg_metrics, preds, targets, datetimes

def _decorate_geo_axes(ax, add_gridlines=True):
    """해안선, 경위도 틱/그리드, 전지구 extent."""
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)

    # 틱: -180..180 / -90..90
    lon_ticks = np.arange(-180, 181, 60)
    lat_ticks = np.arange(-90, 91, 30)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    # === 여기서 포맷터 강제 ===
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))

    # 폰트 크기 ↓
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(8)

    if add_gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, linestyle="dashed",
                          color="gray", alpha=0.6, zorder=6)
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

    ax.set_global()
def visualize_prediction(pred, true, base_time="", title_suffix="", save_path=None,
                         vmin_percentile=1.0, vmax_percentile=99.0, cmap="plasma"):
    """
    pred/true: Tensor (B, C=1, T, H, W) 가정. 첫 배치만 시각화.
    """
    pred = pred[0, 0].numpy()
    true = true[0, 0].numpy()
    T, H, W = true.shape

    fig, axes = plt.subplots(T, 2, figsize=(10, 3 * T), constrained_layout=True)
    fig.suptitle(f"t₀={base_time}  {title_suffix}", fontsize=18)

    for t in range(T):
        vmin = np.percentile(true[t], vmin_percentile)
        vmax = np.percentile(true[t], vmax_percentile)

        ax = axes[t, 0]
        im = ax.imshow(true[t], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT (t₀+{t+1})")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[t, 1]
        im = ax.imshow(pred[t], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred (t₀+{t+1})")
        fig.colorbar(im, ax=ax, shrink=0.8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_prediction_map(
    pred, true, base_time="", title_suffix="", save_path=None,
    vmin_percentile=1.0, vmax_percentile=99.0, cmap="plasma",
    lon=None, lat=None, central_longitude=180.0,
    add_gridlines=True, dpi=300,
):
    """
    pred/true: Tensor (B, C=1, T, H, W) 가정. 첫 배치만 시각화.
    lon, lat: 1D 경/위도 배열 (선택). 없으면 W,H 기반 균등격자 추정.
    0값을 흰색으로 표시하도록 수정됨.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs

    # ---- 데이터 준비 ----
    pred = pred[0, 0].detach().cpu().numpy()
    true = true[0, 0].detach().cpu().numpy()
    T, H, W = true.shape

    # ---- lon/lat 준비 (없으면 추정) ----
    if lon is None:
        lon = np.linspace(0.0, 360.0, W, endpoint=False)  # [0,360)
    if lat is None:
        lat = np.linspace(-90.0, 90.0, H)                 # [-90,90]

    # (선택) central_longitude가 180일 때 -180~180 범위로 정렬
    if central_longitude == 180.0:
        lon = (lon + 180.0) % 360.0 - 180.0
        sort_idx = np.argsort(lon)
        lon = lon[sort_idx]
        pred = pred[:, :, sort_idx]
        true = true[:, :, sort_idx]

    LON, LAT = np.meshgrid(lon, lat)

    # ---- 공통 vmin/vmax (GT 전체 시점 기준) ----
    vmin = np.percentile(true, vmin_percentile)
    vmax = np.percentile(true, vmax_percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(true), np.nanmax(true)

    # ---- cmap 수정: 0을 흰색으로 ----
    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    # 하위 1% 또는 절댓값 0 근처 값을 흰색으로 표시
    zero_thresh = max(vmin, 1e-6)
    n_white = int(256 * (zero_thresh / (vmax - vmin + 1e-9)))
    colors[:n_white] = white
    custom_cmap = mcolors.ListedColormap(colors)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree()

    # ---- Figure & Axes ----
    fig, axes = plt.subplots(
        T, 2,
        figsize=(10, 3.2 * T),
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )
    if T == 1:
        axes = np.array([axes])

    # --- 상단 제목 ---
    if base_time or title_suffix:
        fig.suptitle(f"t₀={base_time}  {title_suffix}")

    # ---- 패널 그리기 ----
    for t in range(T):
        # GT
        ax = axes[t, 0]
        m_left = ax.pcolormesh(
            LON, LAT, true[t], cmap=custom_cmap, vmin=vmin, vmax=vmax,
            transform=data_crs, shading="auto", zorder=3
        )
        ax.set_title("(c) GT")
        _decorate_geo_axes(ax, add_gridlines=add_gridlines)
        cbar = fig.colorbar(m_left, ax=ax, shrink=0.9, fraction=0.035, pad=0.02)
        cbar.set_label("Precipitation")

        # Pred
        ax = axes[t, 1]
        m_right = ax.pcolormesh(
            LON, LAT, pred[t], cmap=custom_cmap, vmin=vmin, vmax=vmax,
            transform=data_crs, shading="auto", zorder=3
        )
        ax.set_title("(d) MSE")
        _decorate_geo_axes(ax, add_gridlines=add_gridlines)
        cbar = fig.colorbar(m_right, ax=ax, shrink=0.9, fraction=0.035, pad=0.02)
        cbar.set_label("Precipitation")

    # ---- 저장/표시 ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()



def _decorate_geo_axes(ax, add_gridlines=True):
    """해안선, 경위도 틱/그리드, 전지구 extent.
       폰트 크기는 rcParams 설정을 따름."""
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)

    # 틱: -180..180 / -90..90
    lon_ticks = np.arange(-180, 181, 60)
    lat_ticks = np.arange(-90, 91, 30)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    # 포맷터 (폰트 크기 → rcParams 적용)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
    ax.tick_params(axis="both", which="major")

    if add_gridlines:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.5, linestyle="dashed",
            color="gray", alpha=0.6, zorder=6
        )
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

    ax.set_global()




################## LPC Test ######################
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import math
from functools import lru_cache
@torch.no_grad()
def _roll_longitude_to_center(img: torch.Tensor,
                              current_center_deg: float,
                              target_center_deg: float = 180.0) -> torch.Tensor:
    """
    경도 중앙을 current_center_deg에서 target_center_deg로 바꾸기 위해 roll 적용.
    img: (H,W) 또는 (B,1,H,W) 텐서
    """
    H, W = img.shape[-2], img.shape[-1]
    delta = target_center_deg - current_center_deg
    shift_cols = int(round((delta / 360.0) * W))
    if shift_cols % W == 0:
        return img
    return torch.roll(img, shifts=shift_cols, dims=-1)

def visualize_lpc_maps(
    pred,
    true,
    base_time="",
    title_suffix="",
    save_path=None,
    # LPC 파라미터
    scales = (1.0, 1.5, 2.0),
    weights = (1.0, -3.0, 2.0),
    norient=8,
    C=2.0,
    Beta_k=1e-4,
    B=None,
    # 그리기 옵션
    cmap="hot",
    show_diff=False,
    dpi=150,
    lpc_out_dir=None,
    extent=(0,360,-90,90),
    central_longitude=180,
):
    """
    pred/true: (B,1,T,H,W)
    - 계산 로직은 그대로, 단 그림만 경위도 축으로 표시.
    """
    gt = true[0,0].detach().cpu()  # (T,H,W)
    pr = pred[0,0].detach().cpu()
    T,H,W = gt.shape

    ncols = 3 if show_diff else 2
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig, axes = plt.subplots(
        T, ncols, figsize=(4.2*ncols, 3.2*T),
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )
    if T==1:
        axes = np.expand_dims(axes,0)

    fig.suptitle(f"[LPC] t₀={base_time}  {title_suffix}", fontsize=18)

    si_true_list, si_pred_list = [], []

    vmin_lpc = 0.0
    if lpc_out_dir:
        os.makedirs(lpc_out_dir, exist_ok=True)

    lon = np.linspace(extent[0], extent[1], W)
    lat = np.linspace(extent[2], extent[3], H)

    def decorate(ax):
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="dimgray", zorder=7)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=False,
            linewidth=0.5, linestyle="dashed",
            color="gray", alpha=0.6, zorder=6
        )
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

    for t in range(T):
        gt_img = _roll_longitude_to_center(gt[t], current_center_deg=0.0, target_center_deg=180.0)
        pr_img = _roll_longitude_to_center(pr[t], current_center_deg=0.0, target_center_deg=180.0)
        si_gt, lmap_gt = lpc_si_torch(gt_img.to(torch.float32),
                                  C=C,Beta_k=Beta_k,scales=scales,
                                  norient=norient,B=B,weights=weights)
        si_pr, lmap_pr = lpc_si_torch(pr_img.to(torch.float32),
                                  C=C,Beta_k=Beta_k,scales=scales,
                                  norient=norient,B=B,weights=weights)

        si_true_list.append(si_gt); si_pred_list.append(si_pr)

        vmax_lpc = max(torch.max(lmap_gt).item(), torch.max(lmap_pr).item())

        # --- GT ---
        ax = axes[t,0]
        im0 = ax.pcolormesh(lon, lat, lmap_gt.cpu().numpy(),
                            cmap=cmap, vmin=vmin_lpc, vmax=vmax_lpc,
                            transform=ccrs.PlateCarree(), shading="auto")
        ax.set_title(f"(a) GT ")
        decorate(ax); fig.colorbar(im0, ax=ax, shrink=0.8, orientation="horizontal")

        # --- Pred ---
        ax = axes[t,1]
        im1 = ax.pcolormesh(lon, lat, lmap_pr.cpu().numpy(),
                            cmap=cmap, vmin=vmin_lpc, vmax=vmax_lpc,
                            transform=ccrs.PlateCarree(), shading="auto")
        ax.set_title(f"(c) FACL+WACL Lv3 ")
        decorate(ax); fig.colorbar(im1, ax=ax, shrink=0.8, orientation="horizontal")

        if show_diff:
            ax = axes[t,2]
            diff = (lmap_pr - lmap_gt).cpu().numpy()
            dmax = np.percentile(np.abs(diff), 99.0)+1e-8
            im2 = ax.pcolormesh(lon, lat, diff,
                                cmap="coolwarm", vmin=-dmax, vmax=dmax,
                                transform=ccrs.PlateCarree(), shading="auto")
            ax.set_title(f"Diff (Pred-GT) (t₀+{t+1})")
            decorate(ax); fig.colorbar(im2, ax=ax, shrink=0.8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return si_true_list, si_pred_list


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
def lpc_si_torch(im, C=2.0, Beta_k=1e-4,
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
    return si, lpc_map



################## Frequency Profile ######################

# =========================
# Helpers
# =========================

def _fft2_shifted(img2d: np.ndarray):
    """2D FFT and shift to center."""
    return np.fft.fftshift(np.fft.fft2(img2d))

def _fftfreq_centers(N: int, d: float, unit: str):
    """fftshift된 '중심 주파수' 1D 벡터 반환."""
    f = np.fft.fftfreq(N, d=d)  # cycles/unit
    if unit == "rad":
        f = 2*np.pi*f           # radians/unit
    return np.fft.fftshift(f)

def _freq_axes(H: int, W: int, unit: str = "cyc", dy: float = 1.0, dx: float = 1.0):
    """fftshift된 주파수 축 (센터들)."""
    fy = _fftfreq_centers(H, dy, unit)  # y축
    fx = _fftfreq_centers(W, dx, unit)  # x축
    return fx, fy

def _freq_edges_from_centers(f: np.ndarray):
    """센터 f(오름차순 가정) → 셀 경계 edges(len=f+1)."""
    df = np.diff(f)
    left  = f[0]  - df[0]/2.0 if df.size else f[0] - 0.5
    right = f[-1] + df[-1]/2.0 if df.size else f[-1] + 0.5
    mids = (f[:-1] + f[1:]) / 2.0
    return np.concatenate([[left], mids, [right]])

def _mesh_edges(fx: np.ndarray, fy: np.ndarray):
    """1D edges → 2D pcolormesh용 그리드."""
    fx_e = _freq_edges_from_centers(fx)
    fy_e = _freq_edges_from_centers(fy)
    FXe, FYe = np.meshgrid(fx_e, fy_e)
    return FXe, FYe

def _nyquist(unit: str, dy: float, dx: float):
    """나이퀴스트 주파수 (fnx, fny)."""
    if unit == "cyc":
        return 1.0/(2*dx), 1.0/(2*dy)
    else:  # 'rad'
        return np.pi/dx, np.pi/dy

def _add_freq_guides(ax, unit: str, dy: float, dx: float, alpha=0.25, show_box=True):
    """중심축 + 나이퀴스트 가이드."""
    fnx, fny = _nyquist(unit, dy, dx)
    # 중심 십자선
    ax.axvline(0.0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.5)
    # 나이퀴스트 라인
    for x in (+fnx, -fnx):
        ax.axvline(x, color="k", lw=0.5, ls=":", alpha=alpha)
    for y in (+fny, -fny):
        ax.axhline(y, color="k", lw=0.5, ls=":", alpha=alpha)
    if show_box:
        ax.plot([-fnx, fnx, fnx, -fnx, -fnx],
                [-fny, -fny, fny,  fny,  -fny],
                color="k", lw=0.5, ls=":", alpha=alpha)

def _radial_bins(H: int, W: int, nbins: int = 128):
    """픽셀 좌표 반지름 bin (fftshift 기준 중심)."""
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, r.max() + 1e-9, nbins + 1)
    bins = np.digitize(r.ravel(), edges) - 1
    return r, bins, edges

def _radial_stat(a2d: np.ndarray, bins: np.ndarray, nbins: int, reducer="median", mask=None):
    """반지름 bin별 집계."""
    a = a2d.ravel()
    m = np.ones_like(a, dtype=bool) if mask is None else mask.ravel().astype(bool)
    out = np.full(nbins, np.nan, dtype=np.float64)
    for k in range(nbins):
        idx = (bins == k) & m
        if np.any(idx):
            out[k] = np.nanmedian(a[idx]) if reducer == "median" else np.nanmean(a[idx])
    return out

def _robust_centered_norm(arr2d: np.ndarray, center: float = 0.0, low_pct=1.0, high_pct=99.0):
    """발산형 컬러맵 robust 범위."""
    lo = np.nanpercentile(arr2d, low_pct)
    hi = np.nanpercentile(arr2d, high_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = np.nanmin(arr2d), np.nanmax(arr2d)
        if lo == hi:
            lo, hi = center - 1.0, center + 1.0
    return mcolors.TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)

def _amp_transform(ratio: np.ndarray, mode: str = "db", eps: float = 1e-8):
    """|Ŷ|/|Y| → 표시값."""
    if mode == "db":
        return 20.0 * np.log10(np.maximum(ratio, eps))
    if mode == "pct":
        return 100.0 * (ratio - 1.0)
    if mode == "linear":
        return ratio
    raise ValueError("amp_mode must be 'db'|'pct'|'linear'")

def _phase_coherence(Fp: np.ndarray, Ft: np.ndarray, mode: str = "cos"):
    """cos(Δφ) (또는 0~1 정규화)."""
    dphi = (np.angle(Fp) - np.angle(Ft) + np.pi) % (2*np.pi) - np.pi
    coh = np.cos(dphi)
    return 0.5*(coh + 1.0) if mode == "cos01" else coh

# =========================
# Main
# =========================

def visualize_freq_analysis(
    pred, true,
    base_time: str = "",
    title_suffix: str = "",
    nbins: int = 128,
    mask_percentile: float = 5.0,
    reducer: str = "median",
    amp_mode: str = "db",            # 'db' | 'pct' | 'linear'
    phase_mode: str = "cos",         # 'cos' | 'cos01'
    cmap_amp: str = "seismic",
    cmap_phase: str = "inferno",
    unit: str = "cyc",               # 'cyc' or 'rad'
    dy: float = 1.0, dx: float = 1.0,
    show_wavelength: bool = False,   # True → λ=1/k
    amp_profile_ylim=None,           # 예: (-12, 6)  (dB 고정)
    save_path: str = None,
    dpi: int = 300,
    add_freq_guides: bool = True
):
    """
    pred/true : torch.Tensor (B,1,T,H,W)  → 첫 배치 사용.
    - 맵: (1) Amplitude ratio, (2) Phase coherence
    - 프로파일: 원형 주파수 k 기준 amp & phase
    - 주파수 축은 pcolormesh(엣지 기반)로 정확히 매칭
    - 프로파일 x축 정렬(또는 λ 모드일 때 내림차순 정렬)로 꼬임 제거
    """
    # → numpy
    pred = pred[0, 0].detach().cpu().numpy()  # (T,H,W)
    true = true[0, 0].detach().cpu().numpy()
    T, H, W = true.shape

    # 주파수 축/엣지
    fx, fy = _freq_axes(H, W, unit=unit, dy=dy, dx=dx)   # centers
    FXe, FYe = _mesh_edges(fx, fy)                       # edges for pcolormesh
    FXc, FYc = np.meshgrid(fx, fy)                        # centers for k
    k_grid = np.sqrt(FXc**2 + FYc**2)

    # radial bins & 대표 k
    _, bins, _ = _radial_bins(H, W, nbins=nbins)
    k_centers = np.zeros(nbins, dtype=np.float64)
    k_flat = k_grid.ravel()
    for k in range(nbins):
        idx = (bins == k)
        k_centers[k] = np.nan if not np.any(idx) else np.nanmedian(k_flat[idx])

    # Figure
    fig, axes = plt.subplots(T, 3, figsize=(14, 4.8*T), constrained_layout=True)
    if T == 1:
        axes = axes[None, :]
    fig.suptitle(f"t₀={base_time}  {title_suffix}", fontsize=16)

    # 라벨 텍스트
    if show_wavelength:
        xlab_map = ("Wavelength x", "Wavelength y")
        xlab_prof = "Wavelength"
        unit_map = "(degree/cycle)" if unit == "cyc" else "(degree/rad)"
    else:
        xlab_map = ("Frequency x", "Frequency y")
        xlab_prof = "Radial frequency k"
        unit_map = "(cycles/degree)" if unit == "cyc" else "(rad/degree)"

    for t in range(T):
        Ft = _fft2_shifted(true[t])
        Fp = _fft2_shifted(pred[t])
        mag_t, mag_p = np.abs(Ft), np.abs(Fp)

        # 마스크 (작은 진폭 제외)
        thr_t = np.percentile(mag_t, mask_percentile)
        thr_p = np.percentile(mag_p, mask_percentile)
        mask_mag = (mag_t >= thr_t) & (mag_p >= thr_p)

        # (1) Amplitude ratio map
        ratio  = mag_p / (mag_t + 1e-8)
        ampmap = _amp_transform(ratio, mode=amp_mode, eps=1e-8)
        center = 0.0 if amp_mode in ("db", "pct") else 1.0
        amp_norm = _robust_centered_norm(ampmap, center=center, low_pct=1.0, high_pct=99.0)

        ax = axes[t, 0]
        pcm = ax.pcolormesh(FXe, FYe, ampmap, cmap=cmap_amp, norm=amp_norm, shading="auto")
        ax.set_aspect("equal") 
        ax.set_title(f"Amp ratio ({amp_mode})  (t₀+{t+1})")
        ax.set_xlabel(f"{xlab_map[0]} {unit_map}")
        ax.set_ylabel(f"{xlab_map[1]} {unit_map}")
        if add_freq_guides:
            _add_freq_guides(ax, unit, dy, dx)
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.85)
        cbar.set_label("Amplitude ratio" + (f" [{amp_mode}]" if amp_mode != "linear" else ""))

        # (2) Phase coherence map
        phmap = _phase_coherence(Fp, Ft, mode=phase_mode)
        vmin_phase = 0.0 if phase_mode == "cos01" else -1.0
        vmax_phase = 1.0

        ax = axes[t, 1]
        pcm = ax.pcolormesh(FXe, FYe, phmap, cmap=cmap_phase,
                            vmin=vmin_phase, vmax=vmax_phase, shading="auto")
        ax.set_aspect("equal") 
        ax.set_title(f"Cos phase diff  (t₀+{t+1})")
        ax.set_xlabel(f"{xlab_map[0]} {unit_map}")
        ax.set_ylabel(f"{xlab_map[1]} {unit_map}")
        if add_freq_guides:
            _add_freq_guides(ax, unit, dy, dx)
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.85)
        cbar.set_label("cos(Δφ)" if phase_mode == "cos" else "cos(Δφ) normalized")

        # (3) Radial profiles
        amp_prof   = _radial_stat(ampmap, bins, nbins, reducer=reducer, mask=mask_mag)
        phase_prof = _radial_stat(phmap,  bins, nbins, reducer=reducer, mask=mask_mag)

        # x축: k 또는 λ
        if show_wavelength:
            eps = 1e-12
            xvals = 1.0 / np.maximum(k_centers, eps)
            order = np.argsort(-xvals)  # 큰 파장 → 작은 파장
        else:
            xvals = k_centers.copy()
            order = np.argsort(xvals)   # 작은 k → 큰 k

        xvals  = xvals[order]
        amp_prof   = amp_prof[order]
        phase_prof = phase_prof[order]

        ax = axes[t, 2]
        ax2 = ax.twinx()
        l1, = ax.plot(xvals, amp_prof,   lw=2, color="C0",      label=f"Amp ratio [{amp_mode}]")
        l2, = ax2.plot(xvals, phase_prof, lw=2, color="#FF8800", label="cos(Δφ)")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f"{xlab_prof} {unit_map if not show_wavelength else '(unit)'}")

        # y축: 사용자 지정 or robust
        if amp_profile_ylim is not None:
            ax.set_ylim(*amp_profile_ylim)
        else:
            if np.isfinite(amp_prof).any():
                lo = np.nanpercentile(amp_prof, 1)
                hi = np.nanpercentile(amp_prof, 99)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = np.nanmin(amp_prof), np.nanmax(amp_prof)
                    if lo == hi: lo, hi = (lo - 1.0, hi + 1.0)
                ax.set_ylim(lo, hi)

        ax.set_ylabel("Amplitude ratio" + (f" [{amp_mode}]" if amp_mode != "linear" else ""))
        ax2.set_ylim((0, 1) if phase_mode == "cos01" else (-1, 1))
        ax2.set_ylabel("cos(Δφ)" if phase_mode == "cos" else "cos(Δφ) (0–1)")
        ax.tick_params(axis='y', colors="C0")
        ax2.tick_params(axis='y', colors="#FF8800")

        lines, labels = [l1, l2], [l1.get_label(), l2.get_label()]
        ax.legend(lines, labels, loc="best")
        ax.set_title(f"(b) FACL+WACL Lv3")

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()




