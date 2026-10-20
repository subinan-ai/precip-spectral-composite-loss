# evaluate_3d.py
from __future__ import annotations
import os
import yaml
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
    figsize=(5.5, 5.4),          # ✅ 맵 크기 약간 확대 (기존 16x6.5 → 조금 작지만 넓게)
    save_path=None, dpi=300,
    add_gridlines=True,
    bar_width=0.42,               # ✅ 살짝 얇게
    bar_color="steelblue",
    # ▼ 컬러바(스케일바) 옵션
    cbar_ticks=None,
    cbar_label="Normalized intensity (1-99th percentile)",
    tick_labelsize: int = 9,      # ✅ 글자 크기 살짝 줄임 (기존 10)
):
    import os, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
    from matplotlib import cm
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import torch

    # ---- 기본 스타일 ----
    plt.style.use('default')
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.facecolor']   = 'white'
    mpl.rcParams['figure.facecolor']    = 'white'
    mpl.rcParams['axes.facecolor']      = 'white'
    mpl.rcParams['axes.linewidth']      = 0.8

    # ---- 컬러맵 (NaN/0 → 흰색) ----
    magma_r_white = cm.get_cmap('magma_r').copy()
    magma_r_white.set_bad('white', 1.0)
    magma_r_white.set_under('white', 1.0)

    def to5d(x): return x.unsqueeze(2) if x.dim()==4 else x
    gt, pred = to5d(gt), to5d(pred)
    gt2d   = torch.nan_to_num(gt[b, c, t],   nan=0.0).float().cpu().numpy()
    pred2d = torch.nan_to_num(pred[b, c, t], nan=0.0).float().cpu().numpy()

    # ---- DTCWT 분해 ----
    _, pyr_gt = dtcwt_forward(gt2d, levels=levels)
    or_lv = pick_orient_per_level(pyr_gt)

    recon_gt, recon_pr = [], []
    for L in (1, 2, 3):
        if L > levels:
            recon_gt.append(None); recon_pr.append(None); continue
        recon_gt.append(reconstruct_single_high_band(gt2d,   levels, L, or_lv[L-1]))
        recon_pr.append(reconstruct_single_high_band(pred2d, levels, L, or_lv[L-1]))

    # ---- LL (저주파) ----
    rg_ll = pyr_gt.lowpass.real.astype(np.float32, copy=False)
    rp_ll = dtcwt_forward(pred2d, levels=levels)[1].lowpass.real.astype(np.float32, copy=False)

    # ---- LPIPS 바 값 ----
    bar_vals = [
        _lpips_scalar_global(rp_ll, rg_ll, lpips_model, crop_n=crop_n),
        _lpips_scalar_global(recon_pr[0], recon_gt[0], lpips_model, crop_n=crop_n) if recon_pr[0] is not None else np.nan,
        _lpips_scalar_global(recon_pr[1], recon_gt[1], lpips_model, crop_n=crop_n) if recon_pr[1] is not None else np.nan,
        _lpips_scalar_global(recon_pr[2], recon_gt[2], lpips_model, crop_n=crop_n) if recon_pr[2] is not None else np.nan,
    ]

    # ---- 정규화 맵 ----
    maps_for_plot = [
        pct_norm(rp_ll, *ll_percentiles),
        pct_norm(recon_pr[0], *high_percentiles) if recon_pr[0] is not None else None,
        pct_norm(recon_pr[1], *high_percentiles) if recon_pr[1] is not None else None,
        pct_norm(recon_pr[2], *high_percentiles) if recon_pr[2] is not None else None,
    ]

    titles = [
        "(a) Low frequency band",
        "(b) Lv1 High frequency band",
        "(c) Lv2 High frequency band",
        "(d) Lv3 High frequency band",
    ]

    # ---- Figure / Layout ----
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs  = fig.add_gridspec(2, 4, wspace=0.32, hspace=0.6, top=0.92, bottom=0.12)

    # ---- 축 꾸미기 (국경선 제거, 글자 크기 축소) ----
    def _decorate(ax):
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.45, color="dimgray", alpha=0.6, zorder=10)
        lon = np.arange(-180, 181, 60); lat = np.arange(-90, 91, 30)
        ax.set_xticks(lon, crs=ccrs.PlateCarree())
        ax.set_yticks(lat, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
        if add_gridlines:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.4, linestyle="dashed",
                              color="silver", alpha=0.65, zorder=6)
            gl.xlocator = mticker.FixedLocator(lon); gl.ylocator = mticker.FixedLocator(lat)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    # ---- 상단 4개 지도 ----
    last_im = None
    for i in range(4):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        if maps_for_plot[i] is None:
            ax.axis("off"); ax.set_title(titles[i]+" (N/A)", loc='left', fontsize=11)
            continue
        im = ax.imshow(
            maps_for_plot[i], origin="lower", extent=extent,
            transform=ccrs.PlateCarree(), cmap=magma_r_white,
            vmin=0, vmax=1, interpolation="nearest"
        )
        last_im = im
        ax.set_title(titles[i], loc='left', fontsize=10.5)
        _decorate(ax)

    # ---- 2행: 바그래프 + 컬러바 ----
    axb = fig.add_subplot(gs[1, 0])
    labels = ["LL", "Lv1", "Lv2", "Lv3"]; x = np.arange(len(labels))
    axb.bar(x, bar_vals, width=bar_width, color=bar_color)
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=9)
    axb.set_title("(e) LPIPS bar (mean over map)", loc='left', fontsize=12)
    axb.set_ylabel("LPIPS", fontsize=9)
    axb.grid(axis="y", linestyle=":", linewidth=0.4)
    axb.tick_params(axis='both', which='major', labelsize=9)
    for s in axb.spines.values():
        s.set_linewidth(0.8)

    ax_cbar = fig.add_subplot(gs[1, 1:3])
    ax_cbar.set_box_aspect(0.03)
    cbar = plt.colorbar(last_im, cax=ax_cbar, orientation='horizontal', ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=12, labelpad=3)
    cbar.ax.tick_params(labelsize=9, length=3, width=0.8, pad=2)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')

    # ---- 저장 ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()
        fig.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="white", pad_inches=0.02)
        plt.close(fig)
    else:
        plt.show()

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
    title_a: str = "MSE",
    title_b: str = "FACL",
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

    # (b) LPIPS(MSE,GT)
    im1 = axes[1].pcolormesh(lon, lat, lp_a, cmap=magma, vmin=0, vmax=vmax_a,
                             transform=ccrs.PlateCarree(), shading="auto")
    axes[1].set_title(f"{title_a}", fontsize=title_fontsize); _decorate(axes[1])

    # (c) ΔLPIPS = LPIPS(MSE,GT) − LPIPS(FACL,GT)
    im2 = axes[2].pcolormesh(lon, lat, lp_d, cmap=bwr,
                             vmin=-vmax_d, vmax=vmax_d,
                             transform=ccrs.PlateCarree(), shading="auto")
    axes[2].set_title(f"ΔLPIPS ({title_a} − {title_b})", fontsize=title_fontsize); _decorate(axes[2])

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
    extent=(0, 360, -90, 90),
    patch: int = 32, stride: int = 16,

    # ============================
    # ✅ 내부 계산용 이름(변경 X)
    # ============================
    title_a_internal: str = "FACL+WACL Lv3",
    title_b_internal: str = "MSE",

    # ============================
    # ✅ 그림 표시용 이름(원하는 대로)
    # ============================
    title_a_display: str = "FWCL Lv3",
    title_b_display: str = "MSE",

    # ✅ delta_order는 내부 계산용 이름을 기준으로 써야 안전함
    delta_order: str = "MSE - FACL+WACL Lv3",

    add_gridlines: bool = True,
    title_fontsize: int = 19,
    tick_labelsize: int = 15,
    dpi: int = 600,
    abs_diff: bool = False,
    vmax_percentile: float = 99.5,   # abs_diff=False일 때 자동 범위 산정용 (사용은 유지)
    fixed_delta_range: float = 0.05, # ✅ ΔLPIPS 고정 범위: [-0.05, 0.05]
    cbar_row_ratio: float = 0.03,    # ✅ 컬러바 행 높이(두께) 비율
    cbar_label_fontsize: int = 7,
    cbar_tick_fontsize: int = 7,
):
    """
    출력 형식:
      [ title_b_display ] [ title_a_display ] [ ΔLPIPS (B − A) ]
      아래:
      [ LPIPS 컬러바 ] [ ΔLPIPS 컬러바 ]

    ✅ A의 '계산 기준'은 title_a_internal (=FACL+WACL Lv3)로 유지
    ✅ 그림에 표시되는 이름만 title_a_display (=FWCL Lv3)로 변경
    ✅ ΔLPIPS 컬러바만 -fixed_delta_range ~ +fixed_delta_range 로 고정
    """

    import os
    import numpy as np
    import torch
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import matplotlib.ticker as mticker
    from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    from utils.config_utils import load_config
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model
    # 외부 제공:
    # - get_lpips_model(device)
    # - lpips_patch_map_single_frame(pred2d, gt2d, lpips_model, patch, stride)

    # ---------------------------
    # style
    # ---------------------------
    plt.style.use("default")
    mpl.rcParams.update({
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # ---------------------------
    # colormaps
    # ---------------------------
    def _cmap_lp():
        colors = ["#ffffff", "#2ca25f", "#2cb1a6", "#1f78b4", "#203a8f"]
        cm = LinearSegmentedColormap.from_list("w_g_teal_navy", colors, N=256)
        cm.set_bad("white", 1.0)
        cm.set_under("#f2f2f2")
        return cm

    def _cmap_div():
        colors = ["#4575b4", "#ffffff", "#d73027"]
        cm = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
        cm.set_bad("white", 1.0)
        return cm

    cmap_lp  = _cmap_lp()
    cmap_div = _cmap_div()

    # ---------------------------
    # LPIPS model
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = get_lpips_model(device=device).eval()

    # ---------------------------
    # load models
    # ---------------------------
    cfg_a = load_config(cfg_a_path)
    cfg_b = load_config(cfg_b_path)
    model_a = build_model(cfg_a).to(device).eval()
    model_b = build_model(cfg_b).to(device).eval()
    _ = load_checkpoint_for_eval(ckpt_a_path, model_a, device=device)
    _ = load_checkpoint_for_eval(ckpt_b_path, model_b, device=device)

    # A/B 샘플을 동일하게 로드하기 위해 A의 loader 사용
    test_loader = get_test_loader(cfg_a)

    preds_a, preds_b, targets = [], [], []
    processed = 0

    for batch in test_loader:
        if isinstance(batch, dict) and all(k in batch for k in ("X1", "X2", "Y")):
            x = (batch["X1"].to(device), batch["X2"].to(device))
            y = batch["Y"].to(device)
        else:
            x, y = batch
            if isinstance(x, (list, tuple)):
                x = tuple(_x.to(device) for _x in x)
            else:
                x = x.to(device)
            y = y.to(device)

        def _forward(model, inp):
            if isinstance(inp, (list, tuple)):
                return model(inp[0], inp[1])
            return model(inp)

        out_a = _forward(model_a, x)
        out_b = _forward(model_b, x)

        preds_a.append(out_a.detach().cpu())
        preds_b.append(out_b.detach().cpu())
        targets.append(y.detach().cpu())

        bs = x[0].size(0) if isinstance(x, (list, tuple)) else x.size(0)
        processed += bs
        if processed > sample_idx:
            break

    # ---------------------------
    # ensure (B,C,T,H,W)
    # ---------------------------
    def _ensure_5d(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 5:
            return t
        if t.dim() == 4:
            return t.unsqueeze(2)  # (B,C,H,W)->(B,C,1,H,W)
        raise ValueError(f"expect 4D/5D, got {tuple(t.shape)}")

    preds_a = _ensure_5d(torch.cat(preds_a, dim=0))
    preds_b = _ensure_5d(torch.cat(preds_b, dim=0))
    targets = _ensure_5d(torch.cat(targets, dim=0))

    gt = targets[sample_idx, ch_idx, t_idx].float().to(device)
    pa = preds_a[sample_idx, ch_idx, t_idx].float().to(device)  # A
    pb = preds_b[sample_idx, ch_idx, t_idx].float().to(device)  # B

    # ---------------------------
    # LPIPS maps (A vs GT, B vs GT)
    # ---------------------------
    lp_a = lpips_patch_map_single_frame(pa, gt, lpips_model, patch=patch, stride=stride)
    lp_b = lpips_patch_map_single_frame(pb, gt, lpips_model, patch=patch, stride=stride)

    # ---------------------------
    # delta order normalize
    # ---------------------------
    def _norm_delta_order(s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip()
        s = s.replace("−", "-").replace("–", "-").replace("—", "-")
        s = s.upper().replace(" ", "")
        return s

    order = _norm_delta_order(delta_order)

    # 내부 계산용 이름으로만 delta_order를 판별 (표시용 이름과 분리)
    aU = title_a_internal.upper().replace(" ", "")
    bU = title_b_internal.upper().replace(" ", "")

    if order == f"{bU}-{aU}":
        lp_d = lp_b - lp_a
        delta_title = f"{title_b_display} − {title_a_display}"
    elif order == f"{aU}-{bU}":
        lp_d = lp_a - lp_b
        delta_title = f"{title_a_display} − {title_b_display}"
    else:
        raise ValueError(
            f"delta_order not recognized: {delta_order!r}. "
            f"Use '{title_b_internal} - {title_a_internal}' or '{title_a_internal} - {title_b_internal}'. "
            f"(normalized='{order}')"
        )

    # ---------------------------
    # scales for LPIPS absolute (shared)
    # ---------------------------
    def _pos_percentile_minmax(arrs, p_lo=0.5, p_hi=99.5):
        vals = np.concatenate([np.asarray(a).ravel() for a in arrs])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return 1e-12, 1.0
        vmin = float(np.nanpercentile(vals, p_lo))
        vmax = float(np.nanpercentile(vals, p_hi))
        if vmin == vmax:
            vmax = vmin + 1e-9
        return vmin, vmax

    vmin_lp, vmax_lp = _pos_percentile_minmax([lp_a, lp_b])
    vmin_lp_plot = max(vmin_lp, 1e-12)

    # ---------------------------
    # delta scale
    # ---------------------------
    if abs_diff:
        lp_d_plot = np.abs(lp_d)
        lp_d_vis = np.ma.masked_invalid(lp_d_plot)
        norm_d = Normalize(vmin=0.0, vmax=float(fixed_delta_range))
        cmap_d = cmap_lp
        d_label = "|ΔLPIPS|"
    else:
        vmax_d = float(fixed_delta_range)
        lp_d_vis = np.ma.masked_invalid(lp_d)
        norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0.0, vmax=vmax_d)
        cmap_d = cmap_div
        d_label = "ΔLPIPS"

    # ---------------------------
    # axis decorator
    # ---------------------------
    def _decorate_geo_axes(ax):
        ax.coastlines(resolution="110m", linewidth=0.55, color="dimgray", alpha=0.6, zorder=10)
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        if add_gridlines:
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(), draw_labels=False,
                linewidth=0.5, linestyle="dashed",
                color="silver", alpha=0.7, zorder=6
            )
            gl.xlocator = mticker.FixedLocator(lon_ticks)
            gl.ylocator = mticker.FixedLocator(lat_ticks)
        ax.set_global()
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    # ---------------------------
    # layout
    # ---------------------------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    fig = plt.figure(figsize=(18.0, 3.7), facecolor="white", dpi=dpi)
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, cbar_row_ratio],
        hspace=0.22,
        wspace=0.18
    )

    ax_b   = fig.add_subplot(gs[0, 0], projection=proj)  # title_b_display (MSE)
    ax_a   = fig.add_subplot(gs[0, 1], projection=proj)  # title_a_display (FWCL Lv3)
    ax_del = fig.add_subplot(gs[0, 2], projection=proj)

    cgs = gs[1, :].subgridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.0], wspace=0.08)
    cax_lp = fig.add_subplot(cgs[0, 0])
    cax_d  = fig.add_subplot(cgs[0, 1])

    for cax in (cax_lp, cax_d):
        cax.set_facecolor("white")
        cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(False)

    # maps
    im_b = ax_b.imshow(
        np.ma.masked_invalid(lp_b), origin="lower", extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap_lp, vmin=vmin_lp_plot, vmax=vmax_lp,
        interpolation="nearest"
    )
    ax_b.set_title(title_b_display, fontsize=title_fontsize, loc="left")
    _decorate_geo_axes(ax_b)

    im_a = ax_a.imshow(
        np.ma.masked_invalid(lp_a), origin="lower", extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap_lp, vmin=vmin_lp_plot, vmax=vmax_lp,
        interpolation="nearest"
    )
    ax_a.set_title(title_a_display, fontsize=title_fontsize, loc="left")
    _decorate_geo_axes(ax_a)

    im_del = ax_del.imshow(
        lp_d_vis, origin="lower", extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap_d, norm=norm_d,
        interpolation="nearest"
    )
    ax_del.set_title(delta_title, fontsize=title_fontsize, loc="left")
    _decorate_geo_axes(ax_del)

    # colorbars
    cbar_lp = fig.colorbar(im_b, cax=cax_lp, orientation="horizontal")
    cbar_lp.set_label("LPIPS (↓)", fontsize=cbar_label_fontsize)
    cbar_lp.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_lp.outline.set_linewidth(0.6)

    cbar_d = fig.colorbar(im_del, cax=cax_d, orientation="horizontal")
    cbar_d.set_label(d_label, fontsize=cbar_label_fontsize)
    cbar_d.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_d.outline.set_linewidth(0.6)

    # save
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(
        out_png,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False
    )
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
    import os
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
        lead_idx=None,
        keep_pred_shape=True,
    )
    # ================= Lead-time별 metric 재평가 (CSV 전용) =================
    import copy
    import csv
    from collections import defaultdict
    logger.info("[Eval-3D] Computing lead-time–specific metrics (lead_idx-based)")

    exp_name = cfg.get("experiment_name", cfg.get("exp_name", "experiment"))
    csv_dir  = cfg["paths"].get("log_dir", out_dir)
    os.makedirs(csv_dir, exist_ok=True)

    leadtime_rows = []

    # 모든 lead에서 동일한 샘플 집합을 보장하기 위해 loader를 한 번만 생성
    cfg_fixed = copy.deepcopy(cfg)
    cfg_fixed.setdefault("data", {})
    cfg_fixed["data"]["label_offsets"] = [1, 2, 3]
    test_loader_fixed = get_test_loader(cfg_fixed)

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

    # CSV 저장
    csv_path = os.path.join(csv_dir, f"metrics_by_leadtime_{exp_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=leadtime_rows[0].keys())
        writer.writeheader()
        for r in leadtime_rows:
            writer.writerow(r)

    logger.info(f"[Eval-3D] Saved lead-time metrics CSV → {csv_path}")

    # =======================================================================
    # ── 음수 강수 분석 (Negative Precipitation Analysis) ───────────────────
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        norm_tp  = cfg.get("normalization", {}).get("tp", {})
        mu_tp    = float(norm_tp.get("mean", 0.002479317132383585))
        sig_tp   = float(norm_tp.get("std",  0.005952117739409208))

        # preds: list of (B,1,T,H,W) tensors in z-score space
        # → mm/day 변환 후 음수 픽셀 집계
        neg_vals_per_lead = {}   # lead_idx(0/1/2) -> 1D numpy array of negative mm values
        total_per_lead    = {}

        for pred_batch in preds:
            arr = pred_batch.float().numpy()            # (B,1,T,H,W) or (B,T,H,W)
            if arr.ndim == 5:
                arr = arr[:, 0, :, :, :]               # (B,T,H,W)
            T_len = arr.shape[1]
            for t in range(T_len):
                mm = (arr[:, t, :, :] * sig_tp + mu_tp) * 1000.0   # (B,H,W) mm/day
                neg_mask = mm < 0.0
                total_per_lead[t]  = total_per_lead.get(t, 0)  + mm.size
                if t not in neg_vals_per_lead:
                    neg_vals_per_lead[t] = []
                neg_vals_per_lead[t].append(mm[neg_mask])

        # ── 콘솔 / 로그 출력 ──────────────────────────────────────────────
        logger.info("[NegPrecip] === Negative Precipitation Summary ===")
        all_neg = []
        summary_rows = []
        for t in sorted(neg_vals_per_lead.keys()):
            neg_arr = np.concatenate(neg_vals_per_lead[t]) if neg_vals_per_lead[t] else np.array([])
            n_neg   = int(neg_arr.size)
            n_total = int(total_per_lead[t])
            ratio   = n_neg / n_total * 100 if n_total > 0 else 0.0
            min_val = float(neg_arr.min()) if n_neg > 0 else 0.0
            mean_val= float(neg_arr.mean()) if n_neg > 0 else 0.0
            logger.info(
                f"  lead t+{t+1}: neg_count={n_neg:,} / {n_total:,} "
                f"({ratio:.3f}%)  min={min_val:.4f}  mean={mean_val:.4f} mm/day"
            )
            all_neg.append(neg_arr)
            summary_rows.append({
                "lead_time": t + 1,
                "neg_count": n_neg,
                "total_count": n_total,
                "neg_ratio_pct": round(ratio, 6),
                "min_mm": round(min_val, 6),
                "mean_mm": round(mean_val, 6),
            })

        # ── CSV 저장 ──────────────────────────────────────────────────────
        neg_csv_path = os.path.join(csv_dir, f"neg_precip_{exp_name}.csv")
        with open(neg_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"[NegPrecip] Saved summary CSV → {neg_csv_path}")

        # ── 히스토그램 저장 ───────────────────────────────────────────────
        all_neg_concat = np.concatenate(all_neg) if any(a.size > 0 for a in all_neg) else np.array([])
        if all_neg_concat.size > 0:
            vis_dir_neg = cfg["paths"].get("eval_dir", out_dir)
            os.makedirs(vis_dir_neg, exist_ok=True)
            hist_path = os.path.join(vis_dir_neg, f"neg_precip_hist_{exp_name}.png")

            fig, axes = plt.subplots(1, len(neg_vals_per_lead), figsize=(5 * len(neg_vals_per_lead), 4),
                                     sharey=False, dpi=150)
            if len(neg_vals_per_lead) == 1:
                axes = [axes]

            for ax, (t, row) in zip(axes, zip(sorted(neg_vals_per_lead.keys()), summary_rows)):
                neg_arr = np.concatenate(neg_vals_per_lead[t]) if neg_vals_per_lead[t] else np.array([])
                if neg_arr.size == 0:
                    ax.text(0.5, 0.5, "No negatives", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.hist(neg_arr, bins=50, color="#e63946", edgecolor="white", linewidth=0.3)
                ax.set_title(f"lead t+{t+1}\n"
                             f"neg {row['neg_count']:,} / {row['total_count']:,} "
                             f"({row['neg_ratio_pct']:.3f}%)", fontsize=9)
                ax.set_xlabel("mm/day", fontsize=8)
                ax.set_ylabel("count", fontsize=8)
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

            fig.suptitle(f"Negative Precipitation  [{exp_name}]", fontsize=10, y=1.01)
            plt.tight_layout()
            plt.savefig(hist_path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[NegPrecip] Saved histogram → {hist_path}")
        else:
            logger.info("[NegPrecip] No negative precipitation values found.")

    except Exception as _neg_e:
        logger.warning(f"[NegPrecip] Skipped: {_neg_e}")

    # ── 전체 모델 음수 강수 비교 (MSE 실험 실행 시 한 번만) ────────────────
    if "mse_r90p_r95p" in exp_name:
        try:
            compare_neg_precip_all_models(
                out_log_dir  = csv_dir,
                out_eval_dir = cfg["paths"].get("eval_dir", out_dir),
                logger       = logger,
            )
        except Exception as _neg_all_e:
            logger.warning(f"[NegPrecip-All] Skipped: {_neg_all_e}")

    # =======================================================================

    # ── Regional CDF (MSE / FACL / WFCL vs Obs) ─────────────────────────────
    try:
        _BASE = os.path.dirname(os.path.abspath(__file__))
        plot_regional_cdf(
            out_dir=out_dir,
            cfg_mse_path=os.path.join(
                _BASE, "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml"),
            cfg_facl_path=os.path.join(
                _BASE, "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml"),
            cfg_wfcl_path=os.path.join(
                _BASE, "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml"),
        )
        logger.info(f"[Eval-3D] Regional CDF saved → {out_dir}/regional_cdf_mse_facl_wfcl.png")
    except Exception as _e:
        logger.warning(f"[Eval-3D] Regional CDF skipped: {_e}")

    # =======================================================================

    # 5) 로그 출력
    logger.info(f"[Eval] ckpt epoch={epoch}")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.6f}")

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
        ''' RHD 코드
        save_path = os.path.join(
            vis_dir,
            f"Map2_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        metrics = visualize_prediction_map(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,

            # ---- 시각화 옵션 ----
            vmin_percentile=1.0,
            vmax_percentile=99.0,
            central_longitude=180.0,
            add_gridlines=True,
            dpi=300,
            palette="white_blue_yellow_red",
            tick_labelsize=11,

            # ---- RHD 관련 ----
            model_name=loss_name.upper(),   # 예: "MSE", "FACL", "FACL+WACL"
            rhd_psz=15,
            rhd_bins=50,
            rhd_robust=True,
            rhd_device="cpu",       # 안전 / GPU 여유 있으면 "pred"
        )

        logger.info(
            f"[Eval-3D] Saved Map2 visualization → {save_path} | "
            f"RHD(t): {['%.3f' % v for v in metrics['rhd']]}"
        )
        '''
        save_path = os.path.join(
            vis_dir,
            f"Map2_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        metrics = visualize_prediction_map(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,

            # ---- 시각화 옵션 ----
            vmin_percentile=1.0,
            vmax_percentile=99.0,
            central_longitude=180.0,
            add_gridlines=True,
            dpi=300,
            palette="white_blue_yellow_red",
            tick_labelsize=11,

            # ---- 표기/모델명 ----
            model_name=loss_name.upper(),

            # ---- ✅ FSS 옵션 (네 fss() 구현 기준) ----
            fss_threshold=1.0,   # mm/day (원하면 cfg에서 가져와도 됨)
            fss_window=5,        # window size
            fss_decimals=3,
        )

        logger.info(
            f"[Eval-3D] Saved Map2 visualization → {save_path} | "
            f"FSS(t): {['%.3f' % v for v in metrics['fss']]} | "
            f"(thr={metrics['fss_threshold']}, win={metrics['fss_window']})"
        )

        ################## Map 3 (4-panel: GT + MSE + FACL + WFCL) ########################
        try:
            from utils.save_utils import load_checkpoint_for_eval as _lcfe_map3
            _map3_compare = cfg.get("freq_snr_comparison", [])
            _map3_model_list = []
            for _e in _map3_compare:
                _mm = build_model(cfg).to(device).float()
                _lcfe_map3(_e["ckpt"], _mm, device=device)
                _mm.eval()
                _map3_model_list.append((_e.get("label", "model"), _mm))

            _map3_loader2 = get_test_loader(cfg)
            _map3_b = next(iter(_map3_loader2))
            _map3_x1 = _map3_b["X1"].to(device).float()
            _map3_x2 = _map3_b.get("X2")
            _map3_x2 = _map3_x2.to(device).float() if isinstance(_map3_x2, torch.Tensor) \
                        else torch.zeros_like(_map3_x1)
            _map3_true = _map3_b["Y"].cpu()

            _map3_pl = []
            with torch.no_grad():
                for _lbl, _mm in _map3_model_list:
                    _map3_pl.append((_mm(_map3_x1, _map3_x2).detach().cpu(), _lbl))

            _map3_save = os.path.join(vis_dir, f"Map3_{exp}_{base_time}_ep{epoch}.png")
            visualize_prediction_map_multimodel(
                true=_map3_true,
                preds_list=_map3_pl,
                base_time=base_time,
                title_suffix=f"[{exp}]",
                save_path=_map3_save,
                vmin_percentile=1.0, vmax_percentile=99.0, vmax_override=45.0,
                central_longitude=180.0, add_gridlines=True, dpi=300,
                palette="precip_jma", tick_labelsize=11,
                mu_tp=0.002479317132383585, sig_tp=0.005952117739409208,
                rhd_psz=15, rhd_bins=50, rhd_decimals=3,
            )
            logger.info(f"[Eval-3D] Saved Map3 visualization → {_map3_save}")

            # Map3 1-day lead time: GT + models in a single row
            _map3_ld1_save = os.path.join(vis_dir, f"Map3_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
            visualize_multimodel_single_lead(
                true=_map3_true,
                preds_list=_map3_pl,
                t_idx=0,
                base_time=base_time,
                save_path=_map3_ld1_save,
                vmax_override=45.0,
                central_longitude=180.0, add_gridlines=True, dpi=300,
                palette="precip_jma", tick_labelsize=16,
                title_fontsize=19.5,
                mu_tp=0.002479317132383585, sig_tp=0.005952117739409208,
                rhd_psz=15, rhd_bins=50, rhd_decimals=3,
            )
            logger.info(f"[Eval-3D] Saved Map3 1-day visualization → {_map3_ld1_save}")
        except Exception as _map3_reg_e:
            logger.warning(f"[Eval-3D] Map3 skipped: {_map3_reg_e}")

        #====================빨간 박스 영역만 고해상도==============
        '''
        import numpy as np

        # --- 줌 영역 설정 (필요 시 변경 가능) ---
        extent_zoom = (270, 350, -10, 30)

        # --- 저장 경로 ---
        pair_path = os.path.join(
            vis_dir, f"Map2_zoom_pair_{exp}_{loss_name}_{base_time}_ep{epoch}_600dpi.png"
        )

        # --- 좌표 계산 (H,W 기반) ---
        H, W = targets[0].shape[-2], targets[0].shape[-1]
        lon = np.linspace(0.0, 360.0, W, endpoint=False)
        lat = np.linspace(-90.0, 90.0,  H)

        # --- 고해상도 줌 시각화 (GT vs Pred) ---
        zoom_metrics = save_highres_zoom_GT_vs_pred(
            pred=preds[0],                     # (B,C,T,H,W) 또는 (B,T,H,W)
            true=targets[0],
            ch_idx=0,
            t_idx=0,
            lon=lon, lat=lat,
            extent_zoom=extent_zoom,
            central_longitude=180.0,

            # ✅ 타이틀: (a)/(b) prefix만 주고, 모델명+RHD는 함수가 붙임
            left_title="Ground Truth",
            right_title=" ",
            model_name=str(loss_name).upper(),   # ✅ 소문자 방지 + "MSE (RHD=...)" 형태

            palette="white_blue_yellow_red",
            vmin_pct=1.0, vmax_pct=99.0,
            mu_tp=0.002479317132383585,
            sig_tp=0.005952117739409208,
            clamp_zero=True,

            # ✅ RHD 설정(표와 동일)
            rhd_psz=15,
            rhd_bins=50,
            rhd_robust=True,
            rhd_device="cpu",                   # GPU 여유 있으면 "pred"

            save_path=pair_path,
            dpi=600,
            tick_labelsize=14,
            title_fontsize=19.5,
        )

        logger.info(
            f"[Eval-3D] Saved high-res zoom pair → {pair_path} | "
            f"Zoom-RHD={zoom_metrics['rhd_zoom']:.3f}"
        )
        '''
        import numpy as np

        # --- 줌 영역 설정 (필요 시 변경 가능) ---
        extent_zoom = (130, 210, -15, 25)

        # --- 좌표 계산 (H,W 기반) ---
        H, W = targets[0].shape[-2], targets[0].shape[-1]
        lon = np.linspace(0.0, 360.0, W, endpoint=False)
        lat = np.linspace(-90.0, 90.0, H)

        ################## Map3 4-panel zoom (GT + MSE + FACL + WFCL) ########################
        try:
            from utils.save_utils import load_checkpoint_for_eval as _lcfe_m3
            _compare_list_map3 = cfg.get("freq_snr_comparison", [])
            _m3_models = []
            for _entry in _compare_list_map3:
                _m3 = build_model(cfg).to(device).float()
                _lcfe_m3(_entry["ckpt"], _m3, device=device)
                _m3.eval()
                _m3_models.append((_entry.get("label", "model"), _m3))

            _m3_loader = get_test_loader(cfg)
            _m3_batch  = next(iter(_m3_loader))
            _m3_x1 = _m3_batch["X1"].to(device).float()
            _m3_x2 = _m3_batch.get("X2")
            _m3_x2 = _m3_x2.to(device).float() if isinstance(_m3_x2, torch.Tensor) \
                      else torch.zeros_like(_m3_x1)
            _m3_true = _m3_batch["Y"].cpu()

            _m3_preds_list = []
            with torch.no_grad():
                for _lbl, _m3m in _m3_models:
                    _m3_preds_list.append((_m3m(_m3_x1, _m3_x2).detach().cpu(), _lbl))

            _m3_path = os.path.join(
                vis_dir, f"Map3_4panel_{exp}_{base_time}_ep{epoch}_600dpi.png"
            )
            save_highres_zoom_4panel(
                true=_m3_true,
                preds_list=_m3_preds_list,
                ch_idx=0, t_idx=0,
                lon=lon, lat=lat,
                extent_zoom=extent_zoom,
                central_longitude=180.0,
                palette="precip_jma",
                vmin_pct=1.0, vmax_pct=99.0, vmax_override=45.0,
                mu_tp=0.002479317132383585,
                sig_tp=0.005952117739409208,
                clamp_zero=True,
                rhd_psz=15, rhd_bins=50,
                rhd_robust=True, rhd_device="cpu",
                save_path=_m3_path,
                dpi=600, tick_labelsize=14, title_fontsize=19.5,
            )
            logger.info(f"[Eval-3D] Saved Map3 4-panel → {_m3_path}")
        except Exception as _map3_e:
            logger.warning(f"[Eval-3D] Map3 4-panel skipped: {_map3_e}")

        # --- 저장 경로 ---
        pair_path = os.path.join(
            vis_dir, f"Map2_zoom_pair_{exp}_{loss_name}_{base_time}_ep{epoch}_600dpi.png"
        )

        # --- 고해상도 줌 시각화 (GT vs Pred, ✅ FSS 기반) ---
        zoom_metrics = save_highres_zoom_GT_vs_pred(
            pred=preds[0],                     # (B,C,T,H,W) 또는 (B,T,H,W)
            true=targets[0],
            ch_idx=0,
            t_idx=0,
            lon=lon,
            lat=lat,
            extent_zoom=extent_zoom,
            central_longitude=180.0,

            # --- 타이틀 ---
            left_title="Ground Truth",
            right_title="Prediction",
            model_name=str(loss_name).upper(),

            # --- ✅ FSS 옵션 (LPC 대체) ---
            fss_threshold=1.0,
            fss_window=5,
            fss_decimals=3,

            # --- 시각화 옵션 ---
            palette="white_blue_yellow_red",
            vmin_pct=1.0,
            vmax_pct=99.0,
            mu_tp=0.002479317132383585,
            sig_tp=0.005952117739409208,
            clamp_zero=True,

            save_path=pair_path,
            dpi=600,
            tick_labelsize=14,
            title_fontsize=19.5,
        )

        # --- 로그 출력 (✅ FSS 기준) ---
        logger.info(
            f"[Eval-3D] Saved high-res zoom pair → {pair_path} | "
            f"FSS(zoom)={zoom_metrics['fss_zoom']:.3f} "
            f"(thr={zoom_metrics['fss_threshold']}, win={zoom_metrics['fss_window']})"
        )

        ################## LPC ########################
       # 저장 경로 구성
        save_path = os.path.join(
            vis_dir,
            f"LPC_{exp}_{loss_name}_{base_time}_ep{epoch}.png"
        )

        # LPC 맵 시각화 호출
        visualize_lpc_maps(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,
            dpi=600,
            crop_to_60=True,            # (선택) 60S~60N 크롭 보기
            show_lpc_mean=True,         # ✅ 우측 상단에 LPC mean 표시
            lpc_mean_scope="global",    # "view"로 하면 크롭된 영역만 평균 계산
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

        ################## SNR Map ########################
        try:
            norm_tp = cfg.get("normalization", {}).get("tp", {})
            snr_save_path = os.path.join(vis_dir, f"SNR_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
            visualize_snr_comparison(
                preds=preds,
                targets=targets,
                save_path=snr_save_path,
                mu_tp=float(norm_tp.get("mean", 0.002479317132383585)),
                sig_tp=float(norm_tp.get("std",  0.005952117739409208)),
                lead_labels=("t+1", "t+2", "t+3"),
                title_suffix=f"[{exp} | {loss_name}]",
                dpi=300,
            )
            logger.info(f"[Eval-3D] Saved SNR visualization → {snr_save_path}")
        except Exception as _snr_e:
            logger.warning(f"[Eval-3D] SNR visualization skipped: {_snr_e}")

        ################## Frequency-wise SNR ########################
        try:
            norm_tp = cfg.get("normalization", {}).get("tp", {})
            freq_snr_save_path = os.path.join(vis_dir, f"FreqSNR_{exp}_{loss_name}_{base_time}_ep{epoch}.png")
            visualize_freq_snr_comparison(
                models_data=[(preds, targets, loss_name.upper())],
                save_path=freq_snr_save_path,
                mu_tp=float(norm_tp.get("mean", 0.002479317132383585)),
                sig_tp=float(norm_tp.get("std",  0.005952117739409208)),
                lead_labels=("t+1", "t+2", "t+3"),
                title_suffix=f"[{exp} | {loss_name}]",
                dpi=300,
            )
            logger.info(f"[Eval-3D] Saved Frequency-wise SNR visualization → {freq_snr_save_path}")
        except Exception as _freq_snr_e:
            logger.warning(f"[Eval-3D] Frequency-wise SNR visualization skipped: {_freq_snr_e}")

        ################## Multi-model Frequency-wise SNR ########################
        _compare_list = cfg.get("freq_snr_comparison", [])
        if _compare_list:
            try:
                from utils.save_utils import load_checkpoint_for_eval as _lcfe
                _norm_tp = cfg.get("normalization", {}).get("tp", {})
                _mu_tp   = float(_norm_tp.get("mean", 0.002479317132383585))
                _sig_tp  = float(_norm_tp.get("std",  0.005952117739409208))

                # 비교 모델 모두 로드
                _cmp_models = []
                for _entry in _compare_list:
                    _m = build_model(cfg).to(device).float()
                    _lcfe(_entry["ckpt"], _m, device=device)
                    _m.eval()
                    _cmp_models.append((_entry.get("label", "model"), _m))

                # 한 번의 패스로 모든 모델 추론
                _all_preds   = [[] for _ in _cmp_models]
                _all_targets = []
                _cmp_loader  = get_test_loader(cfg)
                with torch.no_grad():
                    for _batch in _cmp_loader:
                        _x1 = _batch["X1"].to(device).float()
                        _x2 = _batch.get("X2")
                        _x2 = _x2.to(device).float() if isinstance(_x2, torch.Tensor) else torch.zeros_like(_x1)
                        _y  = _batch["Y"].cpu()
                        _all_targets.append(_y)
                        for _i, (_, _m) in enumerate(_cmp_models):
                            _all_preds[_i].append(_m(_x1, _x2).detach().cpu())

                _models_data = [
                    (_all_preds[i], _all_targets, lbl)
                    for i, (lbl, _) in enumerate(_cmp_models)
                ]
                _multi_snr_path = os.path.join(vis_dir, f"FreqSNR_multi_{exp}_ep{epoch}.png")
                visualize_freq_snr_comparison(
                    models_data=_models_data,
                    save_path=_multi_snr_path,
                    mu_tp=_mu_tp, sig_tp=_sig_tp,
                    lead_labels=("t+1", "t+2", "t+3"),
                    dpi=300,
                )
                logger.info(f"[Eval-3D] Saved multi-model FreqSNR → {_multi_snr_path}")
            except Exception as _multi_e:
                logger.warning(f"[Eval-3D] Multi-model FreqSNR skipped: {_multi_e}")

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
        # 1) LPIPS(A,GT), LPIPS(B,GT), Δ(B−A) 패널 저장
        save_path = os.path.join(
            vis_dir, f"LPIPS_AB_Delta_{exp}_{base_time}_ep{epoch}.png"
        )

        # (선택) 경로 sanity check
        for p in [
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_r90p_r95p/checkpoints/best_val_lpips.pt",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/checkpoints/best_val_lpips.pt",
        ]:
            assert isinstance(p, str) and os.path.exists(p), f"Not found: {p}"

        # ➜ A = FACL+WACL Lv3 (내부 계산), B = MSE
        make_lpips_ab_and_delta_map(
            cfg_a_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
            cfg_b_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
            ckpt_a_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/checkpoints/best_val_lpips.pt",
            ckpt_b_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_r90p_r95p/checkpoints/best_val_lpips.pt",
            out_png=save_path,

            # --- sample 설정 ---
            sample_idx=0,
            ch_idx=0,
            t_idx=0,              # ✅ 1-day lead

            # --- 내부 계산용 이름 (변경 ❌) ---
            title_a_internal="FACL+WACL Lv3",
            title_b_internal="FACL",

            # --- 그림에 표시될 이름 (변경 ⭕) ---
            title_a_display="WFCL",
            title_b_display="FACL",

            # --- Δ 정의 (내부 이름 기준) ---
            delta_order="FACL - FACL+WACL Lv3",

            # --- 시각화 옵션 ---
            central_longitude=180.0,
            extent=(0, 360, -90, 90),
            patch=32,
            stride=16,
            abs_diff=False,        # |Δ|로 보고 싶으면 True
            vmax_percentile=99.5,
        )

        logger.info(f"[Eval-3D] Saved LPIPS A/B & Δ map → {save_path}")

        # ============================================================
        # LPIPS grid: loss × lead time
        #   columns : [2-day] [3-day] [3-day − 1-day]
        #   rows    : MSE / FACL / FACL+WACL Lv3
        # ============================================================

        # ============================================================
        # LPIPS grid (rows=loss, cols=2-day / 3-day / (3-day-1-day))
        # ============================================================
        save_path = os.path.join(
            vis_dir,
            f"LPIPS_Grid_Loss_LeadTime_{exp}_{base_time}_ep{epoch}.png"
        )

        # (선택) 경로 sanity check
        for p in [
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_mse_r90p_r95p/checkpoints/best_val_lpips.pt",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_r90p_r95p/checkpoints/best_val_lpips.pt",
            "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/checkpoints/best_val_lpips.pt",
        ]:
            assert isinstance(p, str) and os.path.exists(p), f"Not found: {p}"

        plot_lpips_grid_by_loss_and_leadtime(
            # --------------------------------------------------------
            # loss specifications (rows)
            # --------------------------------------------------------
            loss_specs=[
                {
                    "name": "MSE",
                    "cfg": "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/"
                        "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
                    "ckpt": "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/"
                            "experiments/subin_an/conv3d_mse_r90p_r95p/checkpoints/best_val_lpips.pt",
                },
                {
                    "name": "FACL",
                    "cfg": "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/"
                        "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
                    "ckpt": "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/"
                            "experiments/subin_an/conv3d_facl_r90p_r95p/checkpoints/best_val_lpips.pt",
                },
                {
                    "name": "WFCL",
                    "cfg": "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/"
                        "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
                    "ckpt": "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/"
                            "experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/checkpoints/best_val_lpips.pt",
                },
            ],

            # --------------------------------------------------------
            # output
            # --------------------------------------------------------
            out_png=save_path,

            # --------------------------------------------------------
            # data selection
            # --------------------------------------------------------
            sample_idx=0,
            ch_idx=0,

            # --------------------------------------------------------
            # lead time indices (T=3 기준)
            #   t1_idx = 1-day
            #   t2_idx = 2-day
            #   t3_idx = 3-day
            # --------------------------------------------------------
            t1_idx=0,
            t2_idx=1,
            t3_idx=2,

            # --------------------------------------------------------
            # LPIPS patch parameters
            # --------------------------------------------------------
            patch=32,
            stride=16,

            # --------------------------------------------------------
            # map display
            # --------------------------------------------------------
            central_longitude=180.0,
            extent=(0, 360, -90, 90),
            add_gridlines=True,

            # --------------------------------------------------------
            # text sizes
            # --------------------------------------------------------
            tick_labelsize=10,
            title_fontsize=13,
            suptitle_fontsize=12,
            row_label_fontsize=13,

            # --------------------------------------------------------
            # color scaling
            # --------------------------------------------------------
            lpips_vmin=None,
            lpips_vmax=None,
            lpips_p_lo=0.5,
            lpips_p_hi=99.5,

            # Δ(3-day − 1-day) 컬럼 설정 (고정 범위)
            delta_fixed_range=(-0.05, 0.05),
            abs_delta=False,

            # --------------------------------------------------------
            # figure layout
            # --------------------------------------------------------
            figsize=(13.8, 6.8),
            cbar_row_ratio=0.12,
            cbar_h="55%",
            cbar_lp_w="60%",     # LPIPS 컬러바 길이(1~2열)
            cbar_d_w="26%",      # Δ 컬러바 길이(3열)
            cbar_gap=0.12,
            wspace=0.26,
            hspace=0.30,
            dpi=600,
        )

        logger.info(f"[Eval-3D] Saved LPIPS grid by loss & lead time → {save_path}")


        #============Lead time에 따른 메트릭별 그래프=========================
        # =========================
        import os

        PROJECT_ROOT = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2"

        save_path = os.path.join(
            vis_dir,
            f"Metric_LeadTime_Grid_{exp}_{base_time}_ep{epoch}.png"
        )
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        csv_specs = [
            {
                "name": "MSE",
                "csv": os.path.join(PROJECT_ROOT, "experiments/subin_an/conv3d_mse_r90p_r95p/logs/metrics_by_leadtime_conv3d_mse_r90p_r95p_fss_1.csv"),
            },
            {
                "name": "FACL",
                "csv": os.path.join(PROJECT_ROOT, "experiments/subin_an/conv3d_facl_r90p_r95p/logs/metrics_by_leadtime_conv3d_facl_r90p_r95p_fss_1.csv"),
            },
            {
                "name": "FACL+WACL Lv3",
                "csv": os.path.join(PROJECT_ROOT, "experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/logs/metrics_by_leadtime_conv3d_facl_wacl_lv3_r90p_r95p_fss_1.csv"),
            },
        ]

        import csv as _csv_mod

        def _load_metrics_from_csv(csv_path):
            rows = []
            with open(csv_path, newline="") as f:
                for row in _csv_mod.DictReader(f):
                    rows.append(row)
            rows.sort(key=lambda r: int(r["lead_time"]))
            skip = {"model", "lead_time"}
            return {k: [float(r[k]) for r in rows] for k in rows[0] if k not in skip}

        metrics_by_model = {}
        for spec in csv_specs:
            csv_path = spec["csv"]
            if not os.path.exists(csv_path):
                logger.warning(f"[Eval-3D] CSV not found, skipping: {csv_path}")
                continue
            metrics_by_model[spec["name"]] = _load_metrics_from_csv(csv_path)

        print(metrics_by_model)

        # ✅ "표시 이름만" 바꾸는 매핑 (키는 그대로!)
        model_label_map = {
            "MSE": "MSE",
            "FACL": "FACL",
            "FACL+WACL Lv3": "WFCL",
        }

        # ✅ 그리드 플롯 저장 (아래 함수 구현에 맞춘 인자만 전달)
        plot_metrics_by_leadtime_grid_3x2(
            metrics_by_model=metrics_by_model,
            out_png=save_path,
            lead_labels=("1-day", "2-day", "3-day"),
            model_order=("MSE", "FACL", "FACL+WACL Lv3"),
            metric_order=("MSE", "LPIPS", "CPBD", "LPC", "RHD", "CSI-R95p", "FSS", "HD"),
            figsize=(10, 10),
            dpi=600,
            y_lim=None,
            lpc_abs=True,
            sharey=False,
            model_label_map=model_label_map,

            csi_zoom=True,
            csi_margin=5e-4,
            csi_pad_ratio=0.05,
            csi_min_range=0.003,
            csi_n_ticks=5,           # ✅ 이게 핵심 (너무 촘촘 방지)
            csi_tick_decimals=3,
        )


        logger.info(f"[Eval-3D] Saved metric grid by lead time → {save_path}")
        
        print_metric_1day_lead(metrics_by_model)
        logger.info("Metrics for 1-day lead time")




        #===========Case Study 시각화================================================
        # Case study 실행 (raw_idx -> case_idx -> 예측 -> ZoomGrid 저장)
        #==========================================================================
        import os

        raw_idx = 6301

        loss_specs = [
            {
                "name": "MSE",
                "cfg": os.path.join(PROJECT_ROOT, "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml"),
                "ckpt": os.path.join(PROJECT_ROOT, "experiments/subin_an/conv3d_mse_r90p_r95p/checkpoints/best_val_lpips.pt"),
            },
            {
                "name": "FACL",
                "cfg": os.path.join(PROJECT_ROOT, "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml"),
                "ckpt": os.path.join(PROJECT_ROOT, "experiments/subin_an/conv3d_facl_r90p_r95p/checkpoints/best_val_lpips.pt"),
            },
            {
                "name": "FWCL Lv3",
                "cfg": os.path.join(PROJECT_ROOT, "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml"),
                "ckpt": os.path.join(PROJECT_ROOT, "experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/checkpoints/best_val_lpips.pt"),
            },
        ]

        row_titles = ("Ground Truth", "MSE", "FACL", "WFCL")

        zoom_save, case_idx = run_case_study_zoomgrid_by_raw_index(
            test_loader=test_loader,
            raw_idx=raw_idx,
            loss_specs=loss_specs,
            device=device,
            vis_dir=vis_dir,
            exp=exp,
            base_time=base_time,
            epoch=epoch,
            extent_zoom=(100, 125, 25, 40),  
            mu_sig_unit="m",
            t_indices=(0, 1, 2),
            lead_titles=("May 30, 2020 (1-day)", "May 31, 2020 (2-day)", "June 1, 2020 (3-day)"),
            row_titles=row_titles,
            mu_tp=0.002479317132383585,
            sig_tp=0.005952117739409208,
            clamp_zero=True,
            vmin_percentile=0.0,
            vmax_percentile=99.5,
            vmax_floor=55.0,    # 최솟값 보정
            central_longitude=0.0,
            add_gridlines=True,
            palette="precip_jma",
            figsize=(9, 8),
            dpi=300,
            debug=True,
        )

        logger.info(f"[CaseStudy] Saved ZoomGrid raw_idx={raw_idx} (dataset_idx={case_idx}) → {zoom_save}")







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
            figsize=(15.5, 5.4),           # ✅ 새 코드에 맞게 수정 (맵 크게)
            dpi=300,
            save_path=save_path,
            bar_width=0.42,                # ✅ 코드 내부 기본과 동일하게 조정
            add_gridlines=False            # ✅ 논문용 / 발표용엔 이게 깔끔함
        )

        # GT의 2D FFT 시각화 저장
        fft_png = save_gt_fft2_plane(
            gt_tensor=gt_tensor,
            vis_dir=vis_dir,            # 자동 파일명   
            loss_name=loss_name,        # 파일명/제목 반영
            base_time=base_time,        # 파일명/제목 반영
            b=0, c=0, t=0,
            unit="cyc", dx=1.0, dy=1.0,
            amp="logmag",
            cmap="viridis",
            add_guides=False,           # ✔ 축 내부 가이드선 끔(여백 최소화에 도움)
            title="",                   # ✔ 상단 여백 거의 제거
            dpi=300,
            low_pct=1.0, high_pct=99.0,
            epoch=epoch                 # kwargs로 무시됨(오류 방지용)
        )

        logger.info(f"[Eval-3D] Saved GT 2D FFT plane → {fft_png}")

        logger.info(f"[Eval-3D] Saved LPIPS bands+bar → {save_path}")
        # 필요하면 반환값 사용:
        # print("chosen orientations:", out["orients"])
        # print("bar LPIPS:", out["bar_vals"])

        
    
        import numpy as np
        import matplotlib.pyplot as plt

        os.makedirs(vis_dir, exist_ok=True)

        # --- 1) 프로파일 계산 호출 ---
        result = get_amp_profiles_from_ckpts(
            cfg_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
            ckpt_mse_path  ="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_mse_r90p_r95p/checkpoints/best_val_lpips.pt",
            ckpt_facl_path ="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_r90p_r95p/checkpoints/best_val_lpips.pt",
            ckpt_fwacl_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_facl_wacl_lv3_r90p_r95p/checkpoints/best_val_lpips.pt",
            sample_idx=0, ch_idx=0, t_idx=0,
            nbins=128, amp_mode="db",
            unit="cyc", dx=1.0, dy=1.0,
            make_plot=False,  # ← 함수 내부에서 그리지 않도록 (중복 저장 방지)
        )

        # --- 2) 결과 unpack (안전가드) ---
        if not result:
            raise RuntimeError("[get_amp_profiles_from_ckpts] returned None/empty result")

        if "error" in result:
            raise RuntimeError(f"[get_amp_profiles_from_ckpts] error: {result['error']}")

        profiles = result.get("profiles", {})
        if not profiles:
            raise RuntimeError("[get_amp_profiles_from_ckpts] missing 'profiles' in result")

        k = np.asarray(profiles.get("k", []))
        if k.size == 0:
            raise RuntimeError("[get_amp_profiles_from_ckpts] 'k' is empty")

        series = []
        if "mse" in profiles and len(profiles["mse"]) == k.size:
            series.append(("MSE", np.asarray(profiles["mse"]), {"lw": 2.2, "alpha": 0.95}))
        if "facl" in profiles and len(profiles["facl"]) == k.size:
            series.append(("FACL", np.asarray(profiles["facl"]), {"lw": 2.2, "alpha": 0.95}))
        if "fwacl" in profiles and len(profiles["fwacl"]) == k.size:
            series.append(("WFCL", np.asarray(profiles["fwacl"]), {"lw": 2.2, "alpha": 0.95}))

        if not series:
            avail = {k_: (len(v_) if hasattr(v_, "__len__") else None)
                    for k_, v_ in profiles.items() if k_ != "k"}
            raise RuntimeError(f"[get_amp_profiles_from_ckpts] no valid series to plot; k={k.size}, avail_lengths={avail}")

        # --- 3) 플로팅 ---
        plt.figure(figsize=(8.8, 3))
        for label, y, style in series:
            plt.plot(k, y, label=label, **style)

        # y=0 기준선 (검정 점선)
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.9)

        plt.grid(True, alpha=0.3)
        plt.xlabel("Radial frequency k (cycles/unit)")
        plt.ylabel("Amplitude ratio [dB]")

        # 제목: 왼쪽 정렬 & 논문용 문구
        plt.title("(e) Radial Frequency Profiles of Amplitude Ratio", loc="left")

        plt.legend(loc="best")

        save_path = os.path.join(vis_dir, "amp_ratio_three_models.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved → {save_path}")




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


import torch
from collections import defaultdict

@torch.no_grad()
def evaluate_epoch(
    model, loader, metric_fns, device, metric_params=None,
    lead_idx=None,          # 0/1/2: t+1/t+2/t+3 metric용 선택
    keep_pred_shape=True,   # True면 preds/targets는 원형 유지
):
    """
    - metric 계산: lead_idx가 있으면 해당 lead만 사용
    - pred가 (B*T,C,H,W)로 flatten된 경우를 감지해서 (B,T,C,H,W)로 복구 후 lead 선택
    - 시각화 호환: preds/targets는 keep_pred_shape=True면 원형 그대로 반환
    """
    model.eval()
    metrics = defaultdict(float)
    count = 0
    preds, targets, datetimes = [], [], []

    def _select_lead_5d(t: torch.Tensor, idx: int) -> torch.Tensor:
        """5D에서 lead/time축 하나만 선택해서 T=1로 만든다."""
        if t.ndim != 5:
            return t
        idx = int(idx)

        # (B, C, T, H, W)
        if t.shape[2] <= 10:
            T = t.shape[2]
            j = max(0, min(idx, T - 1))
            return t[:, :, j:j+1, :, :]

        # (B, T, C, H, W)
        if t.shape[1] <= 10:
            T = t.shape[1]
            j = max(0, min(idx, T - 1))
            return t[:, j:j+1, :, :, :]

        return t

    def _maybe_unflatten_pred(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        ✅ 핵심: pred가 (B*T,C,H,W)이고 y가 (B,C,H,W)면 pred를 (B,T,C,H,W)로 복구.
        lead_idx가 있으면 해당 lead만 (B,C,H,W)로 선택.
        """
        if pred.ndim == 4 and y.ndim == 4:
            By = y.shape[0]
            Bp = pred.shape[0]
            if Bp != By and By > 0 and (Bp % By == 0):
                T = Bp // By  # 예: 48/16 = 3
                # pred는 (B*T,C,H,W)라고 가정 -> (B,T,C,H,W)
                pred_btchw = pred.reshape(By, T, pred.shape[1], pred.shape[2], pred.shape[3])

                if lead_idx is None:
                    # 기본은 0번째 lead (t+1)로 맞추는 게 가장 안전
                    return pred_btchw[:, 0, :, :, :]
                else:
                    li = int(lead_idx)
                    if li < 0 or li >= T:
                        raise RuntimeError(f"[Lead index out of range] lead_idx={li}, but T={T}")
                    return pred_btchw[:, li, :, :, :]
        return pred

    def _to_bchw(t: torch.Tensor) -> torch.Tensor:
        """
        metric용 입력을 (N,C,H,W)로 통일
        - (B,C,H,W) -> 그대로
        - (B,C,1,H,W) -> (B,C,H,W)
        - (B,1,C,H,W) -> (B,C,H,W)
        - (B,C,T,H,W) -> (B*T,C,H,W) (T>1이면 lead 선택 권장)
        """
        if t.ndim == 4:
            return t
        if t.ndim != 5:
            raise ValueError(f"Expected 4D/5D, got {t.ndim}D {tuple(t.shape)}")

        # (B,C,T,H,W)
        if t.shape[2] <= 10:
            B, C, T, H, W = t.shape
            if T == 1:
                return t[:, :, 0, :, :]
            return t.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # (B,T,C,H,W)
        if t.shape[1] <= 10:
            B, T, C, H, W = t.shape
            if T == 1:
                return t[:, 0, :, :, :]
            return t.reshape(B * T, C, H, W)

        raise ValueError(f"Ambiguous 5D layout: {tuple(t.shape)}")

    for batch in loader:
        x1 = batch["X1"].to(device, non_blocking=True).float()
        x2 = batch["X2"].to(device, non_blocking=True).float()
        y_all = batch["Y"].to(device, non_blocking=True).float()
        dt = batch.get("time", None)

        pred = model(x1, x2)

        # ✅ (0) pred가 flatten된 4D면, y 기준으로 복구 + lead 선택(가능하면)
        pred = _maybe_unflatten_pred(pred, y_all)

        # ✅ (1) 마지막 배치 등 방어: B가 다르면 둘 다 같은 B로 자르기
        if y_all.shape[0] != pred.shape[0]:
            B = min(y_all.shape[0], pred.shape[0])
            y_all = y_all[:B]
            pred  = pred[:B]

        # ✅ (2) 시각화용 저장(원형 유지)
        if keep_pred_shape:
            preds.append(pred.detach().cpu())
            targets.append(y_all.detach().cpu())
        else:
            preds.append(pred.detach().cpu())
            targets.append(y_all.detach().cpu())

        # ✅ (3) metric 계산용: 5D면 lead 선택 후 BCHW 변환
        pred_m = pred
        y_m    = y_all
        if lead_idx is not None:
            pred_m = _select_lead_5d(pred_m, lead_idx)
            y_m    = _select_lead_5d(y_m, lead_idx)

        pred_bchw = _to_bchw(pred_m)
        y_bchw    = _to_bchw(y_m)

        # ✅ (4) metric 입력 shape 체크
        if pred_bchw.shape != y_bchw.shape:
            raise RuntimeError(
                f"[Metric shape mismatch] pred={tuple(pred_bchw.shape)} y={tuple(y_bchw.shape)} "
                f"(lead_idx={lead_idx})"
            )

        for name, fn in metric_fns.items():
            try:
                val = fn(pred_bchw, y_bchw, **(metric_params or {}))
            except TypeError:
                val = fn(pred_bchw, y_bchw)
            val = float(val.item()) if hasattr(val, "item") else float(val)
            if not torch.isfinite(torch.tensor(val)):
                val = 0.0
            metrics[name] += val

        if dt is not None:
            if isinstance(dt, (list, tuple)):
                datetimes.extend([str(d).split("T")[0] for d in dt])
            else:
                datetimes.append(str(dt).split("T")[0])

        count += 1

    avg_metrics = {k: v / max(1, count) for k, v in metrics.items()}
    return avg_metrics, preds, targets, datetimes




@torch.no_grad()
def evaluate_epoch_by_lead(
    model,
    loader,
    metric_fns,
    device,
    lead_day: int,        # 1, 2, 3
    metric_params=None,
):
    """
    lead_day: 1 -> t+1, 2 -> t+2, 3 -> t+3
    """
    model.eval()
    metrics = defaultdict(float)
    count = 0

    for batch in loader:
        # 🔥 핵심: batch에서 lead-time에 맞는 target을 선택
        x1 = batch["X1"].to(device).float()
        x2 = batch["X2"].to(device).float()

        # 예시 1: Y가 (B, T, H, W)일 경우
        y_all = batch["Y"].to(device).float()
        if y_all.ndim >= 5:        # (B, T, C, H, W)
            y = y_all[:, lead_day - 1]
        elif y_all.ndim == 4:     # (B, T, H, W)
            y = y_all[:, lead_day - 1]
        else:
            y = y_all             # fallback

        pred = model(x1, x2)      # 항상 t+1 예측

        for name, fn in metric_fns.items():
            try:
                val = fn(pred, y, **(metric_params or {}))
            except TypeError:
                val = fn(pred, y)
            val = float(val.item()) if hasattr(val, "item") else float(val)
            metrics[name] += val

        count += 1

    return {k: v / max(1, count) for k, v in metrics.items()}


import os
import csv
from pathlib import Path

def evaluate_leads_and_save_csv(
    model,
    loader,
    metric_fns,
    device,
    ckpt_path,
    csv_dir,
    metric_params=None,
):
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    # checkpoint load
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    rows = []
    for lead_idx, lead_day in zip([0, 1, 2], [1, 2, 3]):
        metrics = evaluate_epoch_by_lead(
            model=model,
            loader=loader,
            metric_fns=metric_fns,
            device=device,
            lead_idx=lead_idx,
            metric_params=metric_params,
        )
        row = {"lead_time": lead_day}
        row.update(metrics)
        rows.append(row)

        print(f"[t+{lead_day}] " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

    out_csv = os.path.join(csv_dir, "metrics_by_leadtime_conv3d_mse_r90p_r95p.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Saved CSV → {out_csv}")
    return out_csv



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

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker


# ============================================================
# 1) RHD (patch-wise KL(hist_true || hist_pred)) implementation
#    - 당신이 준 코드 기반 (완전 벡터화, torch.bincount)
# ============================================================
def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    입력을 (B,1,H,W)로 맞춤.
    허용 shape:
      - (H,W)
      - (B,H,W)
      - (B,1,H,W) or (B,C,H,W)  (C>1이면 첫 채널만 사용)
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.dim() == 2:  # (H,W)
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:  # (B,H,W)
        return x.unsqueeze(1)
    if x.dim() == 4:  # (B,C,H,W)
        if x.size(1) != 1:
            x = x[:, :1]
        return x
    raise ValueError(f"_to_bchw: unsupported shape {tuple(x.shape)}")


@torch.no_grad()
def rhd(
    y_pred, y_true,
    patch_size=15, num_bins=50, epsilon=1e-6,
    robust=True, q_low=0.01, q_high=0.99, quant_sample=200_000
):
    """
    패치별 히스토그램 기반 KL(hist_true || hist_pred) 평균.
    완전 벡터화: 루프 없음, torch.bincount 사용.
    (값이 작을수록 좋음)
    """
    y_true_bchw = _to_bchw(y_true)  # (B,1,H,W)
    y_pred_bchw = _to_bchw(y_pred)  # (B,1,H,W)

    B, _, H, W = y_true_bchw.shape
    device = y_true_bchw.device
    P = patch_size * patch_size

    # (B, P, Np) -> (Np_tot, P)
    t_unf = F.unfold(y_true_bchw, kernel_size=patch_size, stride=patch_size) \
              .transpose(1, 2).reshape(-1, P)
    p_unf = F.unfold(y_pred_bchw, kernel_size=patch_size, stride=patch_size) \
              .transpose(1, 2).reshape(-1, P)
    N = t_unf.shape[0]                      # 전체 패치 수
    num_patches_per_img = max(1, N // B)

    # ---- min/max 결정 (빠른 전역 or 샘플링 백분위수) ----
    if robust:
        both = torch.cat([t_unf, p_unf], dim=1).reshape(-1)
        K = min(int(quant_sample), both.numel())
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


# ============================================================
# 2) Visualization: lead-time별 RHD(KL) 계산/표시 (CPBD 제거 버전)
# ============================================================
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker


import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

'''
@torch.no_grad()
def visualize_prediction_map(
    pred, true, *,
    base_time: str = "", title_suffix: str = "", save_path: str = None,
    vmin_percentile: float = 1.0, vmax_percentile: float = 99.0,
    lon=None, lat=None, central_longitude: float = 180.0,
    add_gridlines: bool = True, dpi: int = 300,
    palette: str = "white_blue_yellow_red",
    tick_labelsize: int = 14.5,
    title_fontsize: float = 12.5,
    suptitle_fontsize: float = 8.0,

    # z-score → mm/day
    mu_tp: float  = 0.002479317132383585,   # [m]
    sig_tp: float = 0.005952117739409208,   # [m]
    clamp_zero: bool = True,

    # ---- RHD(KL-hist) 옵션 ----
    rhd_psz: int = 15,
    rhd_bins: int = 50,
    rhd_epsilon: float = 1e-6,
    rhd_robust: bool = True,
    rhd_q_low: float = 0.01,
    rhd_q_high: float = 0.99,
    rhd_quant_sample: int = 200_000,

    # ---- 표기/모델명 ----
    model_name: str = "FACL",
    show_rhd_on_gt: bool = False,
    rhd_decimals: int = 3,
    rhd_device: str = "cpu",

    # ---- 레이아웃(겹침 방지) ----
    fig_width: float = 13.2,
    map_height_per_t: float = 3.2,     # 지도 높이(리드타임 1개당)
    cbar_height: float = 0.47,         # 컬러바 행 높이(리드타임 1개당)
    hspace: float = 0.80,              # (지도-컬러바-다음지도) 간격
    wspace: float = 0.05,              # GT-Pred 간격
    cbar_pad_ratio: float = 0.25,      # 컬러바 행에서 위쪽 패딩 비율(0~1)
):
    """
    (T x 2) 지도 시각화 (겹침 방지 GridSpec 버전):
      - 각 t마다:
          지도행:  [GT | Pred]
          컬러바행: [공용 컬러바 1개(두 패널 span)]
    Returns:
      dict: {"rhd": [t별 값]}
    """
    # ------------------------------------------------------------
    # 0) pred/true -> (T,H,W)
    # ------------------------------------------------------------
    if pred.dim() == 5:          # (B,C,T,H,W)
        pred_3d = pred[0, 0]
        true_3d = true[0, 0]
    elif pred.dim() == 4:        # (B,T,H,W)
        pred_3d = pred[0]
        true_3d = true[0]
    else:
        raise ValueError(f"pred/true must be 4D or 5D, got {tuple(pred.shape)}")

    if pred_3d.dim() != 3 or true_3d.dim() != 3:
        raise ValueError(f"Expected (T,H,W), got pred={tuple(pred_3d.shape)}, true={tuple(true_3d.shape)}")

    T, H, W = true_3d.shape

    # ------------------------------------------------------------
    # 1) 좌표/extent
    # ------------------------------------------------------------
    if lon is None:
        lon = np.linspace(0.0, 360.0, W, endpoint=False)
    if lat is None:
        lat = np.linspace(-90.0, 90.0, H)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    # ------------------------------------------------------------
    # 2) z-score -> mm/day
    # ------------------------------------------------------------
    def _denorm_to_mm(z_np, mu, sig):
        x_m = z_np * sig + mu
        if clamp_zero:
            x_m = np.maximum(x_m, 0.0)
        return x_m * 1000.0

    true_z = true_3d.detach().cpu().float().numpy()
    pred_z = pred_3d.detach().cpu().float().numpy()
    true_mm = _denorm_to_mm(true_z, mu_tp, sig_tp)
    pred_mm = _denorm_to_mm(pred_z, mu_tp, sig_tp)

    tv = true_mm[np.isfinite(true_mm)]
    if tv.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(tv, vmin_percentile))
        vmax = float(np.percentile(tv, vmax_percentile))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = float(np.nanmax(tv)) or 1.0
        if vmin >= vmax: vmin, vmax = 0.0, 1.0
    vmin_plot = np.nextafter(vmin, vmax)

    # ------------------------------------------------------------
    # 3) 스타일/컬러맵
    # ------------------------------------------------------------
    plt.style.use("default")
    mpl.rcParams.update({
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    if palette == "white_blue_yellow_red":
        colors = ["#ffffff", "#4575b4", "#ffffbf", "#d73027"]
        cmap_obj = LinearSegmentedColormap.from_list("white_blue_yellow_red", colors, N=256)
    else:
        cmap_obj = plt.get_cmap(palette).copy()
    cmap_obj.set_bad("white", 1.0)
    cmap_obj.set_under("white", 1.0)

    # ------------------------------------------------------------
    # 4) GridSpec: 각 t마다 2행(지도/컬러바)
    # ------------------------------------------------------------
    proj_axes = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs  = ccrs.PlateCarree()

    fig_h = T * (map_height_per_t + cbar_height)
    fig = plt.figure(figsize=(fig_width, fig_h), facecolor="white", dpi=dpi)

    # rows = 2*T (map row + cbar row 반복)
    height_ratios = []
    for _ in range(T):
        height_ratios += [1.15, (cbar_height / map_height_per_t)*0.85]  # 비율로 환산

    gs = fig.add_gridspec(
        nrows=2*T, ncols=2,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace
    )

    if base_time or title_suffix:
        fig.suptitle(f"t₀={base_time}  {title_suffix}", fontsize=suptitle_fontsize, y=0.995)

    def _decorate(ax):
        ax.set_global()
        if add_gridlines:
            lon_ticks = np.arange(-180, 181, 60)
            lat_ticks = np.arange(-90, 91, 30)
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
            ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(), draw_labels=False,
                linewidth=0.4, linestyle="dashed",
                color="gray", alpha=0.55, zorder=6
            )
            gl.xlocator = mticker.FixedLocator(lon_ticks)
            gl.ylocator = mticker.FixedLocator(lat_ticks)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        ax.coastlines(resolution="110m", linewidth=0.5, color="dimgray", alpha=0.6, zorder=10)

    CB_LABEL_FONTSIZE = 11
    CB_TICK_FONTSIZE  = 12

    # ------------------------------------------------------------
    # 5) 리드타임별 RHD + 그림 + 컬러바(겹침 없음)
    # ------------------------------------------------------------
    rhd_list = []

    dev = pred_3d.device if rhd_device == "pred" else torch.device("cpu")

    for t in range(T):
        # RHD 계산(z-score)
        gt_t = true_3d[t].to(dev).float()
        pr_t = pred_3d[t].to(dev).float()

        rhd_val = rhd(
            y_pred=pr_t, y_true=gt_t,
            patch_size=rhd_psz, num_bins=rhd_bins, epsilon=rhd_epsilon,
            robust=rhd_robust, q_low=rhd_q_low, q_high=rhd_q_high,
            quant_sample=rhd_quant_sample
        )
        rhd_list.append(rhd_val)
        rhd_txt = f"{rhd_val:.{rhd_decimals}f}"

        # mm/day (시각화)
        gt_t_mm   = np.ma.masked_invalid(true_mm[t])
        pred_t_mm = np.ma.masked_invalid(pred_mm[t])

        # ----- 지도 axes -----
        ax_l = fig.add_subplot(gs[2*t, 0], projection=proj_axes)
        ax_r = fig.add_subplot(gs[2*t, 1], projection=proj_axes)

        im_l = ax_l.imshow(
            gt_t_mm, origin="lower", extent=extent, transform=data_crs,
            cmap=cmap_obj, vmin=vmin_plot, vmax=vmax, interpolation="nearest"
        )
        if show_rhd_on_gt:
            ax_l.set_title(f"RHD={rhd_txt}", fontsize=title_fontsize, loc="left", pad=3)
        else:
            ax_l.set_title(" ", fontsize=title_fontsize, loc="left", pad=3)
        _decorate(ax_l)

        im_r = ax_r.imshow(
            pred_t_mm, origin="lower", extent=extent, transform=data_crs,
            cmap=cmap_obj, vmin=vmin_plot, vmax=vmax, interpolation="nearest"
        )
        ax_r.set_title(f"RHD={rhd_txt}", fontsize=title_fontsize, loc="left", pad=3)
        _decorate(ax_r)

        # ----- 컬러바 전용 축: 다음 행(2*t+1)에서 col0~1 span -----
        cax = fig.add_subplot(gs[2*t + 1, :])
        # 축 자체는 안 보이게(컬러바만)
        cax.set_axis_off()

        # 컬러바를 cax 안에 "조금 위쪽"에 배치하고 싶으면 inset_axes 사용
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset = inset_axes(
            cax,
            width="100%", height="95%",
            loc="lower center",
            borderpad=0.30
        )

        cbar = fig.colorbar(im_r, cax=inset, orientation="horizontal")
        cbar.set_label("Precipitation (mm/day)", fontsize=CB_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=CB_TICK_FONTSIZE)
        cbar.outline.set_linewidth(0.8)
        cbar.ax.set_facecolor("white")

    # ------------------------------------------------------------
    # 6) 저장/표시
    # ------------------------------------------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return {"rhd": rhd_list}
'''
@torch.no_grad()
def _lpc_mean_tensor(im,
                     C=2.0,
                     scales=None,
                     norient=8,
                     B=None,
                     weights=None,
                     sigma_on_f=0.33):
    """
    im: (H,W) float tensor
    반환: 전체 평균형 LPC(mean of LPC map).
    - 내부 필터/에너지 임계값/거듭제곱 가중은 lpc_si_tensor와 동일하게 적용
      (일관성을 위해 에너지 T = mean + 2*std 사용)
    - 중앙 크롭 가중(정렬 평균)은 하지 않음: 전역 평균만 계산
    """
    if im.ndim != 2:
        raise ValueError("im should be 2D (H,W)")

    H, W = im.shape
    device = im.device
    cdtype = torch.complex64 if im.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.complex128
    rdtype = torch.float32 if im.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float64

    # --- scales / weights (lpc_si_tensor와 동일 규칙) ---
    if scales is None:
        scales = [1.0, 1.5, 2.0]
        w = torch.tensor([1.0, -3.0, 2.0], dtype=rdtype, device=device)
    else:
        scales = [float(v) for v in scales]
        if weights is None:
            if len(scales) == 3 and all(abs(a-b) < 1e-8 for a,b in zip(scales,[1,2,3])):
                w = torch.tensor([1.0, -4.0, 3.0], dtype=rdtype, device=device)
            elif len(scales) == 3 and all(abs(a-b) < 1e-8 for a,b in zip(scales,[1,2,4])):
                w = torch.tensor([1.0, -3.0, 2.0], dtype=rdtype, device=device)
            elif len(scales) == 3 and all(abs(a-b) < 1e-8 for a,b in zip(scales,[1,1.5,2.0])):
                w = torch.tensor([1.0, -3.0, 2.0], dtype=rdtype, device=device)
            else:
                raise ValueError("Custom scales 사용 시 weights도 명시하세요 (예: [1,-3,2])")
        else:
            w = torch.tensor(list(weights), dtype=rdtype, device=device)
            if len(w) != len(scales):
                raise ValueError("len(weights) must equal len(scales)")

    if B is None:
        B = int(round(min(H, W) / 16.0))

    # --- 필터 뱅크(FFT domain) ---
    filt = loggabor_2d_torch(H, W, norient, scales,
                             sigma_on_f=sigma_on_f, device=device, dtype=rdtype)  # (O,S,H,W)

    X = torch.fft.fft2(im.to(rdtype))

    s_lpc = torch.ones((H, W, norient), dtype=cdtype, device=device)
    energy = torch.zeros((H, W, norient), dtype=rdtype, device=device)

    for o in range(norient):
        M_first = None
        for s in range(len(scales)):
            Hso = filt[o, s]                 # (H,W), real
            M = torch.fft.ifft2(X * Hso)     # complex
            if s == 0:
                M_first = M
            s_lpc[..., o] = s_lpc[..., o] * (M ** w[s])

        # 에너지 게이팅: mean+2*std 컷 (lpc_si_tensor와 동일)
        e = torch.abs(M_first)
        if B > 0 and (2*B < H) and (2*B < W):
            e_center = e[B:H-B, B:W-B]
        else:
            e_center = e
        T = e_center.mean() + 2.0 * e_center.std(unbiased=False)
        e = torch.clamp(e - T, min=0.0)
        energy[..., o] = e

    # 위상 일치도 cos(angle), 음수는 0
    coh = torch.cos(torch.angle(s_lpc))
    coh = torch.where(coh < 0, torch.zeros_like(coh), coh)

    num = (coh * energy).sum(dim=2)
    den = energy.sum(dim=2) + C
    lpc_map = num / den  # (H,W)

    # ★ 전역 평균만 반환 (중앙 정렬/가중 없음)
    return float(lpc_map.mean().item())


@torch.no_grad()
def lpc_mean(pred: torch.Tensor,
             target: torch.Tensor,
             *,
             current_central_longitude: float = 0.0,
             target_central_longitude: float = 180.0) -> float:
    """
    pred: (B,1,H,W) or (B,T,H,W) or (H,W)
    전체 평균형 LPC (pred 기준)을 배치 평균하여 스칼라로 반환.
    - 경도 롤/로그정규화는 lpc_si와 동일하게 적용.
    """
    imgs = _to_bchw(pred)  # (B,1,H,W)

    # 1) 경도 정렬
    imgs = _roll_longitude_to_center(
        imgs, current_center_deg=current_central_longitude,
        target_center_deg=target_central_longitude
    )
    # 2) 로그 정규화 (네 코드의 _log_normalize 사용)
    imgs = _log_normalize(imgs)

    # 3) 샘플별 mean-LPC 산출 후 평균
    scores = []
    for i in range(imgs.shape[0]):
        scores.append(_lpc_mean_tensor(imgs[i, 0]))
    return _nan_guard_scalar(sum(scores) / max(1, len(scores)))
@torch.no_grad()
def visualize_prediction_map(
    pred, true, *,
    base_time: str = "", title_suffix: str = "", save_path: str = None,
    vmin_percentile: float = 1.0, vmax_percentile: float = 99.0,
    lon=None, lat=None, central_longitude: float = 180.0,
    add_gridlines: bool = True, dpi: int = 300,
    palette: str = "white_blue_yellow_red",
    tick_labelsize: int = 14.5,
    title_fontsize: float = 12.5,
    suptitle_fontsize: float = 8.0,

    # z-score → mm/day
    mu_tp: float  = 0.002479317132383585,   # [m]
    sig_tp: float = 0.005952117739409208,   # [m]
    clamp_zero: bool = True,

    # ---- 표기/모델명 ----
    model_name: str = "FACL",

    # ---- 레이아웃 ----
    fig_width: float = 13.2,
    map_height_per_t: float = 3.2,
    cbar_height: float = 0.47,
    hspace: float = 0.80,
    wspace: float = 0.05,

    # ============================================================
    # ✅ FSS 옵션 (네 YAML과 동일한 이름으로 받기)
    # ============================================================
    fss_threshold: float = 1.0,   # mm/day
    fss_window: int = 5,          # window_size
    fss_decimals: int = 3,        # title 표기 자릿수

    # ============================================================
    # ✅ RHD 옵션
    # ============================================================
    metric_mode: str = "fss",     # "fss" 또는 "rhd"
    rhd_psz: int = 15,
    rhd_bins: int = 50,
    rhd_decimals: int = 4,

    # vmax 고정값 (None이면 vmax_percentile 기반 자동 계산)
    vmax_override: float = None,
):
    """
    (T x 2) 지도 시각화:
      - 각 t마다: [GT | Pred] + 아래 공용 컬러바
      - metric_mode="fss": t별 FSS, metric_mode="rhd": t별 RHD 타이틀 표기
    Returns:
      dict: metric_mode에 따라 {"fss":..., ...} 또는 {"rhd":..., ...}
    """
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ------------------------------------------------------------
    # helpers: _to_bchw, nan guard, fss (네가 준 구현 기반)
    # ------------------------------------------------------------
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        """
        Accepts (H,W), (B,H,W), (B,1,H,W), (B,C,H,W) -> returns (B,1,H,W) (첫 채널)
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 2:          # (H,W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:        # (B,H,W)
            x = x.unsqueeze(1)
        elif x.dim() == 4:        # (B,C,H,W)
            if x.size(1) != 1:
                x = x[:, :1]
        else:
            raise ValueError(f"_to_bchw expects 2D/3D/4D, got {tuple(x.shape)}")
        return x

    def _nan_guard_scalar(v: float) -> float:
        return float(v) if np.isfinite(v) else float("nan")

    @torch.no_grad()
    def fss_torch(y_pred, y_true, threshold, window_size=5):
        y_true_bchw = _to_bchw(y_true)
        y_pred_bchw = _to_bchw(y_pred)

        # pairwise valid: NaN이 있으면 둘 다 0으로 만들어서 영향 제거 (masking 방식)
        valid = torch.isfinite(y_true_bchw) & torch.isfinite(y_pred_bchw)
        y_true_bchw = torch.where(valid, y_true_bchw, torch.zeros_like(y_true_bchw))
        y_pred_bchw = torch.where(valid, y_pred_bchw, torch.zeros_like(y_pred_bchw))

        y_true_bin = (y_true_bchw >= threshold).float()
        y_pred_bin = (y_pred_bchw >= threshold).float()

        kernel = torch.ones((1, 1, window_size, window_size), device=y_true_bchw.device, dtype=y_true_bchw.dtype)
        y_true_sum = F.conv2d(y_true_bin, kernel, padding=window_size // 2)
        y_pred_sum = F.conv2d(y_pred_bin, kernel, padding=window_size // 2)

        num_pixels_in_window = window_size * window_size
        y_true_frac = y_true_sum / num_pixels_in_window
        y_pred_frac = y_pred_sum / num_pixels_in_window

        # valid 위치만 평균내고 싶으면 분모/분자 모두 valid로 weight 주는 방식이 더 엄밀하지만,
        # 대부분 전지구 격자에 NaN이 거의 없어서 여기서는 간단히 mean 사용.
        mse_frac = ((y_true_frac - y_pred_frac) ** 2).mean()
        ref_mse = (y_true_frac ** 2).mean() + (y_pred_frac ** 2).mean()
        fss_score = 1.0 - mse_frac / (ref_mse + 1e-8)

        return _nan_guard_scalar(float(fss_score.item()))

    # ------------------------------------------------------------
    # 0) pred/true -> (T,H,W)
    # ------------------------------------------------------------
    if pred.dim() == 5:          # (B,C,T,H,W)
        pred_3d = pred[0, 0]
        true_3d = true[0, 0]
    elif pred.dim() == 4:        # (B,T,H,W)
        pred_3d = pred[0]
        true_3d = true[0]
    else:
        raise ValueError(f"pred/true must be 4D or 5D, got {tuple(pred.shape)}")

    T, H, W = true_3d.shape

    # ------------------------------------------------------------
    # 1) 좌표/extent
    # ------------------------------------------------------------
    if lon is None:
        lon = np.linspace(0.0, 360.0, W, endpoint=False)
    if lat is None:
        lat = np.linspace(-90.0, 90.0, H)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    # ------------------------------------------------------------
    # 2) z-score -> mm/day (numpy)
    # ------------------------------------------------------------
    def _denorm_to_mm(z_np, mu, sig):
        x_m = z_np * sig + mu
        if clamp_zero:
            x_m = np.maximum(x_m, 0.0)
        return x_m * 1000.0  # mm

    true_z = true_3d.detach().cpu().float().numpy()
    pred_z = pred_3d.detach().cpu().float().numpy()
    true_mm = _denorm_to_mm(true_z, mu_tp, sig_tp)
    pred_mm = _denorm_to_mm(pred_z, mu_tp, sig_tp)

    tv = true_mm[np.isfinite(true_mm)]
    if tv.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(tv, vmin_percentile))
        vmax = float(np.percentile(tv, vmax_percentile))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = float(np.nanmax(tv)) or 1.0
        if vmin >= vmax: vmin, vmax = 0.0, 1.0
    if vmax_override is not None:
        vmax = float(vmax_override)
    vmin_plot = np.nextafter(vmin, vmax)

    # ------------------------------------------------------------
    # 3) 스타일/컬러맵
    # ------------------------------------------------------------
    plt.style.use("default")
    mpl.rcParams.update({
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    if palette == "white_blue_yellow_red":
        colors = ["#ffffff", "#4575b4", "#ffffbf", "#d73027"]
        cmap_obj = LinearSegmentedColormap.from_list("white_blue_yellow_red", colors, N=256)
    elif palette == "precip_jma":
        precip_colors = [
            "#ffffff",  # 0 (no rain)
            "#c8f0ff",  # very light blue
            "#96d2fa",  # light blue
            "#64b4f0",  # sky blue
            "#3296e6",  # medium blue
            "#50c850",  # green
            "#a0e632",  # yellow-green
            "#f0f000",  # yellow
            "#ffc800",  # amber
            "#ff9600",  # orange
            "#ff5a00",  # orange-red
            "#e61400",  # red
            "#c80000",  # dark red
            "#960050",  # crimson
            "#7800a0",  # purple
            "#3c0078",  # dark purple
        ]
        cmap_obj = LinearSegmentedColormap.from_list("precip_jma", precip_colors, N=256)
    else:
        cmap_obj = plt.get_cmap(palette).copy()
    cmap_obj.set_bad("white", 1.0)
    cmap_obj.set_under("white", 1.0)

    # ------------------------------------------------------------
    # 4) GridSpec
    # ------------------------------------------------------------
    proj_axes = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs  = ccrs.PlateCarree()

    fig_h = T * (map_height_per_t + cbar_height)
    fig = plt.figure(figsize=(fig_width, fig_h), facecolor="white", dpi=dpi)

    height_ratios = []
    for _ in range(T):
        height_ratios += [1.15, (cbar_height / map_height_per_t)*0.85]

    gs = fig.add_gridspec(
        nrows=2*T, ncols=2,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace
    )

    if base_time or title_suffix:
        fig.suptitle(f"t₀={base_time}  {title_suffix}", fontsize=suptitle_fontsize, y=0.995)

    def _decorate(ax):
        ax.set_global()
        if add_gridlines:
            lon_ticks = np.arange(-180, 181, 60)
            lat_ticks = np.arange(-90, 91, 30)
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
            ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(), draw_labels=False,
                linewidth=0.4, linestyle="dashed",
                color="gray", alpha=0.55, zorder=6
            )
            gl.xlocator = mticker.FixedLocator(lon_ticks)
            gl.ylocator = mticker.FixedLocator(lat_ticks)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        ax.coastlines(resolution="110m", linewidth=0.5, color="dimgray", alpha=0.6, zorder=10)

    CB_LABEL_FONTSIZE = 11
    CB_TICK_FONTSIZE  = 12

    # ------------------------------------------------------------
    # 5) ✅ t별 FSS 또는 RHD + 그림 + 컬러바
    # ------------------------------------------------------------
    metric_list = []

    for t in range(T):
        gt_t_mm   = np.ma.masked_invalid(true_mm[t])
        pred_t_mm = np.ma.masked_invalid(pred_mm[t])

        gt_t_torch   = torch.as_tensor(np.asarray(gt_t_mm), dtype=torch.float32)
        pred_t_torch = torch.as_tensor(np.asarray(pred_t_mm), dtype=torch.float32)

        if metric_mode == "rhd":
            from metrics.metric_utils_3d import rhd as rhd_fn
            metric_val = rhd_fn(pred_t_torch.unsqueeze(0).unsqueeze(0), gt_t_torch.unsqueeze(0).unsqueeze(0), patch_size=rhd_psz, num_bins=rhd_bins)
            metric_title = f"RHD={metric_val:.{rhd_decimals}f}"
        else:
            metric_val = fss_torch(pred_t_torch, gt_t_torch, threshold=float(fss_threshold), window_size=int(fss_window))
            metric_title = f"FSS={float(metric_val):.{fss_decimals}f}"

        metric_list.append(float(metric_val))

        # ----- 지도 axes -----
        ax_l = fig.add_subplot(gs[2*t, 0], projection=proj_axes)
        ax_r = fig.add_subplot(gs[2*t, 1], projection=proj_axes)

        im_l = ax_l.imshow(
            gt_t_mm, origin="lower", extent=extent, transform=data_crs,
            cmap=cmap_obj, vmin=vmin_plot, vmax=vmax, interpolation="nearest"
        )
        ax_l.set_title(" ", fontsize=title_fontsize, loc="left", pad=3)
        _decorate(ax_l)

        im_r = ax_r.imshow(
            pred_t_mm, origin="lower", extent=extent, transform=data_crs,
            cmap=cmap_obj, vmin=vmin_plot, vmax=vmax, interpolation="nearest"
        )
        ax_r.set_title(
            metric_title,
            fontsize=title_fontsize, loc="left", pad=3
        )
        _decorate(ax_r)

        # ----- 컬러바 전용 축 -----
        cax = fig.add_subplot(gs[2*t + 1, :])
        cax.set_axis_off()

        inset = inset_axes(
            cax,
            width="100%", height="95%",
            loc="lower center",
            borderpad=0.30
        )

        cbar = fig.colorbar(im_r, cax=inset, orientation="horizontal")
        cbar.set_label("Precipitation (mm/day)", fontsize=CB_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=CB_TICK_FONTSIZE)
        cbar.outline.set_linewidth(0.8)
        cbar.ax.set_facecolor("white")

    # ------------------------------------------------------------
    # 6) 저장/표시
    # ------------------------------------------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    if metric_mode == "rhd":
        return {"rhd": metric_list, "rhd_psz": rhd_psz, "rhd_bins": rhd_bins}
    return {
        "fss": metric_list,
        "fss_threshold": float(fss_threshold),
        "fss_window": int(fss_window),
    }






def visualize_multimodel_single_lead(
    true,
    preds_list,          # list of (pred_tensor, label_str)
    *,
    t_idx: int = 0,
    base_time: str = "", save_path: str = None,
    vmin_percentile: float = 1.0, vmax_percentile: float = 99.0,
    vmax_override: float = None,
    lon=None, lat=None, central_longitude: float = 180.0,
    add_gridlines: bool = True, dpi: int = 300,
    palette: str = "precip_jma",
    tick_labelsize: int = 11,
    title_fontsize: float = 12.5,
    mu_tp: float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    clamp_zero: bool = True,
    fig_width_per_col: float = 7.5,
    map_height: float = 3.5,
    cbar_height: float = 0.47,
    hspace: float = 0.15,
    wspace: float = 0.25,
    rhd_psz: int = 15,
    rhd_bins: int = 50,
    rhd_decimals: int = 3,
):
    """단일 리드타임(t_idx)에 대한 1×(1+N) 레이아웃: GT + N개 모델. 하단 컬러바 하나."""
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from metrics.metric_utils_3d import rhd as rhd_fn

    def _to_3d(t):
        if t.dim() == 5:   return t[0, 0]
        elif t.dim() == 4: return t[0]
        raise ValueError(f"Expected 4D/5D, got {tuple(t.shape)}")

    true_3d = _to_3d(true)
    T, H, W = true_3d.shape
    ti = int(np.clip(t_idx, 0, T - 1))

    if lon is None: lon = np.linspace(0.0, 360.0, W, endpoint=False)
    if lat is None: lat = np.linspace(-90.0, 90.0, H)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    def _denorm(z_np):
        x = z_np * sig_tp + mu_tp
        if clamp_zero: x = np.maximum(x, 0.0)
        return x * 1000.0

    true_mm_all = _denorm(true_3d.detach().cpu().float().numpy())
    true_mm = np.ma.masked_invalid(true_mm_all[ti])

    tv = true_mm_all[np.isfinite(true_mm_all)]
    vmin = float(np.percentile(tv, vmin_percentile)) if tv.size else 0.0
    vmax = float(np.percentile(tv, vmax_percentile)) if tv.size else 1.0
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = float(np.nanmax(tv)) or 1.0
    if vmin >= vmax: vmin, vmax = 0.0, 1.0
    if vmax_override is not None: vmax = float(vmax_override)
    vmin_plot = np.nextafter(vmin, vmax)

    if palette == "precip_jma":
        _pc = ["#ffffff","#c8f0ff","#96d2fa","#64b4f0","#3296e6",
               "#50c850","#a0e632","#f0f000","#ffc800","#ff9600",
               "#ff5a00","#e61400","#c80000","#960050","#7800a0","#3c0078"]
        cmap_obj = LinearSegmentedColormap.from_list("precip_jma", _pc, N=256)
    elif palette == "white_blue_yellow_red":
        cmap_obj = LinearSegmentedColormap.from_list(
            "wbyr", ["#ffffff","#4575b4","#ffffbf","#d73027"], N=256)
    else:
        cmap_obj = plt.get_cmap(palette).copy()
    cmap_obj.set_bad("white", 1.0); cmap_obj.set_under("white", 1.0)

    preds_mm = []
    for pred_t, lbl in preds_list:
        p3d = _to_3d(pred_t)
        preds_mm.append((_denorm(p3d.detach().cpu().float().numpy()), lbl))

    N = len(preds_mm)
    ncols = 1 + N
    col_labels = ["Ground Truth"] + [lbl for _, lbl in preds_mm]
    day_label = f"{ti + 1}-day"

    mpl.rcParams.update({"savefig.facecolor": "white", "figure.facecolor": "white"})
    proj_axes = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs  = ccrs.PlateCarree()

    cbar_ratio = cbar_height / map_height
    fig_w = fig_width_per_col * ncols
    fig_h = map_height + cbar_height + 0.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=dpi)

    gs = fig.add_gridspec(nrows=2, ncols=ncols,
                          height_ratios=[1.0, cbar_ratio],
                          hspace=hspace, wspace=wspace)

    if base_time:
        fig.suptitle(f"t₀={base_time}  ({day_label})", fontsize=10, y=0.998)

    def _decorate(ax, show_xlabels=True, show_ylabels=True):
        ax.set_global()
        if add_gridlines:
            lon_t = np.arange(-180, 181, 60); lat_t = np.arange(-90, 91, 30)
            ax.set_xticks(lon_t, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_t, crs=ccrs.PlateCarree())
            if show_xlabels:
                ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
            else:
                ax.xaxis.set_major_formatter(mticker.NullFormatter())
            if show_ylabels:
                ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
            else:
                ax.yaxis.set_major_formatter(mticker.NullFormatter())
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.4, linestyle="dashed", color="gray", alpha=0.55, zorder=6)
            gl.xlocator = mticker.FixedLocator(lon_t)
            gl.ylocator = mticker.FixedLocator(lat_t)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        ax.coastlines(resolution="110m", linewidth=0.5, color="dimgray", alpha=0.6, zorder=10)

    gt_torch = torch.as_tensor(np.asarray(true_mm), dtype=torch.float32)
    im_ref = None

    for col_idx in range(ncols):
        ax = fig.add_subplot(gs[0, col_idx], projection=proj_axes)
        if col_idx == 0:
            data_2d = true_mm
        else:
            data_2d = np.ma.masked_invalid(preds_mm[col_idx - 1][0][ti])

        im = ax.imshow(data_2d, origin="lower", extent=extent, transform=data_crs,
                       cmap=cmap_obj, vmin=vmin_plot, vmax=vmax, interpolation="nearest")
        if im_ref is None:
            im_ref = im
        _decorate(ax, show_xlabels=True, show_ylabels=True)

        if col_idx > 0:
            pred_mm_t  = data_2d
            pred_torch = torch.as_tensor(np.asarray(pred_mm_t), dtype=torch.float32)
            rhd_val    = rhd_fn(pred_torch.unsqueeze(0).unsqueeze(0),
                                gt_torch.unsqueeze(0).unsqueeze(0),
                                patch_size=rhd_psz, num_bins=rhd_bins)
            ax.set_title(f"RHD={rhd_val:.{rhd_decimals}f}",
                         fontsize=title_fontsize, loc="left", pad=3)

    # Single colorbar at bottom
    cax_row = fig.add_subplot(gs[1, :])
    cax_row.set_axis_off()
    inset = inset_axes(cax_row, width="60%", height="45%",
                       loc="lower center", borderpad=0.3)
    cbar = fig.colorbar(im_ref, cax=inset, orientation="horizontal")
    cbar.set_label("Precipitation (mm/day)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(0.8)
    cbar.locator = mticker.FixedLocator([5, 15, 25, 35, 45])
    cbar.update_ticks()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_prediction_map_multimodel(
    true,
    preds_list,          # list of (pred_tensor, label_str)
    *,
    base_time: str = "", title_suffix: str = "", save_path: str = None,
    vmin_percentile: float = 1.0, vmax_percentile: float = 99.0,
    vmax_override: float = None,
    lon=None, lat=None, central_longitude: float = 180.0,
    add_gridlines: bool = True, dpi: int = 300,
    palette: str = "precip_jma",
    tick_labelsize: int = 11,
    title_fontsize: float = 12.5,
    suptitle_fontsize: float = 8.0,
    mu_tp: float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    clamp_zero: bool = True,
    fig_width_per_col: float = 6.6,
    map_height_per_t: float = 3.2,
    cbar_height: float = 0.47,
    hspace: float = 0.12,
    wspace: float = 0.03,
    rhd_psz: int = 15,
    rhd_bins: int = 50,
    rhd_decimals: int = 3,
):
    """(1+N)×T 레이아웃: 행=모델(GT+N개), 열=리드타임. 하단에 컬러바 하나."""
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from metrics.metric_utils_3d import rhd as rhd_fn

    # -- true -> (T,H,W) --
    if true.dim() == 5:   true_3d = true[0, 0]
    elif true.dim() == 4: true_3d = true[0]
    else: raise ValueError(f"true must be 4D/5D, got {tuple(true.shape)}")
    T, H, W = true_3d.shape

    if lon is None: lon = np.linspace(0.0, 360.0, W, endpoint=False)
    if lat is None: lat = np.linspace(-90.0, 90.0, H)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    def _denorm(z_np):
        x = z_np * sig_tp + mu_tp
        if clamp_zero: x = np.maximum(x, 0.0)
        return x * 1000.0

    true_mm = _denorm(true_3d.detach().cpu().float().numpy())

    # vmin/vmax from GT
    tv = true_mm[np.isfinite(true_mm)]
    vmin = float(np.percentile(tv, vmin_percentile)) if tv.size else 0.0
    vmax = float(np.percentile(tv, vmax_percentile)) if tv.size else 1.0
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = float(np.nanmax(tv)) or 1.0
    if vmin >= vmax: vmin, vmax = 0.0, 1.0
    if vmax_override is not None: vmax = float(vmax_override)
    vmin_plot = np.nextafter(vmin, vmax)

    # colormap
    if palette == "precip_jma":
        _pc = ["#ffffff","#c8f0ff","#96d2fa","#64b4f0","#3296e6",
               "#50c850","#a0e632","#f0f000","#ffc800","#ff9600",
               "#ff5a00","#e61400","#c80000","#960050","#7800a0","#3c0078"]
        cmap_obj = LinearSegmentedColormap.from_list("precip_jma", _pc, N=256)
    elif palette == "white_blue_yellow_red":
        cmap_obj = LinearSegmentedColormap.from_list(
            "wbyr", ["#ffffff","#4575b4","#ffffbf","#d73027"], N=256)
    else:
        cmap_obj = plt.get_cmap(palette).copy()
    cmap_obj.set_bad("white", 1.0); cmap_obj.set_under("white", 1.0)

    # preds -> list of (mm_array (T,H,W), label)
    preds_mm = []
    for pred_t, lbl in preds_list:
        if pred_t.dim() == 5:   p3d = pred_t[0, 0]
        elif pred_t.dim() == 4: p3d = pred_t[0]
        else: raise ValueError
        preds_mm.append((_denorm(p3d.detach().cpu().float().numpy()), lbl))

    N = len(preds_mm)
    nrows_data = 1 + N   # GT row + model rows
    ncols_data = T       # lead time columns

    # row/column labels
    row_labels = ["Ground Truth"] + [lbl for _, lbl in preds_mm]
    col_labels = [f"{t+1}-day" for t in range(T)]

    # GridSpec: nrows_data map rows + 1 colorbar row
    mpl.rcParams.update({"savefig.facecolor": "white", "figure.facecolor": "white"})
    proj_axes = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs  = ccrs.PlateCarree()

    cbar_ratio = cbar_height / map_height_per_t
    height_ratios = [1.0] * nrows_data + [cbar_ratio]

    fig_w = fig_width_per_col * ncols_data
    fig_h = map_height_per_t * nrows_data + cbar_height + 0.4
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=dpi)

    gs = fig.add_gridspec(nrows=nrows_data + 1, ncols=ncols_data,
                          height_ratios=height_ratios,
                          hspace=hspace, wspace=wspace)

    if base_time or title_suffix:
        fig.suptitle(f"t₀={base_time}  {title_suffix}", fontsize=suptitle_fontsize, y=0.998)

    def _decorate(ax, show_xlabels=False, show_ylabels=False):
        ax.set_global()
        if add_gridlines:
            lon_t = np.arange(-180, 181, 60); lat_t = np.arange(-90, 91, 30)
            ax.set_xticks(lon_t, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_t, crs=ccrs.PlateCarree())
            if show_xlabels:
                ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
            else:
                ax.xaxis.set_major_formatter(mticker.NullFormatter())
            if show_ylabels:
                ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))
            else:
                ax.yaxis.set_major_formatter(mticker.NullFormatter())
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                              linewidth=0.4, linestyle="dashed", color="gray", alpha=0.55, zorder=6)
            gl.xlocator = mticker.FixedLocator(lon_t)
            gl.ylocator = mticker.FixedLocator(lat_t)
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        ax.coastlines(resolution="110m", linewidth=0.5, color="dimgray", alpha=0.6, zorder=10)

    im_ref = None
    for row_idx in range(nrows_data):
        for col_idx in range(ncols_data):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=proj_axes)

            if row_idx == 0:
                data_2d = np.ma.masked_invalid(true_mm[col_idx])
            else:
                data_2d = np.ma.masked_invalid(preds_mm[row_idx - 1][0][col_idx])

            im = ax.imshow(data_2d, origin="lower", extent=extent, transform=data_crs,
                           cmap=cmap_obj, vmin=vmin_plot, vmax=vmax, interpolation="nearest")
            if im_ref is None:
                im_ref = im
            _decorate(ax,
                      show_xlabels=(row_idx == nrows_data - 1),
                      show_ylabels=(col_idx == 0))

            # Column title: top row only
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=title_fontsize + 1, pad=5)

            # Row label: leftmost column only
            if col_idx == 0:
                ax.text(-0.22, 0.5, row_labels[row_idx],
                        transform=ax.transAxes,
                        fontsize=title_fontsize,
                        va="center", ha="right", rotation=90, clip_on=False)

    # Single colorbar at bottom spanning all columns
    cax_row = fig.add_subplot(gs[-1, :])
    cax_row.set_axis_off()
    inset = inset_axes(cax_row, width="55%", height="75%",
                       loc="lower center", borderpad=0.3)
    cbar = fig.colorbar(im_ref, cax=inset, orientation="horizontal")
    cbar.set_label("Precipitation (mm/day)", fontsize=11)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)
    cbar.locator = mticker.FixedLocator([5, 15, 25, 35, 45])
    cbar.update_ticks()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


'''
#===========빨간 박스 영역 고해상도 시각화=============
@torch.no_grad()
def save_highres_zoom_GT_vs_pred(
    pred, true, *,
    ch_idx: int = 0,
    t_idx: int = 0,
    lon=None, lat=None,
    extent_zoom=(280, 340, -15, 15),     # lon 0–360E 기준
    central_longitude: float = 180.0,
    left_title: str = "(a) Ground Truth",
    right_title: str = "(b) Prediction",

    # --- RHD 표시용: 모델명 + 포맷 ---
    model_name: str = "FACL",
    rhd_decimals: int = 3,
    show_rhd_on_gt: bool = False,     # True면 GT 타이틀에도 RHD 표시

    # --- RHD 계산 파라미터 (줌 영역 기반) ---
    rhd_psz: int = 15,
    rhd_bins: int = 50,
    rhd_epsilon: float = 1e-6,
    rhd_robust: bool = True,
    rhd_q_low: float = 0.01,
    rhd_q_high: float = 0.99,
    rhd_quant_sample: int = 200_000,
    rhd_device: str = "cpu",          # "cpu" | "pred"

    palette: str = "white_blue_yellow_red",
    vmin_pct: float = 1.0, vmax_pct: float = 99.0,
    vmax_override: float = None,        # mm/day 단위로 직접 지정 시 vmax_pct 무시
    mu_tp: float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    clamp_zero: bool = True,
    scale_scope: str = "global",        # "global" | "zoom"
    scale_source: str = "gt",           # "gt" | "gt+pred"
    cbar_ticks=None,
    save_path: str = "zoom_pair_highres.png",
    dpi: int = 600,
    tick_labelsize: int = 13,
    title_fontsize: int = 18,
):
    """
    pred/true: (B,C,T,H,W) or (B,T,H,W)
    지정된 extent_zoom 영역만 확대해서 저장 (좌: GT, 우: Pred).

    ✅ 추가:
      - 확대(zoom)된 영역(=true_zoom, pred_zoom)만 가지고 RHD(KL-hist) 계산
      - 우측 타이틀을 "{MODEL} (RHD=...)" 형태로 표시 (예: "MSE (RHD=2.343)")
      - left_title/right_title 기존 문자열을 "prefix"로 쓰고 싶으면 그대로 두고,
        자동 표시를 원하면 right_title="(b)" 정도로 두고 model_name 사용
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cartopy.crs as ccrs

    # ---- 1) 입력 정리 ----
    if pred.dim() == 5:
        pred_t = pred[0, ch_idx]   # (T,H,W)
        true_t = true[0, ch_idx]
    elif pred.dim() == 4:
        pred_t = pred[0]
        true_t = true[0]
    else:
        raise ValueError(f"Unsupported shape {tuple(pred.shape)}")

    T, H, W = true_t.shape
    t = int(np.clip(t_idx, 0, T - 1))
    pred_frame = pred_t[t]  # torch (H,W) z-score
    true_frame = true_t[t]  # torch (H,W) z-score

    # lon/lat
    if lon is None: lon = np.linspace(0, 360, W, endpoint=False)
    if lat is None: lat = np.linspace(-90, 90, H)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # ---- 2) zoom mask (경도 래핑 지원) ----
    lon_min, lon_max, lat_min, lat_max = extent_zoom
    lat_mask = (lat >= lat_min) & (lat <= lat_max)

    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    lon_sel, lat_sel = lon[lon_mask], lat[lat_mask]

    # ---- 3) Zoom tensor (z-score) for RHD ----
    # torch indexing: use np.where indices
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    gt_zoom_z = true_frame.index_select(0, torch.from_numpy(lat_idx).to(true_frame.device)) \
                         .index_select(1, torch.from_numpy(lon_idx).to(true_frame.device))
    pr_zoom_z = pred_frame.index_select(0, torch.from_numpy(lat_idx).to(pred_frame.device)) \
                         .index_select(1, torch.from_numpy(lon_idx).to(pred_frame.device))

    # ---- 4) Compute RHD on zoom (z-score space) ----
    if rhd_device == "pred":
        dev = pred_frame.device
    else:
        dev = torch.device("cpu")

    rhd_val = rhd(
        y_pred=pr_zoom_z.to(dev).float(),
        y_true=gt_zoom_z.to(dev).float(),
        patch_size=rhd_psz,
        num_bins=rhd_bins,
        epsilon=rhd_epsilon,
        robust=rhd_robust,
        q_low=rhd_q_low,
        q_high=rhd_q_high,
        quant_sample=rhd_quant_sample,
    )
    rhd_txt = f"{rhd_val:.{rhd_decimals}f}"

    # ---- 5) 역정규화 (mm) for plotting ----
    def to_mm(z_np):
        z_np = z_np * sig_tp + mu_tp
        if clamp_zero: z_np = np.maximum(z_np, 0)
        return z_np * 1000.0

    # numpy frames
    pred_np = pred_frame.detach().cpu().float().numpy()
    true_np = true_frame.detach().cpu().float().numpy()
    true_mm, pred_mm = to_mm(true_np), to_mm(pred_np)

    true_zoom = true_mm[np.ix_(lat_mask, lon_mask)]
    pred_zoom = pred_mm[np.ix_(lat_mask, lon_mask)]

    # ---- 6) 컬러맵 ----
    if palette == "white_blue_yellow_red":
        cmap = LinearSegmentedColormap.from_list(
            "wbyr", ["#ffffff", "#4575b4", "#ffffbf", "#d73027"], N=256
        )
    elif palette == "precip_jma":
        _pc = ["#ffffff","#c8f0ff","#96d2fa","#64b4f0","#3296e6",
               "#50c850","#a0e632","#f0f000","#ffc800","#ff9600",
               "#ff5a00","#e61400","#c80000","#960050","#7800a0","#3c0078"]
        cmap = LinearSegmentedColormap.from_list("precip_jma", _pc, N=256)
    else:
        cmap = plt.get_cmap(palette).copy()
    cmap.set_bad("white", 1.0)
    cmap.set_under("white", 1.0)
    cmap.set_over("#a50026")

    # ---- 7) 퍼센타일 스케일(vmin/vmax) 계산: global/zoom 선택 ----
    if scale_scope not in ("global", "zoom"):
        raise ValueError("scale_scope must be 'global' or 'zoom'")
    if scale_source not in ("gt", "gt+pred"):
        raise ValueError("scale_source must be 'gt' or 'gt+pred'")

    if scale_scope == "global":
        base_gt = true_mm
        base_pred = pred_mm
    else:
        base_gt = true_zoom
        base_pred = pred_zoom

    if scale_source == "gt":
        vals = base_gt[np.isfinite(base_gt)]
    else:
        vals = np.concatenate([base_gt[np.isfinite(base_gt)], base_pred[np.isfinite(base_pred)]], axis=0)

    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(vals, vmin_pct))
        vmax = float(np.percentile(vals, vmax_pct))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = float(np.nanmax(vals)) or 1.0
        if vmin >= vmax: vmin, vmax = 0.0, 1.0
    if vmax_override is not None:
        vmax = float(vmax_override)

    vmin_plot = np.nextafter(vmin, vmax)
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax)

    # ---- 8) 투영 및 figure ----
    data_crs = ccrs.PlateCarree(central_longitude=0)
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    fig = plt.figure(figsize=(11, 4.2), dpi=dpi)
    ax_l = plt.subplot(1, 2, 1, projection=proj)
    ax_r = plt.subplot(1, 2, 2, projection=proj)

    # colorbar extend based on zoom data range
    z_max = np.nanmax([np.nanmax(true_zoom), np.nanmax(pred_zoom)])
    z_min = np.nanmin([np.nanmin(true_zoom), np.nanmin(pred_zoom)])
    has_over = bool(z_max > vmax + 1e-9)
    has_under = bool(z_min < vmin - 1e-9)
    extend_opt = "both" if (has_over and has_under) else ("max" if has_over else ("min" if has_under else "neither"))

    # ---- 9) Titles with RHD ----
    # left: keep your left_title, optionally add RHD too
    if show_rhd_on_gt:
        left_title_use = f"Ground Truth" #f"{left_title} (RHD={rhd_txt})"
    else:
        left_title_use = left_title

    # right: enforce "MODEL (RHD=...)" format (prefix can stay in right_title)
    # If you want "(b) MSE (RHD=...)" style, set right_title="(b)".
    right_title_use = f"RHD={rhd_txt}".strip() #f"{right_title} {model_name} (RHD={rhd_txt})".strip()

    # ---- 10) Plot ----
    for ax, data, title in zip([ax_l, ax_r], [true_zoom, pred_zoom], [left_title_use, right_title_use]):
        im = ax.pcolormesh(
            lon_sel, lat_sel, np.ma.masked_invalid(data),
            cmap=cmap, norm=norm, transform=data_crs, shading="auto"
        )
        ax.coastlines(resolution="110m", linewidth=0.8, color="dimgray", alpha=0.8)
        ax.set_extent(extent_zoom, crs=data_crs)
        ax.set_title(title, fontsize=title_fontsize, loc="left")

        # inset colorbar
        cax = inset_axes(
            ax, width="80%", height="7%", loc="lower center",
            bbox_to_anchor=(0.1, -0.25, 0.8, 1.0),
            bbox_transform=ax.transAxes, borderpad=0
        )
        cb = plt.colorbar(im, cax=cax, orientation="horizontal", extend=extend_opt)
        cb.set_label("Precipitation (mm)", fontsize=tick_labelsize - 1)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        cb.ax.tick_params(labelsize=tick_labelsize - 2)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.23, wspace=0.07)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.canvas.draw()
    fig.savefig(save_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    return {"rhd_zoom": rhd_val}
'''

@torch.no_grad()
def save_highres_zoom_GT_vs_pred(
    pred, true, *,
    ch_idx: int = 0,
    t_idx: int = 0,
    lon=None, lat=None,
    extent_zoom=(280, 340, -15, 15),     # lon 0–360E 기준
    central_longitude: float = 180.0,
    left_title: str = "(a) Ground Truth",
    right_title: str = "(b) Prediction",

    # --- ✅ FSS 옵션 (LPC 대체) ---
    model_name: str = "FACL",
    fss_threshold: float = 1.0,          # mm/day
    fss_window: int = 5,                 # window_size
    fss_decimals: int = 3,

    # --- RHD 옵션 (None이면 FSS 표시, 값 지정 시 RHD 표시) ---
    rhd_psz: int = None,
    rhd_bins: int = 50,
    rhd_epsilon: float = 1e-6,
    rhd_robust: bool = True,
    rhd_q_low: float = 0.01,
    rhd_q_high: float = 0.99,
    rhd_quant_sample: int = 200_000,
    rhd_device: str = "cpu",
    rhd_decimals: int = 3,

    palette: str = "white_blue_yellow_red",
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
    vmax_override: float = None,         # mm/day 단위로 직접 지정 시 vmax_pct 무시
    mu_tp: float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    clamp_zero: bool = True,
    scale_scope: str = "global",         # "global" | "zoom"
    scale_source: str = "gt",            # "gt" | "gt+pred"
    cbar_ticks=None,
    save_path: str = "zoom_pair_highres.png",
    dpi: int = 600,
    tick_labelsize: int = 13,
    title_fontsize: int = 18,
):
    """
    pred/true: (B,C,T,H,W) or (B,T,H,W)

    변경 사항:
      - ✅ LPC 제거
      - ✅ zoom 영역(mm/day)에서 FSS 계산 후 제목/반환에 포함
    """
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cartopy.crs as ccrs

    # ------------------------------------------------------------
    # 0) helpers: _to_bchw, nan guard, fss (네 구현 기반)
    # ------------------------------------------------------------
    def _to_bchw(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 2:          # (H,W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:        # (B,H,W)
            x = x.unsqueeze(1)
        elif x.dim() == 4:        # (B,C,H,W)
            if x.size(1) != 1:
                x = x[:, :1]
        else:
            raise ValueError(f"_to_bchw expects 2D/3D/4D, got {tuple(x.shape)}")
        return x

    def _nan_guard_scalar(v: float) -> float:
        return float(v) if np.isfinite(v) else float("nan")

    @torch.no_grad()
    def fss_torch(y_pred, y_true, threshold, window_size=5):
        y_true_bchw = _to_bchw(y_true).float()
        y_pred_bchw = _to_bchw(y_pred).float()

        # pairwise valid mask (NaN 제거)
        valid = torch.isfinite(y_true_bchw) & torch.isfinite(y_pred_bchw)
        y_true_bchw = torch.where(valid, y_true_bchw, torch.zeros_like(y_true_bchw))
        y_pred_bchw = torch.where(valid, y_pred_bchw, torch.zeros_like(y_pred_bchw))

        y_true_bin = (y_true_bchw >= threshold).float()
        y_pred_bin = (y_pred_bchw >= threshold).float()

        kernel = torch.ones((1, 1, window_size, window_size),
                            device=y_true_bchw.device, dtype=y_true_bchw.dtype)

        y_true_sum = F.conv2d(y_true_bin, kernel, padding=window_size // 2)
        y_pred_sum = F.conv2d(y_pred_bin, kernel, padding=window_size // 2)

        num_pixels_in_window = window_size * window_size
        y_true_frac = y_true_sum / num_pixels_in_window
        y_pred_frac = y_pred_sum / num_pixels_in_window

        mse_frac = ((y_true_frac - y_pred_frac) ** 2).mean()
        ref_mse = (y_true_frac ** 2).mean() + (y_pred_frac ** 2).mean()
        fss_score = 1.0 - mse_frac / (ref_mse + 1e-8)
        return _nan_guard_scalar(float(fss_score.item()))

    # ------------------------------------------------------------
    # 1) 입력 정리
    # ------------------------------------------------------------
    if pred.dim() == 5:
        pred_t = pred[0, ch_idx]   # (T,H,W)
        true_t = true[0, ch_idx]
    elif pred.dim() == 4:
        pred_t = pred[0]
        true_t = true[0]
    else:
        raise ValueError(f"Unsupported shape {tuple(pred.shape)}")

    T, H, W = true_t.shape
    t = int(np.clip(t_idx, 0, T - 1))
    pred_frame = pred_t[t]  # (H,W) z-score
    true_frame = true_t[t]

    # lon/lat
    if lon is None: lon = np.linspace(0, 360, W, endpoint=False)
    if lat is None: lat = np.linspace(-90, 90, H)
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # ------------------------------------------------------------
    # 2) zoom mask (index)
    # ------------------------------------------------------------
    lon_min, lon_max, lat_min, lat_max = extent_zoom
    lat_mask = (lat >= lat_min) & (lat <= lat_max)

    if lon_min <= lon_max:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
    else:
        lon_mask = (lon >= lon_min) | (lon <= lon_max)

    lon_sel = lon[lon_mask]
    lat_sel = lat[lat_mask]
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    # ------------------------------------------------------------
    # 3) 역정규화 (mm/day)
    # ------------------------------------------------------------
    def to_mm(z_np):
        x_m = z_np * sig_tp + mu_tp
        if clamp_zero:
            x_m = np.maximum(x_m, 0.0)
        return x_m * 1000.0

    pred_np = pred_frame.detach().cpu().float().numpy()
    true_np = true_frame.detach().cpu().float().numpy()
    pred_mm = to_mm(pred_np)
    true_mm = to_mm(true_np)

    true_zoom = true_mm[np.ix_(lat_mask, lon_mask)]
    pred_zoom = pred_mm[np.ix_(lat_mask, lon_mask)]

    # ------------------------------------------------------------
    # 4) ✅ FSS 계산 (zoom 영역, mm/day)
    # ------------------------------------------------------------
    # conv2d 위해 torch tensor로 변환 (H,W)
    gt_zoom_t = torch.as_tensor(true_zoom, dtype=torch.float32)
    pr_zoom_t = torch.as_tensor(pred_zoom, dtype=torch.float32)

    fss_val = fss_torch(pr_zoom_t, gt_zoom_t, threshold=float(fss_threshold), window_size=int(fss_window))

    # ------------------------------------------------------------
    # 4-2) RHD 계산 (rhd_psz 지정 시)
    # ------------------------------------------------------------
    rhd_val = None
    if rhd_psz is not None:
        _dev = torch.device("cpu") if rhd_device == "cpu" else pred_frame.device
        _lat_idx = np.where(lat_mask)[0]
        _lon_idx = np.where(lon_mask)[0]
        gt_zoom_z = true_frame.index_select(0, torch.from_numpy(_lat_idx).to(true_frame.device)) \
                               .index_select(1, torch.from_numpy(_lon_idx).to(true_frame.device))
        pr_zoom_z = pred_frame.index_select(0, torch.from_numpy(_lat_idx).to(pred_frame.device)) \
                               .index_select(1, torch.from_numpy(_lon_idx).to(pred_frame.device))
        rhd_val = rhd(
            y_pred=pr_zoom_z.to(_dev).float(),
            y_true=gt_zoom_z.to(_dev).float(),
            patch_size=rhd_psz,
            num_bins=rhd_bins,
            epsilon=rhd_epsilon,
            robust=rhd_robust,
            q_low=rhd_q_low,
            q_high=rhd_q_high,
            quant_sample=rhd_quant_sample,
        )

    # ------------------------------------------------------------
    # 5) 컬러맵
    # ------------------------------------------------------------
    if palette == "white_blue_yellow_red":
        cmap = LinearSegmentedColormap.from_list(
            "wbyr", ["#ffffff", "#4575b4", "#ffffbf", "#d73027"], N=256
        )
    elif palette == "precip_jma":
        _pc = ["#ffffff","#c8f0ff","#96d2fa","#64b4f0","#3296e6",
               "#50c850","#a0e632","#f0f000","#ffc800","#ff9600",
               "#ff5a00","#e61400","#c80000","#960050","#7800a0","#3c0078"]
        cmap = LinearSegmentedColormap.from_list("precip_jma", _pc, N=256)
    else:
        cmap = plt.get_cmap(palette).copy()

    cmap.set_bad("white", 1.0)
    cmap.set_under("white", 1.0)
    cmap.set_over("#a50026")

    # ------------------------------------------------------------
    # 6) vmin/vmax
    # ------------------------------------------------------------
    if scale_scope == "global":
        base_gt, base_pred = true_mm, pred_mm
    else:
        base_gt, base_pred = true_zoom, pred_zoom

    if scale_source == "gt":
        vals = base_gt[np.isfinite(base_gt)]
    else:
        vals = np.concatenate(
            [base_gt[np.isfinite(base_gt)], base_pred[np.isfinite(base_pred)]], axis=0
        )

    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(vals, vmin_pct))
        vmax = float(np.percentile(vals, vmax_pct))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = float(np.nanmax(vals)) or 1.0
        if vmin >= vmax: vmin, vmax = 0.0, 1.0
    if vmax_override is not None:
        vmax = float(vmax_override)

    vmin_plot = np.nextafter(vmin, vmax)
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax)

    # ------------------------------------------------------------
    # 7) Plot
    # ------------------------------------------------------------
    data_crs = ccrs.PlateCarree(central_longitude=0)
    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    fig = plt.figure(figsize=(11, 4.2), dpi=dpi)
    ax_l = plt.subplot(1, 2, 1, projection=proj)
    ax_r = plt.subplot(1, 2, 2, projection=proj)

    # 제목: RHD 지정 시 RHD 표기, 아니면 FSS 표기
    if rhd_val is not None:
        metric_str = f"RHD={rhd_val:.{rhd_decimals}f}"
    else:
        metric_str = f"FSS={fss_val:.{fss_decimals}f}"
    titles = [
        left_title,
        metric_str,
    ]

    for ax, data, title in zip([ax_l, ax_r], [true_zoom, pred_zoom], titles):
        im = ax.pcolormesh(
            lon_sel, lat_sel, np.ma.masked_invalid(data),
            cmap=cmap, norm=norm, transform=data_crs, shading="auto"
        )
        ax.coastlines(resolution="110m", linewidth=0.8, color="dimgray", alpha=0.8)
        ax.set_extent(extent_zoom, crs=data_crs)
        ax.set_title(title, fontsize=title_fontsize, loc="left")

        cax = inset_axes(
            ax, width="80%", height="7%", loc="lower center",
            bbox_to_anchor=(0.1, -0.25, 0.8, 1.0),
            bbox_transform=ax.transAxes, borderpad=0
        )
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Precipitation (mm/day)", fontsize=tick_labelsize - 1)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        cb.ax.tick_params(labelsize=tick_labelsize - 2)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.23, wspace=0.07)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    out = {
        "fss_zoom": float(fss_val),
        "fss_threshold": float(fss_threshold),
        "fss_window": int(fss_window),
    }
    if rhd_val is not None:
        out["rhd_zoom"] = float(rhd_val)
    return out










def save_highres_zoom_4panel(
    true,
    preds_list,          # list of (pred_tensor, label_str)
    *,
    ch_idx: int = 0,
    t_idx: int = 0,
    lon=None, lat=None,
    extent_zoom=(130, 210, -15, 25),
    central_longitude: float = 180.0,
    palette: str = "precip_jma",
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
    vmax_override: float = None,
    mu_tp: float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    clamp_zero: bool = True,
    rhd_psz: int = 15,
    rhd_bins: int = 50,
    rhd_epsilon: float = 1e-6,
    rhd_robust: bool = True,
    rhd_q_low: float = 0.01,
    rhd_q_high: float = 0.99,
    rhd_quant_sample: int = 200_000,
    rhd_device: str = "cpu",
    save_path: str = "map3_4panel.png",
    dpi: int = 600,
    tick_labelsize: int = 14,
    title_fontsize: float = 19.5,
):
    """GT + 3모델 예측을 1×4 패널로 비교. 각 예측 패널에 RHD(소수 3자리) 표시."""
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cartopy.crs as ccrs

    def _to_frame(t, ch, ti):
        if t.dim() == 5:
            return t[0, ch]
        elif t.dim() == 4:
            return t[0]
        raise ValueError(f"Unsupported shape {tuple(t.shape)}")

    # -- lon/lat --
    frame0 = _to_frame(true, ch_idx, t_idx)
    T, H, W = frame0.shape
    ti = int(np.clip(t_idx, 0, T - 1))
    if lon is None: lon = np.linspace(0, 360, W, endpoint=False)
    if lat is None: lat = np.linspace(-90, 90, H)
    lon, lat = np.asarray(lon), np.asarray(lat)

    # -- zoom mask --
    lon_min, lon_max, lat_min, lat_max = extent_zoom
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max) if lon_min <= lon_max \
               else (lon >= lon_min) | (lon <= lon_max)
    lon_sel, lat_sel = lon[lon_mask], lat[lat_mask]
    lat_idx_np = np.where(lat_mask)[0]
    lon_idx_np = np.where(lon_mask)[0]

    def to_mm(z_np):
        x = z_np * sig_tp + mu_tp
        if clamp_zero: x = np.maximum(x, 0.0)
        return x * 1000.0

    true_frame = _to_frame(true, ch_idx, t_idx)[ti]
    true_np = true_frame.detach().cpu().float().numpy()
    true_mm = to_mm(true_np)
    true_zoom = true_mm[np.ix_(lat_mask, lon_mask)]

    # -- vmin/vmax from GT --
    vals = true_mm[np.isfinite(true_mm)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(vals, vmin_pct))
        vmax = float(np.percentile(vals, vmax_pct))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = float(np.nanmax(vals)) or 1.0
        if vmin >= vmax: vmin, vmax = 0.0, 1.0
    if vmax_override is not None:
        vmax = float(vmax_override)
    vmin_plot = np.nextafter(vmin, vmax)

    # -- colormap --
    if palette == "precip_jma":
        _pc = ["#ffffff","#c8f0ff","#96d2fa","#64b4f0","#3296e6",
               "#50c850","#a0e632","#f0f000","#ffc800","#ff9600",
               "#ff5a00","#e61400","#c80000","#960050","#7800a0","#3c0078"]
        cmap = LinearSegmentedColormap.from_list("precip_jma", _pc, N=256)
    elif palette == "white_blue_yellow_red":
        cmap = LinearSegmentedColormap.from_list(
            "wbyr", ["#ffffff","#4575b4","#ffffbf","#d73027"], N=256)
    else:
        cmap = plt.get_cmap(palette).copy()
    cmap.set_bad("white", 1.0)
    cmap.set_under("white", 1.0)
    cmap.set_over("#a50026")
    norm = mpl.colors.Normalize(vmin=vmin_plot, vmax=vmax)

    # -- RHD 계산 헬퍼 --
    _dev = torch.device("cpu") if rhd_device == "cpu" else true_frame.device
    def compute_rhd(pred_tensor):
        pf = _to_frame(pred_tensor, ch_idx, t_idx)[ti]
        gt_z = true_frame.index_select(0, torch.from_numpy(lat_idx_np).to(true_frame.device)) \
                          .index_select(1, torch.from_numpy(lon_idx_np).to(true_frame.device))
        pr_z = pf.index_select(0, torch.from_numpy(lat_idx_np).to(pf.device)) \
                 .index_select(1, torch.from_numpy(lon_idx_np).to(pf.device))
        return rhd(y_pred=pr_z.to(_dev).float(), y_true=gt_z.to(_dev).float(),
                   patch_size=rhd_psz, num_bins=rhd_bins, epsilon=rhd_epsilon,
                   robust=rhd_robust, q_low=rhd_q_low, q_high=rhd_q_high,
                   quant_sample=rhd_quant_sample)

    # -- 패널 데이터 준비 --
    panels = [("Ground Truth", true_zoom, None)]
    for pred_t, lbl in preds_list:
        pf = _to_frame(pred_t, ch_idx, t_idx)[ti]
        pred_mm = to_mm(pf.detach().cpu().float().numpy())
        pred_zoom = pred_mm[np.ix_(lat_mask, lon_mask)]
        rhd_val = compute_rhd(pred_t)
        title = f"RHD={rhd_val:.3f}"
        panels.append((title, pred_zoom, rhd_val))

    # -- 그리기 --
    n = len(panels)
    data_crs = ccrs.PlateCarree(central_longitude=0)
    proj     = ccrs.PlateCarree(central_longitude=central_longitude)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), dpi=dpi,
                              subplot_kw={"projection": proj})
    if n == 1:
        axes = [axes]

    for ax, (title, data, _) in zip(axes, panels):
        im = ax.pcolormesh(lon_sel, lat_sel, np.ma.masked_invalid(data),
                           cmap=cmap, norm=norm, transform=data_crs, shading="auto")
        ax.coastlines(resolution="110m", linewidth=0.8, color="dimgray", alpha=0.8)
        ax.set_extent(list(extent_zoom), crs=data_crs)
        ax.set_title(title, fontsize=title_fontsize, loc="left")
        cax = inset_axes(ax, width="80%", height="7%", loc="lower center",
                         bbox_to_anchor=(0.1, -0.25, 0.8, 1.0),
                         bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Precipitation (mm/day)", fontsize=tick_labelsize - 1)
        cb.ax.tick_params(labelsize=tick_labelsize - 2)
        cb.locator = mpl.ticker.FixedLocator([5, 15, 25, 35, 45])
        cb.update_ticks()

    plt.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.23, wspace=0.07)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return {lbl: panels[i + 1][2] for i, (_, lbl) in enumerate(preds_list)}


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
    base_time: str = "",
    title_suffix: str = "",
    save_path: str = None,
    # --- LPC 파라미터 ---
    scales=(1.0, 1.5, 2.0),
    weights=(1.0, -3.0, 2.0),
    norient=8,
    C=2.0,
    Beta_k=1e-4,
    B=None,
    # --- 그리기 옵션 ---
    cmap: str = "hot_r",        # 작은 값 밝음, 큰 값 어둠
    show_diff: bool = False,
    dpi: int = 300,
    lpc_out_dir: str = None,
    extent=(0, 360, -90, 90),
    central_longitude: float = 180.0,
    # --- 스타일(visualize_prediction_map과 형식 맞춤) ---
    tick_labelsize: int = 13,
    title_fontsize: int = 14,
    suptitle_fontsize: int = 12,
    # --- 레이아웃 제어(겹침 방지) ---
    figure_width_unit: float = 5.0,   # 열당 가로 크기 배수(살짝 확대)
    figure_height_unit: float = 3.3,  # 행당 세로 크기 배수(살짝 확대)
    wspace: float = 0.22,             # 열 간격 ↑
    hspace: float = 0.30,             # 행 간격 ↑
    cbar_height_ratio: float = 0.14,  # 하단 컬러바 전용 행 높이 비율
    margins=(0.06, 0.12, 0.10, 0.06), # (left, right, top, bottom) 여백 비율
    # --- 새로 추가: 표시 영역 크롭 (경도는 0~360 체계) ---
    crop_to_60: bool = True,
    crop_extent: tuple = (60.0, 300.0, -60.0, 60.0),  # (lon_min, lon_max, lat_min, lat_max)
    # --- 새로 추가: LPC mean 주석 옵션 ---
    show_lpc_mean: bool = False,
    lpc_mean_scope: str = "global",   # "global" | "view"
):
    """
    pred/true: (B,1,T,H,W)
    - 각 시점(T)마다 (a) GT, (b) Pred(FACL/MSE) [옵션: (c) Diff]를 같은 스케일로 시각화
    - 맵들 겹침 방지를 위해 여백(wspace/hspace/figsize) 확보 + 컬러바 전용 영역
    - 공용 컬러바는 '가장 아래 가로형' 1개로 배치
    - crop_to_60=True이면 60S~60N, 60E~60W(=300°) 영역만 보여줌 (계산은 글로벌 기준 유지)
    - show_lpc_mean=True이면 각 패널 우측 상단에 LPC mean 값을 표시
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # --- 입력 텐서 정리 ---
    gt = true[0, 0].detach().cpu()   # (T,H,W)
    pr = pred[0, 0].detach().cpu()
    T, H, W = gt.shape

    ncols_maps = 3 if show_diff else 2
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree()

    if lpc_out_dir:
        os.makedirs(lpc_out_dir, exist_ok=True)

    lon = np.linspace(extent[0], extent[1], W, endpoint=False)
    lat = np.linspace(extent[2], extent[3], H)

    # ---- 1) 모든 시점에 대해 LPC 미리 계산해 공통 vmax 결정 (계산은 글로벌) ----
    lmap_gt_list, lmap_pr_list, si_true_list, si_pred_list = [], [], [], []
    vmax_lpc = 0.0

    for t in range(T):
        gt_img = _roll_longitude_to_center(gt[t], current_center_deg=0.0, target_center_deg=180.0)
        pr_img = _roll_longitude_to_center(pr[t], current_center_deg=0.0, target_center_deg=180.0)

        si_gt, lmap_gt = lpc_si_torch(
            gt_img.to(torch.float32),
            C=C, Beta_k=Beta_k, scales=scales, norient=norient, B=B, weights=weights
        )
        si_pr, lmap_pr = lpc_si_torch(
            pr_img.to(torch.float32),
            C=C, Beta_k=Beta_k, scales=scales, norient=norient, B=B, weights=weights
        )

        si_true_list.append(si_gt)
        si_pred_list.append(si_pr)
        lmap_gt_np = lmap_gt.cpu().numpy()
        lmap_pr_np = lmap_pr.cpu().numpy()
        lmap_gt_list.append(lmap_gt_np)
        lmap_pr_list.append(lmap_pr_np)

        vmax_lpc = max(vmax_lpc, float(np.nanmax(lmap_gt_np)), float(np.nanmax(lmap_pr_np)))

    vmin_lpc = 0.0

    # ---- 2) Figure & GridSpec (하단 컬러바 전용 행) ----
    fig_width  = figure_width_unit * ncols_maps
    fig_height = figure_height_unit * T + (figure_height_unit * cbar_height_ratio)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, constrained_layout=True)

    height_ratios = [1.0] * T + [cbar_height_ratio]
    gs = fig.add_gridspec(
        nrows=T + 1, ncols=ncols_maps,
        height_ratios=height_ratios,
        wspace=wspace, hspace=hspace,
        left=margins[0], right=1.0 - margins[1],
        top=1.0 - margins[2], bottom=margins[3],
    )

    if base_time or title_suffix:
        fig.suptitle(f"[LPC] t₀={base_time}  {title_suffix}", fontsize=suptitle_fontsize)

    # --- 크롭 범위 및 눈금 ---
    if crop_to_60:
        # 경도 눈금: 60E, 120E, 180, 120W(=240), 60W(=300)
        lon_ticks = np.array([60, 120, 180, 240, 300], dtype=float)
        lat_ticks = np.array([-60, -30, 0, 30, 60], dtype=float)
        view_extent = crop_extent  # (60, 300, -60, 60)
    else:
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90, 91, 30)
        view_extent = extent

    # --- lpc_mean 계산 스코프 선택: global vs view ---
    # view 스코프일 때 사용할 마스크(경도 0~360 체계 가정; wrap 없음)
    if lpc_mean_scope not in ("global", "view"):
        raise ValueError("lpc_mean_scope must be 'global' or 'view'")

    if lpc_mean_scope == "view":
        lon_min, lon_max, lat_min, lat_max = view_extent
        lat_mask_view = (lat >= lat_min) & (lat <= lat_max)
        if lon_min <= lon_max:
            lon_mask_view = (lon >= lon_min) & (lon <= lon_max)
        else:
            # (필요 시) 경계 래핑 케이스
            lon_mask_view = (lon >= lon_min) | (lon <= lon_max)
        # 2D 인덱싱용
        view_indexer = np.ix_(lat_mask_view, lon_mask_view)
    else:
        view_indexer = None  # global

    # 공통 축 꾸미기
    def decorate(ax):
        ax.set_extent(view_extent, crs=data_crs)

        ax.set_xticks(lon_ticks, crs=data_crs)
        ax.set_yticks(lat_ticks, crs=data_crs)
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))

        gl = ax.gridlines(
            crs=data_crs,
            draw_labels=False,
            linewidth=0.4,
            linestyle="dashed",
            color="gray",
            alpha=0.55,
            zorder=6,
        )
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)

        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
        ax.coastlines(resolution="110m", linewidth=0.5, color="dimgray", alpha=0.6, zorder=10)

        # --- 강제 세로 점선 추가: 120°W = 240°E ---
        if crop_to_60:
            ax.plot(
                [240, 240], 
                [lat_ticks.min(), lat_ticks.max()],
                transform=data_crs,
                linestyle="dashed",
                linewidth=0.6,
                color="gray",
                alpha=0.55,
                zorder=7
            )


    # --- 3) 맵 그리기 (글로벌 필드 위에 set_extent로 뷰만 크롭) ---
    im_for_cbar = None
    for t in range(T):
        # ── (a) Ground Truth
        ax0 = fig.add_subplot(gs[t, 0], projection=proj)
        im0 = ax0.pcolormesh(
            lon, lat, lmap_gt_list[t],
            cmap=cmap, vmin=vmin_lpc, vmax=vmax_lpc,
            transform=data_crs, shading="auto"
        )
        ax0.set_title("(a) Ground Truth", fontsize=title_fontsize, loc="left")
        decorate(ax0)

        # ── (b) Prediction (FACL/MSE)
        ax1 = fig.add_subplot(gs[t, 1], projection=proj)
        im1 = ax1.pcolormesh(
            lon, lat, lmap_pr_list[t],
            cmap=cmap, vmin=vmin_lpc, vmax=vmax_lpc,
            transform=data_crs, shading="auto"
        )
        ax1.set_title("(d) WFCL", fontsize=title_fontsize, loc="left")
        decorate(ax1)

        # ── LPC mean 주석 (GT / Pred)
        if show_lpc_mean:
            if view_indexer is None:
                m_gt = float(np.nanmean(lmap_gt_list[t]))
                m_pr = float(np.nanmean(lmap_pr_list[t]))
            else:
                m_gt = float(np.nanmean(lmap_gt_list[t][view_indexer]))
                m_pr = float(np.nanmean(lmap_pr_list[t][view_indexer]))

            # 우측 상단(axes 좌표계) - 얇은 흰 배경 상자
            # ── LPC mean: 맵 바깥 우측 상단에 표시
            box_kw = dict(facecolor="white", alpha=0.78, edgecolor="none", boxstyle="round,pad=0.25")

            # Ground Truth
            # ── LPC mean: 맵 위쪽 바깥 영역에 표시
            ax0.text(
                1.0, 1.05, f" ",
                transform=ax0.transAxes,
                ha="right", va="bottom",
                fontsize=max(tick_labelsize-1, 8),
                zorder=50, clip_on=False
            )
            ax1.text(
                1.0, 1.05, f" ",
                transform=ax1.transAxes,
                ha="right", va="bottom",
                fontsize=max(tick_labelsize-1, 8),
                zorder=50, clip_on=False
            )



        if im_for_cbar is None:
            im_for_cbar = im1

        # ── (c) Diff (선택)
        if show_diff:
            ax2 = fig.add_subplot(gs[t, 2], projection=proj)
            diff = lmap_pr_list[t] - lmap_gt_list[t]
            dmax = np.percentile(np.abs(diff), 99.0) + 1e-8
            im2 = ax2.pcolormesh(
                lon, lat, diff,
                cmap="coolwarm", vmin=-dmax, vmax=dmax,
                transform=data_crs, shading="auto"
            )
            ax2.set_title(f"(c) Diff (Pred-GT) (t₀+{t+1})", fontsize=title_fontsize, loc="left")
            decorate(ax2)
            # Diff에는 lpc_mean 표기는 생략 (요청 시 추가 가능)

    # ---- 4) 하단 공용 컬러바(가로형, 전용 축) ----
    if im_for_cbar is not None:
        gs_cbar = gs[-1, :].subgridspec(1, 1)
        cax = fig.add_subplot(gs_cbar[0, 0])
        cbar = fig.colorbar(im_for_cbar, cax=cax, orientation="horizontal")
        cbar.set_label("Intensity", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

    # 저장/표시
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

    return si_true_list, si_pred_list


import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from utils.config_utils import load_config
from utils.save_utils import load_checkpoint_for_eval
from datasets.dataloader import get_test_loader
from models.get_model import build_model


@torch.no_grad()
def plot_lpips_grid_by_loss_and_leadtime(
    loss_specs,
    out_png: str,
    sample_idx: int = 0,
    ch_idx: int = 0,
    t1_idx: int = 0,   # 1-day
    t2_idx: int = 1,   # 2-day
    t3_idx: int = 2,   # 3-day
    patch: int = 32,
    stride: int = 16,
    central_longitude: float = 180.0,
    extent=(0, 360, -90, 90),
    add_gridlines: bool = True,

    # --- text sizes ---
    tick_labelsize: int = 9,
    title_fontsize: int = 12,
    suptitle_fontsize: int = 12,
    row_label_fontsize: int = 11,

    # --- LPIPS scale ---
    lpips_vmin: float | None = None,
    lpips_vmax: float | None = None,
    lpips_p_lo: float = 0.5,
    lpips_p_hi: float = 99.5,

    # --- delta (3-day - 1-day) ---
    delta_fixed_range: tuple[float, float] = (-0.05, 0.05),  # ✅ 항상 고정
    abs_delta: bool = False,

    # --- layout ---
    figsize=(13.8, 6.8),

    # ✅ bottom colorbar row controls
    cbar_row_ratio: float = 0.12,    # 하단 컬러바 row 높이(비율)
    cbar_h: str = "55%",             # 두 컬러바 두께(동일)
    cbar_gap: float = 0.12,          # 두 컬러바 사이 간격(axes 내부)
    cbar_lp_w: str = "62%",          # ✅ 1~2열용(LPIPS) 컬러바: 지금보다 "조금 줄인" 길이
    cbar_d_w: str = "30%",           # ✅ 3열(Δ) 컬러바 길이
    cbar_label_fontsize: int = 11,
    cbar_tick_fontsize: int = 10,

    # grid spacing
    wspace: float = 0.26,
    hspace: float = 0.30,
    dpi: int = 300,
):
    """
    columns: [2-day] [3-day] [3-day − 1-day]
    rows: loss별

    ✅ 요구사항:
      1) 1열/2열은 같은 LPIPS 컬러바(하나)를 사용하고, 길이를 지금보다 조금 줄여서 표시
      2) 3열은 ΔLPIPS 컬러바를 별도로 추가 (범위 -0.05~0.05 고정)
      3) 두 컬러바는 같은 높이(두께)

    참고:
      - LPIPS 컬러바는 (2-day, 3-day) 공용 스케일(vmin/vmax)
      - Δ 컬러바는 TwoSlopeNorm으로 0 중심 발산, 범위는 delta_fixed_range 고정
    """
    # ---------------------------
    # style
    # ---------------------------
    plt.style.use("default")
    mpl.rcParams.update({
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # colormaps
    def _cmap_lp():
        colors = ["#ffffff", "#2ca25f", "#2cb1a6", "#1f78b4", "#203a8f"]
        cm = LinearSegmentedColormap.from_list("w_g_teal_navy", colors, N=256)
        cm.set_bad("white", 1.0)
        cm.set_under("#f2f2f2")
        return cm

    def _cmap_div():
        colors = ["#4575b4", "#ffffff", "#d73027"]
        cm = LinearSegmentedColormap.from_list("blue_white_red", colors, N=256)
        cm.set_bad("white", 1.0)
        return cm

    cmap_lp = _cmap_lp()
    cmap_div = _cmap_div()

    # ---------------------------
    # lpips model + device
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = get_lpips_model(device=device).eval()

    # ---------------------------
    # helper: sample 1개 로드
    # ---------------------------
    def _get_one_sample(loader, target_idx: int):
        processed = 0
        for batch in loader:
            if isinstance(batch, dict) and all(k in batch for k in ("X1", "X2", "Y")):
                x1 = batch["X1"].to(device)
                x2 = batch["X2"].to(device)
                y  = batch["Y"].to(device)
                bs = x1.size(0)
            else:
                x, y = batch
                if isinstance(x, (list, tuple)):
                    x1, x2 = x[0].to(device), x[1].to(device)
                    bs = x1.size(0)
                else:
                    raise ValueError("Expected dict{X1,X2,Y} or tuple((X1,X2),Y).")
                y = y.to(device)

            if processed + bs > target_idx:
                i = target_idx - processed
                return x1[i:i+1], x2[i:i+1], y[i:i+1]
            processed += bs
        raise IndexError(f"sample_idx={target_idx} exceeds dataset size")

    if len(loss_specs) == 0:
        raise ValueError("loss_specs is empty")

    # common loader
    cfg0 = load_config(loss_specs[0]["cfg"])
    loader = get_test_loader(cfg0)
    x1_1, x2_1, y_1 = _get_one_sample(loader, sample_idx)

    if y_1.dim() != 5:
        raise ValueError(f"Expected Y shape (B,C,T,H,W), got {tuple(y_1.shape)}")

    gt1 = y_1[0, ch_idx, t1_idx].float()
    gt2 = y_1[0, ch_idx, t2_idx].float()
    gt3 = y_1[0, ch_idx, t3_idx].float()

    # ---------------------------
    # run all losses -> maps
    # ---------------------------
    lp_maps = []
    for spec in loss_specs:
        name = str(spec["name"])
        cfg  = load_config(spec["cfg"])
        model = build_model(cfg).to(device).eval()
        _ = load_checkpoint_for_eval(spec["ckpt"], model, device=device)

        pred = model(x1_1, x2_1)
        if pred.dim() != 5:
            raise ValueError(f"Expected pred (B,C,T,H,W), got {tuple(pred.shape)}")

        pr1 = pred[0, ch_idx, t1_idx].float()
        pr2 = pred[0, ch_idx, t2_idx].float()
        pr3 = pred[0, ch_idx, t3_idx].float()

        lp1 = lpips_patch_map_single_frame(pr1, gt1, lpips_model, patch=patch, stride=stride)
        lp2 = lpips_patch_map_single_frame(pr2, gt2, lpips_model, patch=patch, stride=stride)
        lp3 = lpips_patch_map_single_frame(pr3, gt3, lpips_model, patch=patch, stride=stride)

        d31 = (np.asarray(lp3) - np.asarray(lp1))
        if abs_delta:
            d31 = np.abs(d31)

        lp_maps.append({"name": name, "lp2": np.asarray(lp2), "lp3": np.asarray(lp3), "d31": np.asarray(d31)})

    # ---------------------------
    # scales
    # ---------------------------
    def _finite(vals):
        v = np.asarray(vals).ravel()
        return v[np.isfinite(v)]

    # LPIPS vmin/vmax (2-day,3-day 전체 공통)
    if lpips_vmin is None or lpips_vmax is None:
        all_lp = np.concatenate([_finite(d["lp2"]) for d in lp_maps] + [_finite(d["lp3"]) for d in lp_maps])
        vmin_lp = float(np.nanpercentile(all_lp, lpips_p_lo)) if all_lp.size else 1e-12
        vmax_lp = float(np.nanpercentile(all_lp, lpips_p_hi)) if all_lp.size else 1.0
        vmin_lp = max(vmin_lp, 1e-12)
        if not np.isfinite(vmax_lp) or vmin_lp >= vmax_lp:
            vmax_lp = vmin_lp + 1e-9
    else:
        vmin_lp, vmax_lp = float(lpips_vmin), float(lpips_vmax)

    # Δ norm: -0.05~0.05 고정 (요청)
    dmin, dmax = float(delta_fixed_range[0]), float(delta_fixed_range[1])
    if abs_delta:
        norm_d = Normalize(vmin=0.0, vmax=max(abs(dmin), abs(dmax)))
        cmap_d = cmap_lp
    else:
        norm_d = TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)
        cmap_d = cmap_div

    # ---------------------------
    # axes decorator (x=60°, y=30°, 모든 패널 라벨 ON)
    # ---------------------------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    def _decorate(ax):
        ax.coastlines(resolution="110m", linewidth=0.45, color="dimgray", alpha=0.6, zorder=10)

        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-90,   91, 30)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f", degree_symbol="°"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f", degree_symbol="°"))

        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, pad=1, length=3, width=0.8)

        if add_gridlines:
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(), draw_labels=False,
                linewidth=0.35, linestyle="dashed",
                color="silver", alpha=0.55, zorder=6
            )
            gl.xlocator = mticker.FixedLocator(lon_ticks)
            gl.ylocator = mticker.FixedLocator(lat_ticks)

        ax.set_global()
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    # ---------------------------
    # plot layout
    # ---------------------------
    R = len(lp_maps)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    gs = fig.add_gridspec(
        nrows=R + 1, ncols=3,
        height_ratios=[1.0] * R + [cbar_row_ratio],
        hspace=hspace, wspace=wspace
    )

    col_titles = ["2-day", "3-day", "3-day − 1-day"]
    im_lp_for_cbar = None
    im_d_for_cbar = None

    for i, d in enumerate(lp_maps):
        # col0: 2-day (LPIPS)
        ax0 = fig.add_subplot(gs[i, 0], projection=proj)
        im0 = ax0.imshow(
            np.ma.masked_invalid(d["lp2"]),
            origin="lower", extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap_lp, vmin=vmin_lp, vmax=vmax_lp,
            interpolation="nearest"
        )
        _decorate(ax0)

        ax0.text(
            -0.20, 0.5, d["name"],
            transform=ax0.transAxes,
            ha="right", va="center",
            fontsize=row_label_fontsize,
            fontweight="normal",
            clip_on=False
        )

        # col1: 3-day (LPIPS)
        ax1 = fig.add_subplot(gs[i, 1], projection=proj)
        im1 = ax1.imshow(
            np.ma.masked_invalid(d["lp3"]),
            origin="lower", extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap_lp, vmin=vmin_lp, vmax=vmax_lp,
            interpolation="nearest"
        )
        _decorate(ax1)

        # col2: delta (Δ)
        ax2 = fig.add_subplot(gs[i, 2], projection=proj)
        im2 = ax2.imshow(
            np.ma.masked_invalid(d["d31"]),
            origin="lower", extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap_d, norm=norm_d,
            interpolation="nearest"
        )
        _decorate(ax2)

        if i == 0:
            ax0.set_title(col_titles[0], fontsize=title_fontsize, pad=6)
            ax1.set_title(col_titles[1], fontsize=title_fontsize, pad=6)
            ax2.set_title(col_titles[2], fontsize=title_fontsize, pad=6)

        if im_lp_for_cbar is None:
            im_lp_for_cbar = im0
        if im_d_for_cbar is None:
            im_d_for_cbar = im2

    # ---------------------------
    # bottom: two colorbars (same height)
    #   - left: LPIPS (shared for col0/1)  ← 길이 조금 줄임
    #   - right: ΔLPIPS for col2          ← [-0.05, 0.05]
    # ---------------------------
    cax_host = fig.add_subplot(gs[R, :])
    cax_host.set_axis_off()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # LPIPS colorbar (left)
    cax_lp = inset_axes(
        cax_host,
        width=cbar_lp_w,
        height=cbar_h,
        loc="center left",
        borderpad=0.0
    )
    # Δ colorbar (right)
    cax_d = inset_axes(
        cax_host,
        width=cbar_d_w,
        height=cbar_h,
        loc="center right",
        borderpad=0.0
    )

    # 조금 더 벌리고 싶으면 host 내부에서 위치를 살짝 미는 방식(axes fraction)도 가능하지만,
    # 여기서는 width 자체를 줄였고 left/right 정렬로 gap이 자연스럽게 생김.

    cb_lp = fig.colorbar(im_lp_for_cbar, cax=cax_lp, orientation="horizontal")
    cb_lp.set_label("LPIPS (↓)", fontsize=cbar_label_fontsize)
    cb_lp.ax.tick_params(labelsize=cbar_tick_fontsize, pad=1)
    cb_lp.outline.set_linewidth(0.8)

    cb_d = fig.colorbar(im_d_for_cbar, cax=cax_d, orientation="horizontal")
    if abs_delta:
        cb_d.set_label("|ΔLPIPS|", fontsize=cbar_label_fontsize)
    else:
        cb_d.set_label("ΔLPIPS", fontsize=cbar_label_fontsize)
    cb_d.ax.tick_params(labelsize=cbar_tick_fontsize, pad=1)
    cb_d.outline.set_linewidth(0.8)

    fig.suptitle("Lead Time", fontsize=suptitle_fontsize, y=0.98)

    # 여백 확보
    fig.subplots_adjust(left=0.22, right=0.99, top=0.92, bottom=0.10)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)
    print(f"[OK] saved → {out_png}")




#===========리드타임에 따른 메트릭 그래프 (3x2)=============================
import os
import copy
import torch

def compute_metrics_by_leadtime_from_yamls(
    loss_specs,
    lead_days=(1, 2, 3),
    metric_fns=None,            # dict[str -> fn(pred, y, **params)]
    metric_params=None,         # dict or None
    device=None,
):
    """
    loss_specs: list of dict
      [{"name":"MSE", "cfg":"...yaml", "ckpt":"...pt"}, ...]
    lead_days: (1,2,3) 같은 실제 'day' 값
    metric_fns: evaluate_epoch에 넘길 metric 함수들 dict
    return:
      metrics_by_model = {
        "MSE": {"LPIPS":[v1,v2,v3], "CPBD":[...], ...},
        ...
      }

    NOTE
    - label_offsets=list(lead_days) 로 고정된 하나의 loader를 사용하여
      모든 lead에서 동일한 샘플 집합을 보장.
      lead_idx 로 pred/GT에서 해당 time step을 선택.
    """

    if metric_fns is None or len(metric_fns) == 0:
        raise ValueError("metric_fns is required (dict of metric functions).")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import copy
    import torch
    from utils.config_utils import load_config
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model

    metrics_by_model = {}

    # metric_fns에 있는 키들을 "기본 metric set"으로 간주 (누락/순서 안정화 용도)
    metric_names = list(metric_fns.keys())

    for spec in loss_specs:
        name = str(spec["name"])
        cfg_path = spec["cfg"]
        ckpt_path = spec["ckpt"]

        cfg = load_config(cfg_path)

        model = build_model(cfg).to(device).eval()
        _ = load_checkpoint_for_eval(ckpt_path, model, device=device)

        # 모든 lead에서 동일한 샘플 집합을 사용하기 위해 loader를 한 번만 생성
        cfg_fixed = copy.deepcopy(cfg)
        cfg_fixed.setdefault("data", {})
        cfg_fixed["data"]["label_offsets"] = [int(ld) for ld in lead_days]
        loader = get_test_loader(cfg_fixed)

        # model별로 leadtime별 metric 누적
        per_metric_lists = {mn: [] for mn in metric_names}  # metric_name -> [lead1, lead2, lead3]

        for i_ld, ld in enumerate(lead_days):
            lead_idx = i_ld   # (1-day→0, 2-day→1, 3-day→2)

            avg_metrics, _, _, _ = evaluate_epoch(
                model=model,
                loader=loader,
                metric_fns=metric_fns,
                device=device,
                metric_params=metric_params,
                lead_idx=lead_idx,
                keep_pred_shape=True,
            )

            for mn in metric_names:
                val = avg_metrics.get(mn, 0.0)
                per_metric_lists[mn].append(float(val))

        metrics_by_model[name] = per_metric_lists

    return metrics_by_model



def plot_metrics_by_leadtime_grid_3x2(
    metrics_by_model: dict,
    out_png: str,
    lead_labels=("1-day", "2-day", "3-day"),
    model_order=("MSE", "FACL", "FACL+WACL Lv3"),
    metric_order=("MSE", "LPIPS", "CPBD", "LPC", "RHD", "CSI-R95p", "FSS", "HD"),
    metric_key_map=None,
    figsize=(10, 10),
    dpi=300,
    y_lim=None,
    sharey=False,
    lpc_abs=False,
    metric_better_higher=None,

    # ✅ extra
    model_label_map=None,

    # ✅ CSI zoom
    csi_zoom=True,
    csi_margin=5e-4,
    csi_pad_ratio=0.05,
    csi_min_range=0.003,

    # ✅ CSI tick control (핵심)
    csi_n_ticks=5,             # ✅ tick 개수 강제 (촘촘 문제 해결)
    csi_tick_decimals=3,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if metric_key_map is None:
        metric_key_map = {
            "MSE" : "mse",
            "LPIPS": "lpips",
            "CPBD": "cpbd",
            "RHD": "rhd",
            "CSI-R95p": "csi-r95p",
            "FSS": "fss",
            "HD": "hausdorff95",
        }

    if model_label_map is None:
        model_label_map = {
            "MSE": "MSE",
            "FACL": "FACL",
            "FACL+WACL Lv3": "FWCL Lv3",
        }

    csi_key_candidates = (
        "csi-r95p",
        "csi_r95p",
        "csi_r95",
        "csi95p",
        "csi_r95p_mean",
        "csi_r95p_t",
        "csi_r95p_by_lead",
        "csi_r95p_lead",
        "csi_r95p_scores",
        "csi",
    )

    if metric_better_higher is None:
        metric_better_higher = {
            "LPIPS": False,
            "CPBD": True,
            "LPC": False,
            "RHD": False,
            "CSI-R95p": True,
            "FSS": True,
        }

    L = len(lead_labels)
    x = np.arange(L)

    n_metrics = len(metric_order)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 3), dpi=dpi, sharey=sharey)
    axes = axes.reshape(-1)

    legend_handles, legend_labels = [], []
    seen_model = set()

    for i, metric_disp in enumerate(metric_order):
        ax = axes[i]
        drew_any = False
        all_y_vals = []

        for model in model_order:
            if model not in metrics_by_model:
                continue
            data = metrics_by_model[model]

            # ---- LPC ----
            if metric_disp == "LPC":
                if "lpc_mean" not in data or "lpc_mean_true" not in data:
                    continue
                y = np.asarray(data["lpc_mean"]) - np.asarray(data["lpc_mean_true"])
                if lpc_abs:
                    y = np.abs(y)
                y = y * 1e4  # 스케일만 하고, 표기는 title에서 처리

            # ---- CSI ----
            elif metric_disp == "CSI-R95p":
                key = None
                preferred = metric_key_map.get(metric_disp, "csi-r95p")
                if preferred in data:
                    key = preferred
                else:
                    for cand in csi_key_candidates:
                        if cand in data:
                            key = cand
                            break
                if key is None:
                    continue
                y = np.asarray(data[key], dtype=float)

            # ---- others ----
            else:
                metric_key = metric_key_map.get(metric_disp, metric_disp)
                if metric_key not in data:
                    continue
                y = np.asarray(data[metric_key], dtype=float)

            if y.ndim == 0 or y.size != L:
                continue

            all_y_vals.append(y)

            label_disp = model_label_map.get(model, model)
            line, = ax.plot(x, y, marker="o", linewidth=2, markersize=4, label=label_disp)
            drew_any = True

            if model not in seen_model:
                legend_handles.append(line)
                legend_labels.append(label_disp)
                seen_model.add(model)

        # ---- CSI y-axis zoom + ticks(개수 제한) ----
        if metric_disp == "CSI-R95p" and csi_zoom and all_y_vals:
            y_all = np.concatenate(all_y_vals)
            y_min = float(np.nanmin(y_all))
            y_max = float(np.nanmax(y_all))

            if np.isfinite(y_min) and np.isfinite(y_max):
                y_range = float(y_max - y_min)

                if y_range < csi_min_range:
                    mid = 0.5 * (y_min + y_max)
                    half = 0.5 * csi_min_range
                    y_lo, y_hi = mid - half, mid + half
                else:
                    pad = max(csi_margin, y_range * csi_pad_ratio)
                    y_lo, y_hi = y_min - pad, y_max + pad

                ax.set_ylim(y_lo, y_hi)

                # ✅ 촘촘 문제 해결: tick 개수 강제
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=csi_n_ticks))
                ax.yaxis.set_major_formatter(
                    mticker.FormatStrFormatter(f"%.{csi_tick_decimals}f")
                )

        # ---- title ----
        arrow = "(↑)" if metric_better_higher.get(metric_disp, False) else "(↓)"
        if metric_disp == "LPC":
            ax.set_title(f"{metric_disp} (×10$^4$) {arrow}", fontsize=14)  # ✅ 제목에 표기
        else:
            ax.set_title(f"{metric_disp} {arrow}", fontsize=14)

        ax.set_xticks(x)
        ax.set_xticklabels(lead_labels, fontsize=10)
        ax.grid(True, linewidth=0.6, alpha=0.4)

        if y_lim is not None and metric_disp != "CSI-R95p":
            ax.set_ylim(*y_lim)

        if not drew_any:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, alpha=0.6)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.02),
        )

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] saved → {out_png}")


#1-day만 출력하는 함수
def print_metric_1day_lead(metrics_by_model):
    """
    metrics_by_model:
        compute_metrics_by_leadtime_from_yamls의 반환값

    모든 모델에 대해 1-day lead metric을 출력
    """

    print("\n📊 1-Day Lead Metrics (All Models)")
    print("=" * 60)

    for model_name, model_metrics in metrics_by_model.items():
        print(f"\n🔹 Model: {model_name}")
        print("-" * 40)

        for metric_name, values in model_metrics.items():
            if len(values) == 0:
                continue

            val_1day = values[0]  # 1-day = index 0
            print(f"{metric_name:15s}: {val_1day:.6f}")

    print("=" * 60)


#======================================================================
# Case study - 확대 영역만 그리는 코드 (4x3)
#======================================================================
# =========================================================
# 0) 유틸: (torch/np) + 다양한 shape를 (T,H,W)로 변환
# =========================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# 0) helpers
# =========================================================
def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_thw(x):
    """
    (T,H,W)로 맞춤
    - 이미 (T,H,W)이면 그대로
    - (H,W,T) 들어오면 자동 교정은 안 함 (필요하면 여기서 추가)
    """
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"_to_thw expects 3D (T,H,W), got {x.shape}")
    return x

def _z_to_mmday(z_np, mu_tp, sig_tp, clamp_zero=True):
    """
    z-score -> m -> mm/day
    mu_tp, sig_tp are in [m]
    """
    x_m = z_np * sig_tp + mu_tp
    if clamp_zero:
        x_m = np.maximum(x_m, 0.0)
    return x_m * 1000.0  # mm/day (단위는 네 데이터 정의에 따름)

def _make_cmap_white_blue_yellow_red():
    colors = ["#ffffff", "#4575b4", "#ffffbf", "#d73027"]
    cm = LinearSegmentedColormap.from_list("white_blue_yellow_red", colors, N=256)
    cm.set_bad("white", 1.0)
    cm.set_under("white", 1.0)
    return cm


# =========================================================
# 1) ZoomGrid (mm/day 변환 + visualize_prediction_map 스타일)
# =========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import torch


# =========================================================
# helpers
# =========================================================
def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_thw(x):
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got {x.shape}")
    return x

def _z_to_mmday(z_thw, mu_tp, sig_tp, clamp_zero=True, mu_sig_unit="m"):
    """
    z-score -> mm/day

    z_thw: (T,H,W)
    mu_tp/sig_tp: 평균/표준편차 (mu_sig_unit에 따라 m 또는 mm 단위)
    mu_sig_unit:
        - "m"  : mu/sig in meters  -> *1000 해서 mm/day로 변환
        - "mm" : mu/sig already in mm -> 그대로 mm/day
    """
    x = z_thw * sig_tp + mu_tp  # still in mu_sig_unit
    if clamp_zero:
        x = np.maximum(x, 0.0)

    if mu_sig_unit == "m":
        return x * 1000.0
    elif mu_sig_unit == "mm":
        return x
    else:
        raise ValueError(f"mu_sig_unit must be 'm' or 'mm', got {mu_sig_unit}")


def _make_cmap_white_blue_yellow_red():
    colors = ["#ffffff", "#4575b4", "#ffffbf", "#d73027"]
    cm = LinearSegmentedColormap.from_list("white_blue_yellow_red", colors, N=256)
    cm.set_bad("white", 1.0)
    cm.set_under("white", 1.0)
    return cm


# =========================================================
# 1) Zoom grid plot (✅ zoom GT 기준 vmin/vmax + 0제외 percentile + vmax_floor)
# =========================================================
def plot_zoom_grid_only_mmday(
    gt, mse, facl, facl_wacl,
    extent=(100, 140, 25, 40),
    t_indices=(0, 1, 2),
    lead_titles=("May 30, 2020 (1-day)", "May 31, 2020 (2-day)", "June 1, 2020 (3-day)"),
    row_titles=("Ground Truth", "MSE", "FACL", "WFCL"),
    lon=None, lat=None,
    central_longitude: float = 0.0,
    add_gridlines: bool = True,
    tick_labelsize: float = 10.0,
    title_fontsize: float = 12.0,
    row_label_fontsize: float = 12.0,
    mu_tp: float  = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    mu_sig_unit: str = "m",
    clamp_zero: bool = True,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    vmin=None, vmax=None,
    vmax_floor: float = None,
    palette: str = "white_blue_yellow_red",
    figsize=(13, 11),
    dpi=300,
    out_png=None,
    b=0, c=0,
    debug: bool = False,
):

    plt.style.use("default")

    if palette == "white_blue_yellow_red":
        cmap_obj = _make_cmap_white_blue_yellow_red()
    elif palette == "precip_jma":
        from matplotlib.colors import LinearSegmentedColormap as _LSC
        _precip_colors = [
            "#ffffff", "#c8f0ff", "#96d2fa", "#64b4f0", "#3296e6",
            "#50c850", "#a0e632", "#f0f000", "#ffc800", "#ff9600",
            "#ff5a00", "#e61400", "#c80000", "#960050", "#7800a0", "#3c0078",
        ]
        cmap_obj = _LSC.from_list("precip_jma", _precip_colors, N=256)
        cmap_obj.set_bad("white", 1.0)
        cmap_obj.set_under("white", 1.0)
    else:
        cmap_obj = plt.get_cmap(palette).copy()
        cmap_obj.set_bad("white", 1.0)
        cmap_obj.set_under("white", 1.0)

    def pick_bc(x):
        x = _to_numpy(x)
        if x.ndim == 5:
            return x[b, c]
        if x.ndim == 4:
            return x[0]
        if x.ndim == 3:
            return x
        raise ValueError(f"Unsupported input shape: {x.shape}")

    gt_z   = _to_thw(pick_bc(gt))
    mse_z  = _to_thw(pick_bc(mse))
    facl_z = _to_thw(pick_bc(facl))
    fw_z   = _to_thw(pick_bc(facl_wacl))

    gt_mm   = _z_to_mmday(gt_z,   mu_tp, sig_tp, clamp_zero=clamp_zero, mu_sig_unit=mu_sig_unit)
    mse_mm  = _z_to_mmday(mse_z,  mu_tp, sig_tp, clamp_zero=clamp_zero, mu_sig_unit=mu_sig_unit)
    facl_mm = _z_to_mmday(facl_z, mu_tp, sig_tp, clamp_zero=clamp_zero, mu_sig_unit=mu_sig_unit)
    fw_mm   = _z_to_mmday(fw_z,   mu_tp, sig_tp, clamp_zero=clamp_zero, mu_sig_unit=mu_sig_unit)

    T, H, W = gt_mm.shape

    if lon is None:
        lon = np.linspace(0.0, 360.0, W, endpoint=False)
    if lat is None:
        lat = np.linspace(-90.0, 90.0, H)

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    full_extent = (lon.min(), lon.max(), lat.min(), lat.max())

    # -------------------
    # vmin/vmax 계산
    # -------------------
    if vmin is None or vmax is None:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)

        gt_zoom = gt_mm[:, lat_mask][:, :, lon_mask]
        tv = gt_zoom[np.isfinite(gt_zoom)]
        tv_pos = tv[tv > 0]
        base = tv_pos if tv_pos.size >= 50 else tv

        vmin = float(np.percentile(base, vmin_percentile))
        vmax = float(np.percentile(base, vmax_percentile))

    if vmax_floor is not None:
        vmax = max(vmax, vmax_floor)

    vmin_plot = np.nextafter(float(vmin), float(vmax))

    proj_axes = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs  = ccrs.PlateCarree()

    def _decorate(ax, show_lon=False, show_lat=False):
        ax.set_extent(extent, crs=data_crs)
        ax.coastlines(resolution="110m", linewidth=0.5, color="dimgray", alpha=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="dimgray", alpha=0.6)

        if add_gridlines:
            gl = ax.gridlines(
                crs=data_crs,
                draw_labels=True,
                linewidth=0.4,
                linestyle="dashed",
                color="gray",
                alpha=0.5,
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.bottom_labels = show_lon
            gl.left_labels = show_lat

            lon_min, lon_max, lat_min, lat_max = extent
            gl.xlocator = mticker.FixedLocator(np.arange(lon_min, lon_max+1, 10))
            gl.ylocator = mticker.FixedLocator(np.arange(lat_min, lat_max+1, 5))
            ax.set_xticks([])
            ax.set_yticks([])

    fig, axes = plt.subplots(
        nrows=4, ncols=3,
        figsize=figsize, dpi=dpi,
        subplot_kw={"projection": proj_axes},
    )

    data_rows = [gt_mm, mse_mm, facl_mm, fw_mm]
    img = None

    for r in range(4):
        for cc in range(3):
            ax = axes[r, cc]
            t = int(t_indices[cc])

            arr = np.ma.masked_invalid(data_rows[r][t])

            img = ax.imshow(
                arr,
                origin="lower",
                extent=full_extent,
                transform=data_crs,
                cmap=cmap_obj,
                vmin=vmin_plot,
                vmax=float(vmax),
            )

            _decorate(ax, show_lon=(r==3), show_lat=(cc==0))

            if r == 0:
                ax.set_title(lead_titles[cc], fontsize=title_fontsize)

    # 🔥 왼쪽 여백 확보
    fig.subplots_adjust(left=0.18)

    # 🔥 정확한 행 중앙 정렬 (핵심 수정 부분)
    for r in range(4):
        ax = axes[r, 0]
        pos = ax.get_position()
        y_center = pos.y0 + pos.height / 2 +0.05

        fig.text(
            pos.x0 - 0.03,
            y_center,
            row_titles[r],
            va="center",
            ha="right",
            fontsize=row_label_fontsize,
        )

    fig.tight_layout(rect=(0.15, 0.08, 1, 1))

    # 컬러바
    cbar = fig.colorbar(
        img,
        ax=axes,
        orientation="horizontal",
        fraction=0.023,
        pad=0.06,
        shrink=1.0,
        aspect=40
    )

    cbar.set_label("Precipitation (mm/day)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] saved → {out_png}")
    else:
        plt.show()




# =========================================================
# 2) case study: raw_idx -> case_idx 매핑 후 loader에서 샘플 1개 뽑아 예측 + ZoomGrid(mm/day) 저장
# =========================================================
@torch.no_grad()
def run_case_study_zoomgrid_by_raw_index(
    test_loader,
    raw_idx: int,
    loss_specs,
    device,
    vis_dir: str,
    exp: str,
    base_time: str,
    epoch: int,
    extent_zoom=(100, 140, 25, 40),
    t_indices=(0, 1, 2),
    lead_titles=("May 30, 2020 (1-day)", "May 31, 2020 (2-day)", "June 1, 2020 (3-day)"),
    row_titles=("Ground Truth", "MSE", "FACL", "WFCL"),
    mu_tp: float  = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    mu_sig_unit: str = "m",          # ✅ 추가
    clamp_zero: bool = True,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    vmax_floor: float = None,        # ✅ 추가 (예: 20.0)
    central_longitude: float = 0.0,
    add_gridlines: bool = True,
    palette: str = "white_blue_yellow_red",
    figsize=(13, 11),
    dpi=300,
    debug: bool = False,             # ✅ 추가
):
    from utils.config_utils import load_config
    from utils.save_utils import load_checkpoint_for_eval
    from models.get_model import build_model

    os.makedirs(vis_dir, exist_ok=True)

    dataset = test_loader.dataset
    if not hasattr(dataset, "idx"):
        raise AttributeError("dataset has no attribute 'idx' (raw index mapping not available).")

    idx_arr = np.asarray(dataset.idx).astype(int)
    hit = np.where(idx_arr == int(raw_idx))[0]
    if hit.size == 0:
        raise ValueError(f"raw_idx={raw_idx} is NOT included in this test dataset split (len={len(idx_arr)}).")
    case_idx = int(hit[0])

    def _get_one_sample_from_loader(loader, target_idx: int):
        processed = 0
        for batch in loader:
            x1b = batch["X1"].to(device).float()
            x2b = batch["X2"].to(device).float()
            yb  = batch["Y"].to(device).float()
            bs = x1b.size(0)
            if processed + bs > target_idx:
                i = target_idx - processed
                return x1b[i:i+1], x2b[i:i+1], yb[i:i+1]
            processed += bs
        raise IndexError(f"target_idx={target_idx} exceeds loader size")

    x1, x2, gt = _get_one_sample_from_loader(test_loader, case_idx)

    # ✅ dataset에서 lon/lat 자동 탐색
    lon = None
    lat = None
    for lon_key in ("lon", "lons", "longitude", "longitudes"):
        if hasattr(dataset, lon_key):
            lon = np.asarray(getattr(dataset, lon_key))
            break
    for lat_key in ("lat", "lats", "latitude", "latitudes"):
        if hasattr(dataset, lat_key):
            lat = np.asarray(getattr(dataset, lat_key))
            break
    if lon is None or lat is None:
        raise AttributeError(
            "dataset에서 lon/lat을 찾지 못했습니다. dataset.lon/dataset.lat (또는 유사 필드)를 확인해 주세요."
        )

    def _load_model_from_spec(spec):
        cfg_m = load_config(spec["cfg"])
        m = build_model(cfg_m).to(device).float()
        load_checkpoint_for_eval(spec["ckpt"], m, device=device)
        m.eval()
        return m

    preds_by_name = {}
    for spec in loss_specs:
        m = _load_model_from_spec(spec)
        preds_by_name[str(spec["name"])] = m(x1, x2)

    key_mse  = "MSE"
    key_facl = "FACL"
    key_fwcl = "FWCL Lv3"

    pred_mse  = preds_by_name[key_mse]
    pred_facl = preds_by_name[key_facl]
    pred_fwcl = preds_by_name[key_fwcl]

    save_path = os.path.join(
        vis_dir,
        f"ZoomGrid_CASE_{exp}_{base_time}_ep{epoch}_raw{raw_idx}_ds{case_idx}_mmday.png"
    )

    plot_zoom_grid_only_mmday(
        gt=gt, mse=pred_mse, facl=pred_facl, facl_wacl=pred_fwcl,
        extent=extent_zoom,
        t_indices=t_indices,
        lead_titles=lead_titles,
        row_titles=row_titles,
        lon=lon, lat=lat,
        central_longitude=central_longitude,
        add_gridlines=add_gridlines,
        palette=palette,
        figsize=figsize,
        dpi=dpi,
        out_png=save_path,
        b=0, c=0,
        mu_tp=mu_tp, sig_tp=sig_tp,
        mu_sig_unit=mu_sig_unit,       
        clamp_zero=clamp_zero,
        vmin_percentile=vmin_percentile,
        vmax_percentile=vmax_percentile,
        vmax_floor=vmax_floor,         # ✅ 전달
        debug=debug,                   # ✅ 전달
    )

    return save_path, case_idx






#=======================================================================
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
# SNR Comparison
# =========================

def visualize_snr_comparison(
    preds,
    targets,
    save_path: str,
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    lead_labels=("t+1", "t+2", "t+3"),
    title_suffix: str = "",
    central_longitude: float = 180.0,
    dpi: int = 300,
    snr_vmin: float = -10.0,
    snr_vmax: float = 20.0,
    eps: float = 1e-6,
):
    """
    전체 테스트셋 예측·정답을 mm/day 단위로 변환한 뒤, 리드타임별로
    - 상단: 픽셀별 SNR 지도 (dB) — SNR(x,y) = 10·log10(Var(GT) / Var(GT - Pred))
    - 하단: 전구 평균 SNR / Signal Power / Noise Power 리드타임 비교 차트

    Args
    ----
    preds, targets : list of (B,1,T,H,W) tensors in z-score space
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import torch

    # ── 1) 전체 배치 합산 ────────────────────────────────────────────────
    pred_all = torch.cat(preds,   dim=0).float()   # (N,1,T,H,W) or (N,T,H,W)
    gt_all   = torch.cat(targets, dim=0).float()

    if pred_all.ndim == 5:
        pred_all = pred_all[:, 0]   # (N,T,H,W)
        gt_all   = gt_all[:,   0]

    N, T, H, W = gt_all.shape
    n_leads = min(T, len(lead_labels))

    # ── 2) mm/day 변환 (clamping 없이 그대로) ───────────────────────────
    def _to_mm(z):
        return (z.numpy() * sig_tp + mu_tp) * 1000.0

    pred_mm = _to_mm(pred_all)   # (N,T,H,W)
    gt_mm   = _to_mm(gt_all)

    error_mm = pred_mm - gt_mm   # (N,T,H,W)

    # ── 3) 픽셀별 분산 계산 ──────────────────────────────────────────────
    # signal_var : GT의 공간 분산 (N 방향)
    # noise_var  : 오차의 공간 분산
    signal_var = gt_mm.var(axis=0)    # (T,H,W)
    noise_var  = error_mm.var(axis=0) # (T,H,W)

    snr_map = 10.0 * np.log10((signal_var + eps) / (noise_var + eps))   # (T,H,W)  dB

    # 전구 평균 SNR / signal power / noise power
    mean_snr    = snr_map[:n_leads].mean(axis=(1, 2))                      # (n_leads,)
    mean_sigpow = 10.0 * np.log10(signal_var[:n_leads].mean(axis=(1, 2)) + eps)
    mean_noipow = 10.0 * np.log10(noise_var[:n_leads].mean(axis=(1, 2))  + eps)

    # ── 4) Figure 레이아웃: 상단 SNR 지도 × n_leads + 하단 차트 ─────────
    proj  = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree()
    lon = np.linspace(0.0, 360.0, W, endpoint=False)
    lat = np.linspace(-90.0, 90.0, H)

    fig = plt.figure(figsize=(5.5 * n_leads, 8), dpi=dpi, facecolor="white")
    gs  = fig.add_gridspec(2, n_leads, height_ratios=[2.5, 1.0], hspace=0.45, wspace=0.15)

    cmap_snr = plt.get_cmap("RdYlGn")
    norm_snr = mcolors.Normalize(vmin=snr_vmin, vmax=snr_vmax)

    for i in range(n_leads):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="dimgray", alpha=0.7)
        ax.gridlines(linewidth=0.3, linestyle="--", color="gray", alpha=0.4)

        im = ax.pcolormesh(lon, lat, snr_map[i],
                           transform=data_crs,
                           cmap=cmap_snr, norm=norm_snr, rasterized=True)
        ax.set_title(f"SNR map  {lead_labels[i]}\n"
                     f"mean={mean_snr[i]:.2f} dB",
                     fontsize=9)

        cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                            pad=0.04, fraction=0.046, aspect=30)
        cbar.set_label("SNR (dB)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    # ── 5) 하단 차트: mean SNR / Signal / Noise vs lead time ─────────────
    ax_bot = fig.add_subplot(gs[1, :])
    x = np.arange(n_leads)
    w = 0.25

    ax_bot.bar(x - w,  mean_sigpow, width=w, label="Signal power (dB)",
               color="#2196F3", alpha=0.85, edgecolor="white")
    ax_bot.bar(x,      mean_noipow, width=w, label="Noise power (dB)",
               color="#F44336", alpha=0.85, edgecolor="white")
    ax_bot.bar(x + w,  mean_snr,    width=w, label="SNR (dB)",
               color="#4CAF50", alpha=0.85, edgecolor="white")

    for j, (s, n, r) in enumerate(zip(mean_sigpow, mean_noipow, mean_snr)):
        ax_bot.text(j - w, s + 0.3, f"{s:.1f}", ha="center", va="bottom", fontsize=7)
        ax_bot.text(j,     n + 0.3, f"{n:.1f}", ha="center", va="bottom", fontsize=7)
        ax_bot.text(j + w, r + 0.3, f"{r:.1f}", ha="center", va="bottom", fontsize=7)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(lead_labels, fontsize=9)
    ax_bot.set_ylabel("dB", fontsize=9)
    ax_bot.set_title("Global Mean Signal / Noise / SNR by Lead Time", fontsize=9)
    ax_bot.legend(fontsize=8, loc="upper right")
    ax_bot.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_bot.grid(axis="y", linewidth=0.4, alpha=0.5)

    if title_suffix:
        fig.suptitle(f"Signal-to-Noise Ratio  {title_suffix}", fontsize=11, y=1.01)

    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================
# Frequency-wise SNR Comparison
# =========================

def _compute_radial_freq_snr(preds, targets, mu_tp, sig_tp, n_leads, n_radial_bins, eps):
    """
    여러 배치의 preds/targets로부터 반경 평균 frequency-wise SNR을 계산한다.

    Returns
    -------
    k_centers  : (n_radial_bins,)            radial frequency (cycles/degree)
    snr_1d     : (n_leads, n_radial_bins)    SNR in dB
    gt_psd_1d  : (n_leads, n_radial_bins)    GT power spectrum in dB
    pred_psd_1d: (n_leads, n_radial_bins)    Pred power spectrum in dB
    noi_1d     : (n_leads, n_radial_bins)    noise (error) power spectrum in dB
    """
    import numpy as np
    import torch

    pred_all = torch.cat(preds,   dim=0).float()
    gt_all   = torch.cat(targets, dim=0).float()

    if pred_all.ndim == 5:
        pred_all = pred_all[:, 0]
        gt_all   = gt_all[:,   0]

    N, T, H, W = gt_all.shape
    n_leads = min(T, n_leads)

    def _to_mm(z):
        return (z.numpy() * sig_tp + mu_tp) * 1000.0

    pred_mm  = _to_mm(pred_all)
    gt_mm    = _to_mm(gt_all)
    error_mm = pred_mm - gt_mm

    # 2D FFT 파워 스펙트럼 (샘플 평균)
    gt_pow2d   = np.zeros((n_leads, H, W))  # E[|FFT(GT)|²]
    pred_pow2d = np.zeros((n_leads, H, W))  # E[|FFT(Pred)|²]
    noi_pow2d  = np.zeros((n_leads, H, W))  # E[|FFT(GT-Pred)|²]
    for t in range(n_leads):
        for n in range(N):
            gt_pow2d[t]   += np.abs(np.fft.fftshift(np.fft.fft2(gt_mm[n, t])))**2
            pred_pow2d[t] += np.abs(np.fft.fftshift(np.fft.fft2(pred_mm[n, t])))**2
            noi_pow2d[t]  += np.abs(np.fft.fftshift(np.fft.fft2(error_mm[n, t])))**2
        gt_pow2d[t]   /= N
        pred_pow2d[t] /= N
        noi_pow2d[t]  /= N

    # 반경 주파수 좌표
    ky = np.fft.fftshift(np.fft.fftfreq(H))
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    KX, KY = np.meshgrid(kx, ky)
    K_radius = np.sqrt(KX**2 + KY**2)

    k_max = K_radius.max()
    k_edges   = np.linspace(0.0, k_max, n_radial_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    snr_1d      = np.zeros((n_leads, n_radial_bins))
    gt_psd_1d   = np.zeros((n_leads, n_radial_bins))
    pred_psd_1d = np.zeros((n_leads, n_radial_bins))
    noi_1d      = np.zeros((n_leads, n_radial_bins))

    for t in range(n_leads):
        for b in range(n_radial_bins):
            mask = (K_radius >= k_edges[b]) & (K_radius < k_edges[b + 1])
            if mask.sum() == 0:
                continue
            g  = gt_pow2d[t][mask].mean()
            p  = pred_pow2d[t][mask].mean()
            nv = noi_pow2d[t][mask].mean()
            gt_psd_1d[t, b]   = 10.0 * np.log10(g  + eps)
            pred_psd_1d[t, b] = 10.0 * np.log10(p  + eps)
            noi_1d[t, b]      = 10.0 * np.log10(nv + eps)
            snr_1d[t, b]      = 10.0 * np.log10((g  + eps) / (nv + eps))

    return k_centers, snr_1d, gt_psd_1d, pred_psd_1d, noi_1d


def visualize_freq_snr_comparison(
    models_data,
    save_path: str,
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    lead_labels=("t+1", "t+2", "t+3"),
    n_radial_bins: int = 60,
    title_suffix: str = "",
    dpi: int = 150,
    eps: float = 1e-6,
):
    """
    여러 모델의 Frequency-wise SNR + GT/Pred 파워 스펙트럼을 리드타임별로 비교한다.

    레이아웃 (리드타임별 열):
      Row 1 — Frequency-wise SNR 곡선 (모델별 오버레이)
               SNR(k) = 10·log10( E[|FFT(GT)|²] / E[|FFT(GT−Pred)|²] )
      Row 2 — Power Spectral Density 비교
               GT PSD (검정 실선, 레퍼런스) + 각 모델 Pred PSD (컬러 실선)
               + 각 모델 Noise PSD (컬러 점선)
               → GT와 Pred PSD가 얼마나 일치하는지(spectral fidelity) 직접 확인

    Parameters
    ----------
    models_data : list of (preds_list, targets_list, label)
        preds_list   : list of (B,1,T,H,W) tensors (z-score 공간)
        targets_list : list of (B,1,T,H,W) tensors (z-score 공간)
        label        : str  (범례에 표시될 모델 이름)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_leads = len(lead_labels)
    colors  = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    # ── 각 모델 SNR / PSD 계산 ──────────────────────────────────────────
    all_results = []
    for preds, targets, label in models_data:
        k_centers, snr_1d, gt_psd_1d, pred_psd_1d, noi_1d = _compute_radial_freq_snr(
            preds, targets, mu_tp, sig_tp, n_leads, n_radial_bins, eps
        )
        all_results.append((k_centers, snr_1d, gt_psd_1d, pred_psd_1d, noi_1d, label))

    # GT PSD는 모든 모델에서 동일 → 첫 모델에서 추출
    k_ref     = all_results[0][0]
    gt_psd_ref = all_results[0][2]   # (n_leads, n_radial_bins)

    # ── Figure 레이아웃 ───────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, n_leads,
        figsize=(11.0 * n_leads, 9),
        dpi=dpi,
        facecolor="white",
        gridspec_kw={"height_ratios": [1.2, 1.2], "hspace": 0.50, "wspace": 0.30},
    )
    if n_leads == 1:
        axes = axes.reshape(2, 1)

    for i, lead in enumerate(lead_labels):
        ax_snr = axes[0, i]
        ax_psd = axes[1, i]

        # ── Row 1: SNR 곡선 ─────────────────────────────────────────
        for j, (k_centers, snr_1d, _, _, _, label) in enumerate(all_results):
            col = colors[j % len(colors)]
            ax_snr.plot(k_centers, snr_1d[i], color=col, linewidth=1.8, label=label)

        ax_snr.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.8,
                       label="0 dB (signal = noise)")
        ax_snr.fill_between(
            k_ref,
            np.min([r[1][i] for r in all_results], axis=0),
            0,
            where=np.all([r[1][i] < 0 for r in all_results], axis=0),
            color="red", alpha=0.07,
        )
        ax_snr.set_xlabel("Radial frequency k (cycles/degree)", fontsize=9)
        ax_snr.set_ylabel("SNR (dB)", fontsize=9)
        ax_snr.set_title(f"Frequency-wise SNR  [{lead}]", fontsize=10, fontweight="bold")
        ax_snr.legend(fontsize=7, loc="upper right")
        ax_snr.grid(linewidth=0.4, alpha=0.5)
        ax_snr.tick_params(labelsize=7)

        # ── Row 2: PSD 비교 (GT + Pred + Noise) ────────────────────
        # GT 레퍼런스 (검정 굵은 실선)
        ax_psd.plot(k_ref, gt_psd_ref[i], color="black", linewidth=2.2,
                    linestyle="-", label="Ground Truth", zorder=10)

        for j, (k_centers, _, _, pred_psd_1d, noi_1d, label) in enumerate(all_results):
            col = colors[j % len(colors)]
            ax_psd.plot(k_centers, pred_psd_1d[i], color=col, linewidth=1.5,
                        linestyle="-",  label=label, alpha=0.85)

        ax_psd.set_xlabel("Radial frequency k (cycles/degree)", fontsize=9)
        ax_psd.set_ylabel("Power Spectral Density (dB)", fontsize=9)
        ax_psd.set_title("(e) Radial Profiles of the Power Spectral Density", fontsize=10, loc="left")
        ax_psd.legend(fontsize=7, loc="upper right")
        ax_psd.grid(linewidth=0.4, alpha=0.5)
        ax_psd.tick_params(labelsize=7)

    suptitle = "Frequency-wise SNR & Spectral Density Comparison"
    if title_suffix:
        suptitle += f"  {title_suffix}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)

    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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

#===========MSE, FACL, FACL+WACL Lv3 amplitude ratio plot========
# ============================================================
# Simple & Clean: One-Plot Radial Amplitude (3 models by ckpt)
# - Load val dataset once from a single cfg
# - Pick sample by index (default 0)
# - Run 3 models (MSE / FACL / FWACL Lv3) by their ckpt paths
# - Plot amplitude ratio radial profiles in one figure
# ============================================================
import os
import importlib
from typing import Any, Dict, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

# ==========================================================
# YAML 로더 (경로/환경변수 확장)
# ==========================================================
def _load_cfg(path_or_cfg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(path_or_cfg, dict):
        cfg = path_or_cfg
    else:
        with open(os.path.expanduser(path_or_cfg), "r") as f:
            cfg = yaml.safe_load(f) or {}

    # 문자열은 ~, $HOME 등 확장
    def _expand(v):
        if isinstance(v, str):
            return os.path.expandvars(os.path.expanduser(v))
        return v

    # paths 섹션은 전체 확장
    if isinstance(cfg.get("paths"), dict):
        cfg["paths"] = {k: _expand(v) for k, v in cfg["paths"].items()}
    return cfg

# ==========================================================
# 동적 import 유틸
# ==========================================================
def _import_obj(module: str, name: str):
    mod = importlib.import_module(module)
    try:
        return getattr(mod, name)
    except AttributeError:
        raise AttributeError(f"[import] '{module}'에 '{name}'(이)가 없습니다.")

# ==========================================================
# Dataset 빌더 (파일 내 자급자족)
# ==========================================================
def _resolve_dataset_spec(dcfg: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    data 섹션에서 (module, class, args) 추출.
    허용 스키마:
      1) data.module, data.class, data.val.args
      2) data.dataset.module, data.dataset.class, data.dataset.args
      3) data.target: 'pkg.mod:Class'
      + 평평한 키(tp_file 등)는 args로 흡수
    """
    mod = dcfg.get("module")
    cls = dcfg.get("class")
    args = ((dcfg.get("val") or {}).get("args") or {}).copy()

    if not (mod and cls):
        ds = dcfg.get("dataset") or {}
        mod = mod or ds.get("module")
        cls = cls or ds.get("class")
        if not args:
            args = (ds.get("args") or {}).copy()

    if not (mod and cls) and isinstance(dcfg.get("target"), str) and ":" in dcfg["target"]:
        mod, cls = dcfg["target"].split(":", 1)

    # 평평한 키 흡수 (module/class/target/val/dataset 제외)
    if not args:
        exclude = {"module", "class", "target", "val", "dataset"}
        args = {k: v for k, v in dcfg.items() if k not in exclude}

    return mod, cls, args

def _build_valset(cfg_or_path: Union[str, Dict[str, Any]]):
    cfg = _load_cfg(cfg_or_path)
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise KeyError("[build] cfg['data'] 섹션이 없습니다.")
    dcfg = cfg["data"]

    module, clsname, args = _resolve_dataset_spec(dcfg)
    if not module or not clsname:
        raise KeyError(f"[build] data.module/class 또는 data.target이 필요합니다. data keys={list(dcfg.keys())}")

    # 편의: paths.tp_file 자동 주입 (없을 때만)
    paths = cfg.get("paths") or {}
    if "tp_file" not in args and "tp_file" in paths:
        args["tp_file"] = paths["tp_file"]

    # 기본 split
    args.setdefault("split", "val")

    DatasetCls = _import_obj(module, clsname)
    try:
        return DatasetCls(**args)
    except TypeError as e:
        raise TypeError(f"[build] {module}.{clsname}(**args) 인자 오류: {e}\nargs={args}")

# ==========================================================
# Model 로더 (파일 내 자급자족)
# ==========================================================
def _pick_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("state_dict", "model_state", "model_state_dict", "model", "net", "state"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # torch.save(model.state_dict()) 형태
    if isinstance(ckpt, dict) and all(not isinstance(v, dict) for v in ckpt.values()):
        return ckpt
    raise KeyError("[ckpt] state_dict 키를 찾지 못했습니다.")

def _strip_module_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):] : v for k, v in sd.items()}
    return sd

def _map_device(device: Union[str, torch.device, None]):
    if device is None:
        return None
    if isinstance(device, str):
        d = device.lower()
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda"):
            return torch.device(d if torch.cuda.is_available() else "cpu")
    return device

def _load_model(cfg_or_path: Union[str, Dict[str, Any]],
                ckpt_path: Union[str, None],
                device: str = "cuda",
                *,
                model_section="model",
                module_key="module",
                class_key="class",
                init_key="init_args",
                strict: bool = False) -> torch.nn.Module:
    cfg = _load_cfg(cfg_or_path)

    # 1) ckpt 경로 결정
    if ckpt_path is None:
        paths = cfg.get("paths", {}) or {}
        mcfg  = cfg.get(model_section, {}) or {}
        ckpt_path = paths.get("ckpt") or paths.get("ckpt_path") \
                    or mcfg.get("ckpt") or mcfg.get("ckpt_path")
        if ckpt_path is None:
            raise ValueError("[load_model] ckpt_path 인자 또는 cfg.paths.ckpt/model.ckpt가 필요합니다.")
    ckpt_path = os.path.expanduser(os.path.expandvars(ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[load_model] 체크포인트 없음: {ckpt_path}")

    # 2) 모델 클래스 import & 인스턴스 생성
    mcfg = cfg.get(model_section, {}) or {}
    module = mcfg.get(module_key)
    cls    = mcfg.get(class_key)
    init_args = mcfg.get(init_key, {}) or {}
    if not module or not cls:
        raise KeyError(f"[load_model] cfg['{model_section}']에 '{module_key}'와 '{class_key}'가 필요합니다.")

    ModelCls = _import_obj(module, cls)
    model = ModelCls(**init_args)

    # 3) state_dict 로드
    map_loc = _map_device(device)
    ckpt = torch.load(ckpt_path, map_location=map_loc)
    sd = _strip_module_prefix(_pick_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if not strict and (missing or unexpected):
        print(f"[load_model] missing={missing}, unexpected={unexpected}")

    if isinstance(map_loc, torch.device):
        model.to(map_loc)
    model.eval()
    return model

# ==========================================================
# FFT / 주파수 헬퍼
# ==========================================================
def _fft2_shifted(img2d: np.ndarray):
    return np.fft.fftshift(np.fft.fft2(img2d))

def _fftfreq_centers(N: int, d: float, unit: str):
    f = np.fft.fftfreq(N, d=d)  # cycles/unit
    if unit == "rad":
        f = 2*np.pi*f           # radians/unit
    return np.fft.fftshift(f)

def _freq_axes(H: int, W: int, unit: str = "cyc", dy: float = 1.0, dx: float = 1.0):
    fy = _fftfreq_centers(H, dy, unit)
    fx = _fftfreq_centers(W, dx, unit)
    return fx, fy

def _radial_bins(H: int, W: int, nbins: int = 128):
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, r.max() + 1e-9, nbins + 1)
    bins = np.digitize(r.ravel(), edges) - 1
    return r, bins, edges

def _radial_stat(a2d: np.ndarray, bins: np.ndarray, nbins: int, reducer="median", mask=None):
    a = a2d.ravel()
    m = np.ones_like(a, dtype=bool) if mask is None else mask.ravel().astype(bool)
    out = np.full(nbins, np.nan, dtype=np.float64)
    for k in range(nbins):
        idx = (bins == k) & m
        if np.any(idx):
            out[k] = np.nanmedian(a[idx]) if reducer == "median" else np.nanmean(a[idx])
    return out

# ==========================================================
# 샘플 언팩
# ==========================================================
def _unpack_sample(sample, device="cuda"):
    """
    Dataset 샘플에서 (x, y, base_time:str)를 추출.
    허용 형태:
      - dict: {'x'/'inputs', 'y'/'targets', 'base_time'?}
      - tuple/list: (x, y[, meta{'base_time': ...}])
    """
    base_time = ""
    if isinstance(sample, dict):
        x = sample.get("x", sample.get("inputs"))
        y = sample.get("y", sample.get("targets"))
        bt = sample.get("base_time", None)
        base_time = str(bt) if bt is not None else ""
    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
        x, y = sample[0], sample[1]
        if len(sample) >= 3 and isinstance(sample[2], dict):
            bt = sample[2].get("base_time", None)
            base_time = str(bt) if bt is not None else ""
    else:
        raise ValueError("sample 형식을 파악할 수 없습니다. dict 또는 (x,y,meta) 튜플을 기대합니다.")
    return x.to(device), y.to(device), base_time

# ==========================================================
# 3모델 Amplitude ratio 한 그래프 (선형 ratio만)
# ==========================================================
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================
# Amplitude-ratio 비교(한 그래프, 3개 모델) - Self-contained
# ============================================

import os
import yaml
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# -------------------------
# FFT / freq helpers
# -------------------------
def _fft2_shifted(img2d: np.ndarray):
    return np.fft.fftshift(np.fft.fft2(img2d))

def _fftfreq_centers(N: int, d: float, unit: str):
    f = np.fft.fftfreq(N, d=d)  # cycles/unit
    if unit == "rad":
        f = 2*np.pi*f           # radians/unit
    return np.fft.fftshift(f)

def _freq_axes(H: int, W: int, unit: str = "cyc", dy: float = 1.0, dx: float = 1.0):
    fy = _fftfreq_centers(H, dy, unit)
    fx = _fftfreq_centers(W, dx, unit)
    return fx, fy

def _radial_bins(H: int, W: int, nbins: int = 128):
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0, r.max() + 1e-9, nbins + 1)
    bins = np.digitize(r.ravel(), edges) - 1
    return r, bins, edges

def _radial_stat(a2d: np.ndarray, bins: np.ndarray, nbins: int, reducer="median", mask=None):
    a = a2d.ravel()
    m = np.ones_like(a, dtype=bool) if mask is None else mask.ravel().astype(bool)
    out = np.full(nbins, np.nan, dtype=np.float64)
    for k in range(nbins):
        idx = (bins == k) & m
        if np.any(idx):
            out[k] = np.nanmedian(a[idx]) if reducer == "median" else np.nanmean(a[idx])
    return out

def _amp_transform(ratio: np.ndarray, mode: str = "db", eps: float = 1e-8):
    if mode == "db":
        return 20.0 * np.log10(np.maximum(ratio, eps))
    if mode == "pct":
        return 100.0 * (ratio - 1.0)
    if mode == "linear":
        return ratio
    raise ValueError("amp_mode must be 'db'|'pct'|'linear'")

# -------------------------
# 프로젝트 어댑터(최소 의존)
# -------------------------
def _build_valset_from_cfg(cfg_path: str):
    """
    utils.build.build_val_dataset_from_cfg(cfg) 를 호출.
    - cfg['data']에 우리가 이전에 합의했던 스키마(target/module/class/args 등)를 사용.
    """
    from utils import load_config  # utils/config_utils.py
    cfg = load_config(cfg_path)
    from utils.build import build_val_dataset_from_cfg
    return build_val_dataset_from_cfg(cfg)

def _load_model_from_ckpt(cfg_path: str, ckpt_path: str, device: str = "cuda"):
    """
    utils.config_utils.load_model_from_ckpt 가 무엇을 반환하든
    (model) / (model, ...) / {"model": model, ...} 모두 안전하게 모델만 뽑아오는 래퍼.
    """
    from utils import load_config
    cfg = load_config(cfg_path)

    # 원래 로더 가져오기
    from utils.config_utils import load_model_from_ckpt as _inner_load

    obj = _inner_load(cfg, ckpt_path, device=device)

    # 다양한 반환형 대응
    model = None
    if isinstance(obj, tuple):
        # (model, something, ...) 형태
        model = obj[0]
    elif isinstance(obj, dict):
        # {"model": model, ...} 형태
        model = obj.get("model", None)
        if model is None:
            # 혹시 다른 키로 들어왔을 때 첫 번째 텐서화 가능한 항목을 선택
            for v in obj.values():
                if hasattr(v, "state_dict") or hasattr(v, "forward"):
                    model = v
                    break
    else:
        # 순수 모델
        model = obj

    if model is None:
        raise TypeError("[_load_model_from_ckpt] 모델 객체를 찾지 못했습니다. 반환값 형식을 확인하세요.")

    # 디바이스/평가 모드 정리
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return model


def _unpack_sample(sample, device="cuda"):
    """
    Dataset 샘플에서 (x, y, base_time:str) 추출.
    (dict 또는 (x,y,meta) 형태 지원)
    """
    base_time = ""
    if isinstance(sample, dict):
        x = sample.get("x", sample.get("inputs"))
        y = sample.get("y", sample.get("targets"))
        bt = sample.get("base_time", None)
        base_time = str(bt) if bt is not None else ""
    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
        x, y = sample[0], sample[1]
        if len(sample) >= 3 and isinstance(sample[2], dict):
            bt = sample[2].get("base_time", None)
            base_time = str(bt) if bt is not None else ""
    else:
        raise ValueError("sample 형식을 파악할 수 없습니다.")
    return x.to(device), y.to(device), base_time

# -------------------------
# 텐서 모양/입력 구성 헬퍼
# -------------------------
def _ensure_BCTHW(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 5:  # (B,C,T,H,W)
        return t
    if t.dim() == 4:  # (C,T,H,W)
        return t.unsqueeze(0)
    if t.dim() == 3:  # (T,H,W)
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 2:  # (H,W)
        return t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported tensor dim={t.dim()}, shape={tuple(t.shape)}")

def _normalize_inputs_to_BCTHW(sample_x, device="cuda"):
    """
    샘플 x를 (x,), (x1,x2) 등으로 표준화.
    """
    # dict
    if isinstance(sample_x, dict):
        x = (sample_x.get("x") or sample_x.get("inputs")
             or sample_x.get("x1") or sample_x.get("tp"))
        x2 = sample_x.get("x2")
        if torch.is_tensor(x):
            x = _ensure_BCTHW(x.to(device))
            if torch.is_tensor(x2):
                x2 = _ensure_BCTHW(x2.to(device))
                return (x, x2)
            return (x,)
        tens = [v for v in sample_x.values() if torch.is_tensor(v)]
        if len(tens) >= 2:
            return (_ensure_BCTHW(tens[0].to(device)),
                    _ensure_BCTHW(tens[1].to(device)))
        if len(tens) == 1:
            return (_ensure_BCTHW(tens[0].to(device)),)

    # tuple/list
    if isinstance(sample_x, (list, tuple)):
        tens = [t for t in sample_x if torch.is_tensor(t)]
        if len(tens) >= 2:
            return (_ensure_BCTHW(tens[0].to(device)),
                    _ensure_BCTHW(tens[1].to(device)))
        if len(tens) == 1:
            return (_ensure_BCTHW(tens[0].to(device)),)

    # 텐서
    if torch.is_tensor(sample_x):
        return (_ensure_BCTHW(sample_x.to(device)),)

    raise ValueError(f"[inputs] Unsupported input type: {type(sample_x)}")

def _force_two_inputs(inps):
    """
    inps가 1개면 → (x1,x2)로 확장:
      - C>=2이면 채널 분리 (첫 채널/둘째 채널)
      - C==1이면 zeros_like로 x2 생성
    """
    if len(inps) >= 2:
        return inps[0], inps[1]
    x1 = inps[0]
    if x1.dim() != 5:
        x1 = _ensure_BCTHW(x1)
    B, C, T, H, W = x1.shape
    if C >= 2:
        return x1[:, :1], x1[:, 1:2]
    else:
        return x1, torch.zeros_like(x1)

def _call_model_robust(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor):
    """
    먼저 (x1,x2)로 호출 → 실패(TypeError) 시 (x1,)로 재시도.
    (forward가 x1,x2 구조인 Geo_UNet 대응)
    """
    try:
        return model(x1, x2)
    except TypeError:
        return model(x1)

# -------------------------
# 핵심: 한 그래프에 3모델 amp-ratio
# -------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

# =========================
# Helpers (FFT, freq axes, radial bins, stats, etc.)
# =========================

# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
# evaluate_3d.py 상단 import에 추가
from omegaconf import OmegaConf
try:
    from omegaconf import DictConfig
except Exception:
    DictConfig = None

# ---- Robust model builder import (supports multiple repo variants) ----
import importlib

def _resolve_build_model():
    candidates = [
        ("models.get_model", ("get_model", "build_model", "make_model")),
        ("models.model",     ("get_model", "build_model", "make_model")),
    ]
    for module_path, names in candidates:
        try:
            m = importlib.import_module(module_path)
        except Exception:
            continue
        for name in names:
            if hasattr(m, name):
                return getattr(m, name)
    raise ImportError(
        "Could not find a model builder. "
        "Tried: models.get_model.{get_model,build_model,make_model} and "
        "models.model.{get_model,build_model,make_model}."
    )

build_model = _resolve_build_model()
# evaluate_3d.py 상단 import 근처에 추가
# evaluate_3d.py 내 load_val_sample 정의를 아래처럼 수정
from utils.build import build_val_dataset_from_cfg
from torch.utils.data import DataLoader
import torch

def load_val_sample(
    cfg,
    sample_idx: int = 0,
    batch_size: int = 1,
    # 필요 시 키를 직접 지정해 override 가능
    x1_keys=None, x2_keys=None, y_keys=None,
):
    """
    검증셋에서 sample_idx번째 배치를 (x1, x2, y, base_times)로 반환.
    x2가 없는 단일 입력 모델이면 x2=None.
    """
    # 0) DictConfig → dict
    if DictConfig is not None and isinstance(cfg, DictConfig):
        cfg_in = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, (str, dict)):
        cfg_in = cfg
    else:
        raise TypeError(f"[load_val_sample] cfg must be DictConfig/str/dict, got {type(cfg)}")

    # 1) dataset/loader
    val_ds = build_val_dataset_from_cfg(cfg_in)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2) sample_idx 배치 가져오기
    batch = None
    for i, b in enumerate(val_dl):
        if i == sample_idx:
            batch = b
            break
    if batch is None:
        raise IndexError(f"[load_val_sample] sample_idx={sample_idx} out of range (len={i+1 if 'i' in locals() else 0})")

    # 3) 키 후보
    if x1_keys is None:
        x1_keys = ("x1", "x", "inputs", "data", "input", "image", "images", "feat", "features")
    if x2_keys is None:
        # 습도/보조변수 등 2번째 입력의 흔한 키
        x2_keys = ("x2", "aux", "qavg", "q", "humidity", "humid", "inputs2", "feat2", "features2")
    if y_keys is None:
        y_keys  = ("y", "targets", "target", "label", "labels", "gt", "truth")

    def _extract_from_dict(d: dict):
        x1 = next((d[k] for k in x1_keys if k in d), None)
        x2 = next((d[k] for k in x2_keys if k in d), None)
        y  = next((d[k] for k in y_keys  if k in d), None)
        bt = d.get("base_times", d.get("base_time", d.get("times", d.get("time", ""))))
        # 백업: 텐서만 추려서 자동 유추 (x1=첫번째, x2=두번째, y=세번째)
        if x1 is None or (x2 is None and y is None):
            tensors = [v for v in d.values() if isinstance(v, torch.Tensor)]
            if x1 is None and len(tensors) >= 1: x1 = tensors[0]
            if x2 is None and len(tensors) >= 2: x2 = tensors[1]
            if y  is None and len(tensors) >= 3: y  = tensors[2]
        return x1, x2, y, bt, batch

    def _extract_from_seq(seq):
        """
        허용 케이스 예:
        - (x1, y)
        - (x1, x2, y)
        - (x1, {meta})
        - ({dict 단일})
        - (x1, (y, meta))  ← 드물지만 대비
        """
        x1 = seq[0] if len(seq) >= 1 else None
        x2 = None
        y  = None
        bt = ""

        if len(seq) == 1:
            # dict 하나만 들어온 케이스
            if isinstance(seq[0], dict):
                return _extract_from_dict(seq[0])
            # 텐서 하나만: x1만
            return x1, x2, y, bt

        if len(seq) == 2:
            # (x1, y) 일반 패턴
            if isinstance(seq[1], torch.Tensor):
                y = seq[1]
            elif isinstance(seq[1], dict):
                meta = seq[1]
                bt = meta.get("base_times", meta.get("base_time", ""))
            elif isinstance(seq[1], (list, tuple)):
                # (x1, (y, meta)) 같은 드문 케이스
                inner = seq[1]
                if len(inner) >= 1 and isinstance(inner[0], torch.Tensor):
                    y = inner[0]
                if len(inner) >= 2 and isinstance(inner[1], dict):
                    meta = inner[1]
                    bt = meta.get("base_times", meta.get("base_time", ""))
            return x1, x2, y, bt

        # len(seq) >= 3: (x1, x2, y) 또는 변형
        # 우선 마지막 텐서를 y로 간주, 두 번째 텐서가 있으면 x2 후보로
        last = seq[-1]
        if isinstance(last, torch.Tensor):
            y = last
            if isinstance(seq[1], torch.Tensor):
                x2 = seq[1]
            # 중간에 dict가 있으면 base_time 추출
            for item in seq[1:-1]:
                if isinstance(item, dict):
                    bt = item.get("base_times", item.get("base_time", bt))
        else:
            # 마지막이 dict인 경우 등: 텐서들만 추려서 (x1,x2,y) 추정
            tensors = [s for s in seq[1:] if isinstance(s, torch.Tensor)]
            if len(tensors) >= 1:
                # (x1, y) 패턴
                if len(tensors) == 1:
                    y = tensors[0]
                else:
                    x2, y = tensors[0], tensors[-1]
            for item in seq[1:]:
                if isinstance(item, dict):
                    bt = item.get("base_times", item.get("base_time", bt))

        return x1, x2, y, bt


    if isinstance(batch, dict):
        x1, x2, y, base_times = _extract_from_dict(batch)
    elif isinstance(batch, (list, tuple)):
        x1, x2, y, base_times = _extract_from_seq(batch)
    elif isinstance(batch, torch.Tensor):
        x1, x2, y, base_times = batch, None, None, ""
    else:
        raise TypeError(f"[load_val_sample] Unsupported batch type: {type(batch)}")

    if not isinstance(x1, torch.Tensor):
        info = list(batch.keys()) if isinstance(batch, dict) else type(batch)
        raise KeyError(f"[load_val_sample] Could not find x1 tensor. keys={x1_keys}, batch={info}")

    return x1, x2, y, base_times


# =========================
# FFT / 주파수 축 / 원형 bin / 집계
# =========================

def _fft2_shifted(img2d: np.ndarray):
    """2D FFT and shift to center."""
    return np.fft.fftshift(np.fft.fft2(img2d))

def _fftfreq_centers(N: int, d: float, unit: str):
    """fftshift된 '중심 주파수' 1D 벡터 반환."""
    f = np.fft.fftfreq(N, d=d)  # cycles per unit (ex: cyc/deg)
    if unit == "rad":
        f = 2*np.pi*f           # radians per unit (ex: rad/deg)
    return np.fft.fftshift(f)

def _freq_axes(H: int, W: int, unit: str = "cyc", dy: float = 1.0, dx: float = 1.0):
    """fftshift된 주파수 축 (센터들)."""
    fy = _fftfreq_centers(H, dy, unit)  # y축
    fx = _fftfreq_centers(W, dx, unit)  # x축
    return fx, fy

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

def _amp_transform(ratio: np.ndarray, mode: str = "db", eps: float = 1e-8):
    """|Ŷ|/|Y| → 표시값."""
    if mode == "db":
        return 20.0 * np.log10(np.maximum(ratio, eps))
    if mode == "pct":
        return 100.0 * (ratio - 1.0)
    if mode == "linear":
        return ratio
    raise ValueError("amp_mode must be 'db'|'pct'|'linear'")

# =========================
# Core: 단일 (H,W) 프레임에 대한 amplitude-ratio radial profile
# =========================

def _compute_amp_profile_for_frame(pred_2d: np.ndarray, true_2d: np.ndarray,
                                   nbins: int = 128,
                                   mask_percentile: float = 5.0,
                                   reducer: str = "median",
                                   amp_mode: str = "db",
                                   unit: str = "cyc",
                                   dy: float = 1.0, dx: float = 1.0):
    """
    pred_2d, true_2d: (H, W) numpy arrays
    반환: (x_k_centers, amp_profile)
    - x축은 'radial frequency k'의 대표값(각 bin의 중앙값)
    - unit/dx/dy 반영
    """
    H, W = true_2d.shape
    Ft = _fft2_shifted(true_2d)
    Fp = _fft2_shifted(pred_2d)
    mag_t, mag_p = np.abs(Ft), np.abs(Fp)

    # 작은 진폭 제거용 마스크
    thr_t = np.percentile(mag_t, mask_percentile)
    thr_p = np.percentile(mag_p, mask_percentile)
    mask_mag = (mag_t >= thr_t) & (mag_p >= thr_p)

    # amplitude ratio 맵 (표시 모드 변환 전)
    ratio  = mag_p / (mag_t + 1e-8)
    ampmap = _amp_transform(ratio, mode=amp_mode, eps=1e-8)

    # 원형 k 그리드 (unit/dx/dy 반영)
    fx, fy = _freq_axes(H, W, unit=unit, dy=dy, dx=dx)  # centers
    FXc, FYc = np.meshgrid(fx, fy)
    k_grid = np.sqrt(FXc**2 + FYc**2)

    # binning
    _, bins, _ = _radial_bins(H, W, nbins=nbins)
    amp_prof = _radial_stat(ampmap, bins, nbins, reducer=reducer, mask=mask_mag)

    # 각 bin의 대표 k (중앙값)
    k_centers = np.zeros(nbins, dtype=np.float64)
    k_flat = k_grid.ravel()
    for k in range(nbins):
        idx = (bins == k)
        k_centers[k] = np.nan if not np.any(idx) else np.nanmedian(k_flat[idx])

    # 정렬: 작은 k → 큰 k
    order = np.argsort(k_centers)
    xvals = k_centers[order]
    yvals = amp_prof[order]
    return xvals, yvals

# =========================
# Public: 세 모델의 amplitude profile을 하나의 축에 플로팅
# =========================
'''
@torch.no_grad()
def visualize_amp_three_models(
    pred_mse: torch.Tensor,
    pred_facl: torch.Tensor,
    pred_wacl: torch.Tensor,   # FACL+WACL Lv3
    true: torch.Tensor,
    *,
    b: int = 0,
    ch_idx: int = 0,
    t_idx: int = 0,
    nbins: int = 128,
    mask_percentile: float = 5.0,
    reducer: str = "median",        # 'median' or 'mean'
    amp_mode: str = "db",           # 'db' | 'pct' | 'linear'
    unit: str = "cyc",
    dx: float = 1.0,
    dy: float = 1.0,
    base_time: str = "",
    title_suffix: str = "",
    amp_profile_ylim=None,          # e.g., (-12, 6) for dB
    xlabel: str = None,
    save_path: str = None,
    dpi: int = 300
):
    """
    입력 텐서: (B, C, T, H, W)
    b: 배치 인덱스, ch_idx: 채널 인덱스, t_idx: 시간 인덱스
    """
    def _pick_frame(x: torch.Tensor) -> np.ndarray:
        # x[b, ch_idx, t_idx] → (H, W)
        return x[b, ch_idx, t_idx].detach().cpu().numpy()

    mse_2d  = _pick_frame(pred_mse)
    facl_2d = _pick_frame(pred_facl)
    wacl_2d = _pick_frame(pred_wacl)
    true_2d = _pick_frame(true)

    # 각각 radial profile 계산 (unit/dx/dy 반영)
    x_k, y_mse  = _compute_amp_profile_for_frame(
        mse_2d, true_2d, nbins=nbins, mask_percentile=mask_percentile,
        reducer=reducer, amp_mode=amp_mode, unit=unit, dx=dx, dy=dy
    )
    _,   y_facl = _compute_amp_profile_for_frame(
        facl_2d, true_2d, nbins=nbins, mask_percentile=mask_percentile,
        reducer=reducer, amp_mode=amp_mode, unit=unit, dx=dx, dy=dy
    )
    _,   y_wacl = _compute_amp_profile_for_frame(
        wacl_2d, true_2d, nbins=nbins, mask_percentile=mask_percentile,
        reducer=reducer, amp_mode=amp_mode, unit=unit, dx=dx, dy=dy
    )

    # 라벨
    if xlabel is None:
        xlabel = "Radial frequency k (cycles/unit)" if unit == "cyc" else "Radial frequency k (rad/unit)"

    # 플롯
    plt.figure(figsize=(8.8, 5.3))
    title = f"Amp ratio profile [{amp_mode}]  t₀={base_time}  (t₀+{t_idx+1})"
    if title_suffix:
        title += f"  {title_suffix}"
    plt.title(title)

    l1, = plt.plot(x_k, y_mse,  lw=2.2, label="MSE",           alpha=0.95)
    l2, = plt.plot(x_k, y_facl, lw=2.2, label="FACL",          alpha=0.95)
    l3, = plt.plot(x_k, y_wacl, lw=2.2, label="FACL+WACL Lv3", alpha=0.95)

    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    ylabel = "Amplitude ratio"
    if amp_mode == "db":
        ylabel += " [dB]"
    elif amp_mode == "pct":
        ylabel += " [%]"
    plt.ylabel(ylabel)

    # y축 범위
    if amp_profile_ylim is not None:
        plt.ylim(*amp_profile_ylim)
    else:
        y_all = np.concatenate([y_mse, y_facl, y_wacl])
        if np.isfinite(y_all).any():
            lo = np.nanpercentile(y_all, 1)
            hi = np.nanpercentile(y_all, 99)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = np.nanmin(y_all), np.nanmax(y_all)
                if lo == hi:
                    lo, hi = (lo - 1.0, hi + 1.0)
            plt.ylim(lo, hi)

    plt.legend(loc="best")

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()'''

# =========================
# CKPT 로드 → 동일 샘플 예측 → 한 그래프에 그리기
# =========================

@torch.no_grad()
def get_amp_profiles_from_ckpts(
    cfg_path: str,
    ckpt_mse_path: str,
    ckpt_facl_path: str,
    ckpt_fwacl_path: str,
    sample_idx: int = 0,
    ch_idx: int = 0,
    t_idx: int = 0,
    nbins: int = 128,
    mask_percentile: float = 5.0,
    reducer: str = "median",          # "median" | "mean"
    amp_mode: str = "db",             # "db" | "pct" | "linear"
    unit: str = "cyc",
    dx: float = 1.0,
    dy: float = 1.0,
    base_time: str = "",
    # ▼▼ 새로 추가: 플롯 옵션 ▼▼
    make_plot: bool = False,
    plot_save_path: str = None,
    plot_figsize=(8.8, 3),
    plot_dpi: int = 300,
    legend_loc: str = "best",
):
    """
    체크포인트 3개(MSE/FACL/FACL+WACL Lv3)를 로드하여 동일 샘플의
    amplitude-ratio radial profile (GT 대비)을 계산해 dict로 반환.
    실패 시에도 None을 반환하지 않고 {"error","trace"}를 포함한 dict를 반환.

    make_plot=True 일 때, 다음 형식으로 그래프를 그림:
      - 제목: "(b) Radial Frequency Profiles of Amplitude Ratio" (왼쪽 정렬)
      - y=0 기준선: 검정 점선
    """
    import numpy as np, torch, traceback, os
    from omegaconf import OmegaConf
    from models.get_model import build_model
    from datasets.dataloader import get_test_loader

    # --------------------------
    # 유연한 모델 호출자
    # --------------------------
    def _call_model(model, x1, x2, batch_raw=None, *, allow_dummy_x2=True, device=None):
        import inspect
        if device is None:
            try:
                device = next(model.parameters()).device
            except Exception:
                device = x1.device
        x1 = x1.to(device)
        if isinstance(x2, torch.Tensor):
            x2 = x2.to(device)

        aux_keys = ("x2", "x_humid", "qavg", "aux", "x_aux")
        f = model.forward
        sig = inspect.signature(f)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        names = [p.name for p in params if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        )]

        def _get_x2():
            x2_eff = x2
            if x2_eff is None and batch_raw:
                for k in aux_keys:
                    if k in batch_raw:
                        return batch_raw[k].to(device)
            if x2_eff is None and allow_dummy_x2:
                return torch.zeros_like(x1)
            return x2_eff

        if len(names) == 1 and names[0] in ("x", "x1", "inputs", "data"):
            return model(x1)
        if len(names) >= 2 and {"x1","x2"}.issubset(set(names)):
            x2_eff = _get_x2()
            if x2_eff is None:
                raise RuntimeError("Model expects (x1,x2) but x2 is missing.")
            return model(x1, x2_eff)
        if len(names) == 1:
            return model(**{names[0]: x1})
        if len(names) == 2:
            x2_eff = _get_x2()
            if x2_eff is None:
                raise RuntimeError(f"Model expects two args {names}, but x2 is missing.")
            return model(**{names[0]: x1, names[1]: x2_eff})
        if has_var_kw:
            kwargs = {}
            if "x1" in names: kwargs["x1"] = x1
            elif "x" in names: kwargs["x"]  = x1
            x2_eff = _get_x2()
            if x2_eff is not None and "x2" in names: kwargs["x2"] = x2_eff
            return model(**kwargs)
        raise RuntimeError(f"Model forward signature not matched: {sig}, names={names}")

    # --------------------------
    # 방사형 프로파일
    # --------------------------
    def _radial_amp_ratio_profile(gt2d: np.ndarray, pr2d: np.ndarray,
                                  nbins: int, mask_percentile: float,
                                  reducer: str, amp_mode: str,
                                  dx: float, dy: float):
        assert gt2d.shape == pr2d.shape, "GT/Pred shape mismatch"
        H, W = gt2d.shape
        eps = 1e-12
        G = np.fft.fftshift(np.fft.fft2(gt2d))
        P = np.fft.fftshift(np.fft.fft2(pr2d))
        AG = np.abs(G); AP = np.abs(P)
        thr = np.nanpercentile(AG, mask_percentile)
        mask = AG > max(thr, eps)

        kx = np.fft.fftshift(np.fft.fftfreq(W, d=dx))
        ky = np.fft.fftshift(np.fft.fftfreq(H, d=dy))
        KX, KY = np.meshgrid(kx, ky)
        KR = np.sqrt(KX**2 + KY**2)

        R = AP / np.maximum(AG, eps)
        R = np.where(mask, R, np.nan)

        kmax = np.nanmax(KR)
        edges = np.linspace(0.0, kmax, nbins + 1)
        centers = 0.5*(edges[:-1] + edges[1:])
        y = np.empty_like(centers)

        for i in range(nbins):
            m = (KR >= edges[i]) & (KR < edges[i+1])
            vals = R[m]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                y[i] = np.nan
            else:
                y[i] = float(np.mean(vals)) if reducer.lower()=="mean" else float(np.median(vals))

        if amp_mode.lower() == "db":
            y = 20.0*np.log10(np.maximum(y, eps))
        elif amp_mode.lower() == "pct":
            y = (y - 1.0) * 100.0
        return centers, y

    try:
        # 1) cfg/device
        cfg = OmegaConf.load(cfg_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) 모델 로드
        def _load_model(ckpt_path: str):
            model = build_model(cfg).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            state = None
            if isinstance(ckpt, dict):
                for k in ("state_dict","model_state_dict","model","net","ema_state_dict","ema"):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state = ckpt[k]; break
                if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    state = ckpt
            if state is None:
                tops = list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt)
                raise RuntimeError(f"state_dict not found; top-keys={tops}")
            if any(k.startswith("module.") for k in state.keys()):
                state = {k[len("module."):] : v for k,v in state.items()}
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError:
                model.load_state_dict(state, strict=False)
            model.eval()
            return model

        m_mse   = _load_model(ckpt_mse_path)
        m_facl  = _load_model(ckpt_facl_path)
        m_fwacl = _load_model(ckpt_fwacl_path)

        # 3) get_test_loader에서 sample_idx번째 샘플 추출
        test_loader = get_test_loader(cfg)
        seen = 0
        batch_raw_for_sig = None
        x1 = x2 = y = base_times = None

        for batch in test_loader:
            if isinstance(batch, dict):
                if all(k in batch for k in ("X1","X2","Y")):
                    bx1, bx2, by = batch["X1"], batch["X2"], batch["Y"]
                elif all(k in batch for k in ("x1","x2","y")):
                    bx1, bx2, by = batch["x1"], batch["x2"], batch["y"]
                else:
                    vals = [v for v in batch.values() if torch.is_tensor(v)]
                    if len(vals) < 2:
                        raise RuntimeError(f"Unknown batch dict keys: {list(batch.keys())}")
                    bx1, by = vals[0], vals[1]
                    bx2 = vals[2] if len(vals) >= 3 else None
                base_times = batch.get("base_times", base_times)
                batch_raw_for_sig = batch
            else:
                if len(batch) == 3:
                    (bx1, bx2, by) = batch
                elif len(batch) == 2:
                    (bx1, by) = batch; bx2 = None
                else:
                    raise RuntimeError(f"Unexpected batch structure len={len(batch)}")

            bsz = bx1.size(0)
            start = seen
            end = seen + bsz
            if sample_idx < end:
                i = sample_idx - start
                x1 = bx1[i].unsqueeze(0)
                y  = by[i].unsqueeze(0)
                x2 = bx2[i].unsqueeze(0) if isinstance(bx2, torch.Tensor) else None
                break
            seen = end

        if x1 is None:
            raise RuntimeError(f"sample_idx={sample_idx} is out of range (seen={seen})")

        if not base_time:
            base_time = "?"

        # 4) 디바이스 이동
        x1 = x1.to(device)
        if isinstance(x2, torch.Tensor): x2 = x2.to(device)
        y  = y.to(device)

        # 5) 추론
        pred_mse   = _call_model(m_mse,   x1, x2, batch_raw_for_sig, allow_dummy_x2=True, device=device)
        pred_facl  = _call_model(m_facl,  x1, x2, batch_raw_for_sig, allow_dummy_x2=True, device=device)
        pred_fwacl = _call_model(m_fwacl, x1, x2, batch_raw_for_sig, allow_dummy_x2=True, device=device)

        # 6) (B,C,T,H,W) 보장 → 프레임 선택
        def _ensure_5d(t):
            if t.dim() == 5: return t
            if t.dim() == 4: return t.unsqueeze(2)
            if t.dim() == 3: return t.unsqueeze(0).unsqueeze(0)
            raise ValueError(f"expect 3D/4D/5D, got {tuple(t.shape)}")

        pred_mse   = _ensure_5d(pred_mse)
        pred_facl  = _ensure_5d(pred_facl)
        pred_fwacl = _ensure_5d(pred_fwacl)
        y          = _ensure_5d(y)

        def _pick_frame(t):
            b = 0
            c = min(ch_idx, t.size(1)-1)
            tt= min(t_idx,  t.size(2)-1)
            return t[b, c, tt].detach().float().cpu().numpy()

        gt2d   = _pick_frame(y)
        mse2d  = _pick_frame(pred_mse)
        facl2d = _pick_frame(pred_facl)
        fwacl2d= _pick_frame(pred_fwacl)

        # 7) 프로파일 계산
        k, y_mse   = _radial_amp_ratio_profile(gt2d, mse2d,   nbins, mask_percentile, reducer, amp_mode, dx, dy)
        _, y_facl  = _radial_amp_ratio_profile(gt2d, facl2d,  nbins, mask_percentile, reducer, amp_mode, dx, dy)
        _, y_fwacl = _radial_amp_ratio_profile(gt2d, fwacl2d, nbins, mask_percentile, reducer, amp_mode, dx, dy)

        # 8) (옵션) 플롯 생성
        if make_plot:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=plot_figsize, dpi=plot_dpi)
            ax = fig.add_subplot(111)

            # 곡선
            ax.plot(k, y_mse,   lw=2.2, label="MSE",           alpha=0.95)
            ax.plot(k, y_facl,  lw=2.2, label="FACL",          alpha=0.95)
            ax.plot(k, y_fwacl, lw=2.2, label="FWCL", alpha=0.95)

            # 0 기준선 (검정 점선)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.9)

            # 축 라벨
            x_unit = "cycles/unit" if unit == "cyc" else unit
            ax.set_xlabel(f"Radial frequency k ({x_unit})")
            if amp_mode.lower() == "db":
                ax.set_ylabel("Amplitude ratio [dB]")
            elif amp_mode.lower() == "pct":
                ax.set_ylabel("Amplitude ratio [%]")
            else:
                ax.set_ylabel("Amplitude ratio")

            # 제목: 왼쪽 정렬
            ax.set_title("(e) Radial Frequency Profiles of Amplitude Ratio", loc="left")

            ax.grid(True, alpha=0.3)
            ax.legend(loc=legend_loc)

            if plot_save_path:
                os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
                fig.savefig(plot_save_path, bbox_inches="tight", dpi=plot_dpi)
                plt.close(fig)
            else:
                plt.show()

        # 9) 반환
        return {
            "profiles": {"k": k, "mse": y_mse, "facl": y_facl, "fwacl": y_fwacl},
            "indices": {"sample_idx": sample_idx, "ch_idx": ch_idx, "t_idx": t_idx},
            "base_time": base_time,
            "meta": {"notes": "ok"}
        }

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def plot_freq_snr_from_ckpts(
    cfg_path: str,
    ckpt_mse_path: str,
    ckpt_facl_path: str,
    ckpt_wfcl_path: str,
    save_path: str,
    model_labels=("MSE", "FACL", "WFCL (lv3)"),
    lead_labels=("t+1", "t+2", "t+3"),
    n_radial_bins: int = 60,
    max_batches: int = None,
    title_suffix: str = "",
    dpi: int = 150,
):
    """
    체크포인트 3개(MSE / FACL / WFCL)를 로드하여 전체 테스트셋에서
    Frequency-wise SNR 비교 그래프를 생성한다.

    SNR(k) = 10·log10( E[|FFT(GT)|²](k) / E[|FFT(GT−Pred)|²](k) )

    Parameters
    ----------
    cfg_path        : 공통 config YAML 경로 (normalization 등 파라미터 사용)
    ckpt_mse_path   : MSE 모델 체크포인트 (.pt)
    ckpt_facl_path  : FACL 모델 체크포인트 (.pt)
    ckpt_wfcl_path  : WFCL(lv3) 모델 체크포인트 (.pt)
    save_path       : 저장할 PNG 경로
    max_batches     : None이면 전체 테스트셋, 정수면 해당 배치 수만 사용 (빠른 확인용)
    """
    import numpy as np
    import torch
    import traceback
    from omegaconf import OmegaConf
    from models.get_model import build_model
    from datasets.dataloader import get_test_loader

    try:
        cfg    = OmegaConf.load(cfg_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        norm_tp = cfg.get("normalization", {}).get("tp", {})
        mu_tp  = float(norm_tp.get("mean", 0.002479317132383585))
        sig_tp = float(norm_tp.get("std",  0.005952117739409208))

        # ── 모델 로드 ────────────────────────────────────────────────────
        def _load_model(ckpt_path):
            model = build_model(cfg).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device)
            state = None
            if isinstance(ckpt, dict):
                for k in ("state_dict", "model_state_dict", "model", "net"):
                    if k in ckpt and isinstance(ckpt[k], dict):
                        state = ckpt[k]; break
                if state is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    state = ckpt
            if state is None:
                raise RuntimeError(f"state_dict not found in {ckpt_path}")
            if any(k.startswith("module.") for k in state.keys()):
                state = {k[len("module."):]: v for k, v in state.items()}
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError:
                model.load_state_dict(state, strict=False)
            model.eval()
            return model

        m_mse  = _load_model(ckpt_mse_path)
        m_facl = _load_model(ckpt_facl_path)
        m_wfcl = _load_model(ckpt_wfcl_path)

        # ── 테스트 데이터 로더 ───────────────────────────────────────────
        test_loader = get_test_loader(cfg)

        preds_mse,  targets_mse  = [], []
        preds_facl, targets_facl = [], []
        preds_wfcl, targets_wfcl = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if isinstance(batch, dict):
                    x1 = batch["X1"].to(device).float()
                    x2 = batch.get("X2")
                    if isinstance(x2, torch.Tensor):
                        x2 = x2.to(device).float()
                    y  = batch["Y"].to(device).float()
                else:
                    x1, x2, y = (batch[0].to(device).float(),
                                  batch[1].to(device).float(),
                                  batch[2].to(device).float())

                x2_in = x2 if isinstance(x2, torch.Tensor) else torch.zeros_like(x1)

                pred_mse  = m_mse(x1,  x2_in).detach().cpu()
                pred_facl = m_facl(x1, x2_in).detach().cpu()
                pred_wfcl = m_wfcl(x1, x2_in).detach().cpu()
                y_cpu = y.detach().cpu()

                preds_mse.append(pred_mse);   targets_mse.append(y_cpu)
                preds_facl.append(pred_facl); targets_facl.append(y_cpu)
                preds_wfcl.append(pred_wfcl); targets_wfcl.append(y_cpu)

        # ── 비교 플롯 생성 ───────────────────────────────────────────────
        visualize_freq_snr_comparison(
            models_data=[
                (preds_mse,  targets_mse,  model_labels[0]),
                (preds_facl, targets_facl, model_labels[1]),
                (preds_wfcl, targets_wfcl, model_labels[2]),
            ],
            save_path=save_path,
            mu_tp=mu_tp,
            sig_tp=sig_tp,
            lead_labels=lead_labels,
            n_radial_bins=n_radial_bins,
            title_suffix=title_suffix,
            dpi=dpi,
        )
        print(f"[freq_snr] Saved → {save_path}")

    except Exception as e:
        print(f"[freq_snr] ERROR: {e}")
        print(traceback.format_exc())


#=================ppt 슬라이드 1- 2D FFT 시각화================
def save_gt_fft2_plane(
    gt_tensor: torch.Tensor,
    out_png: str = None,
    *,
    vis_dir: str = None,
    loss_name: str = None,
    b: int = 0, c: int = 0, t: int = 0,
    unit: str = "cyc",
    dy: float = 1.0, dx: float = 1.0,
    amp: str = "logmag",
    cmap: str = "viridis",
    add_guides: bool = True,
    dpi: int = 300,
    title: str = "2D FFT (Ground Truth)",
    base_time: str = "",
    low_pct: float = 1.0, high_pct: float = 99.0,
    **kwargs,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # ---- 저장 경로
    def _sanitize(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (s or ""))

    if out_png is None:
        if vis_dir is None:
            raise ValueError("save_gt_fft2_plane: 'out_png' 또는 'vis_dir' 중 하나는 반드시 지정해야 합니다.")
        os.makedirs(vis_dir, exist_ok=True)
        loss_tag = f"_loss-{_sanitize(loss_name)}" if loss_name else ""
        fname = (
            f"fft2_gt_base-{_sanitize(base_time)}_b{b}_c{c}_t{t}{loss_tag}.png"
            if base_time else f"fft2_gt_b{b}_c{c}_t{t}{loss_tag}.png"
        )
        out_png = os.path.join(vis_dir, fname)

    # ---- (H,W) 슬라이스
    x = gt_tensor
    if x.dim() == 5:        # (B,C,T,H,W)
        x2d = x[b, c, t].detach().cpu().numpy()
    elif x.dim() == 4:      # (C,T,H,W)
        x2d = x[c, t].detach().cpu().numpy()
    elif x.dim() == 2:      # (H,W)
        x2d = x.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected gt_tensor shape: {tuple(x.shape)}")
    H, W = x2d.shape

    # ---- FFT2
    F = _fft2_shifted(x2d)
    mag = np.abs(F) + 1e-12
    if amp in ("logmag", "powerdb"):
        ampmap = 20.0 * np.log10(mag)
        clabel = "20·log10|F(kx, ky)| [dB]"
    else:
        ampmap = mag
        clabel = "|F(kx, ky)|"

    # ---- 주파수 축/엣지
    fx, fy = _freq_axes(H, W, unit=unit, dy=dy, dx=dx)
    FXe, FYe = _mesh_edges(fx, fy)

    # ---- Figure (inset_axes 대신 divider 사용)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))  # constrained_layout=False
    ttl = f"{title}"
    ttl += f"  (t₀={base_time}, b={b}, c={c}, t={t})" if base_time else f"  (b={b}, c={c}, t={t})"
    if loss_name:
        ttl += f"  | loss={loss_name}"
    ax.set_title(ttl, fontsize=12, pad=4)

    # 퍼센타일 범위 (안전)
    valid = np.isfinite(ampmap)
    if not np.any(valid):
        vmin, vmax = 0.0, 1.0
    else:
        lo, hi = np.nanpercentile(ampmap[valid], [low_pct, high_pct])
        if not (lo < hi):
            hi = lo + (1e-6 if lo == 0 else abs(lo) * 1e-6)
        vmin, vmax = lo, hi

    pcm = ax.pcolormesh(FXe, FYe, ampmap, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("kx (cycles/unit)" if unit == "cyc" else "kx (rad/unit)", labelpad=2)
    ax.set_ylabel("ky (cycles/unit)" if unit == "cyc" else "ky (rad/unit)", labelpad=2)
    if add_guides:
        _add_freq_guides(ax, unit=unit, dy=dy, dx=dx, alpha=0.25, show_box=True)

    # === 컬러바: axes divider로 오른쪽에 딱 붙이기 (tight와 호환) ===
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.04)  # pad를 작게 → 여백 최소
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(clabel)

    # 여백 미세 조정
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)

    # 저장 (tight 적용, pad=0)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_png


# ============================================================
# Frequency Bias & FSS for high-precipitation thresholds
# ============================================================
def plot_freq_bias_fss(
    all_obs, all_mse, all_facl, all_wfcl,
    out_dir: str,
    sig_tp: float = 0.005952117739409208,
    mu_tp:  float = 0.002479317132383585,
    window_size: int = 5,
    dpi: int = 150,
):
    """
    all_* : np.ndarray (N, C, T, H, W), z-score space
    두 임계값(obs 95th/99th 퍼센타일)에 대해 모델별 Frequency Bias와 FSS를 계산하고
    grouped bar chart로 저장한다.
    """
    import torch.nn.functional as F

    def denorm(z):
        return np.maximum(z * sig_tp + mu_tp, 0.0) * 1000.0

    obs_mm  = denorm(all_obs)
    mse_mm  = denorm(all_mse)
    facl_mm = denorm(all_facl)
    wfcl_mm = denorm(all_wfcl)

    obs_flat = obs_mm.ravel()
    thr_95 = float(np.percentile(obs_flat, 95))
    thr_99 = float(np.percentile(obs_flat, 99))
    thresholds = {"R95p": thr_95, "R99p": thr_99}

    import matplotlib.patches as mpatches

    model_arrays = {"MSE": mse_mm, "FACL": facl_mm, "WFCL": wfcl_mm}
    COLORS  = {"MSE": "#1f77b4", "FACL": "#ff7f0e", "WFCL": "#d62728"}
    HATCHES = {"R95p": "",       "R99p": "///"}

    freq_bias  = {t: {} for t in thresholds}
    fss_scores = {t: {} for t in thresholds}

    N, C, T, H, W = obs_mm.shape
    kernel = torch.ones(1, 1, window_size, window_size)
    n_win  = float(window_size * window_size)
    obs_t  = torch.from_numpy(obs_mm).float().reshape(N * C * T, 1, H, W)

    for i, (thr_name, thr_val) in enumerate(thresholds.items()):
        obs_exc = float((obs_flat > thr_val).sum())
        obs_bin  = (obs_t >= thr_val).float()
        obs_frac = F.conv2d(obs_bin, kernel, padding=window_size // 2) / n_win

        for model_name, pred_mm in model_arrays.items():
            # Frequency Bias
            pred_exc = float((pred_mm.ravel() > thr_val).sum())
            freq_bias[thr_name][model_name] = pred_exc / max(obs_exc, 1.0)

            # FSS
            pred_t    = torch.from_numpy(pred_mm).float().reshape(N * C * T, 1, H, W)
            pred_bin  = (pred_t >= thr_val).float()
            pred_frac = F.conv2d(pred_bin, kernel, padding=window_size // 2) / n_win
            mse_f  = ((obs_frac - pred_frac) ** 2).mean(dim=[2, 3])
            ref    = (obs_frac ** 2).mean(dim=[2, 3]) + (pred_frac ** 2).mean(dim=[2, 3])
            fss_scores[thr_name][model_name] = float((1.0 - mse_f / (ref + 1e-8)).mean().item())

    model_names = list(model_arrays.keys())
    thr_names   = list(thresholds.keys())
    x     = np.arange(len(model_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric_label, scores in [
        (axes[0], "Frequency Bias", freq_bias),
        (axes[1], "FSS",            fss_scores),
    ]:
        for i, thr_name in enumerate(thr_names):
            offset = (i - (len(thr_names) - 1) / 2) * width
            for j, model_name in enumerate(model_names):
                v = scores[thr_name][model_name]
                ax.bar(x[j] + offset, v, width,
                       color=COLORS[model_name],
                       hatch=HATCHES[thr_name],
                       edgecolor="black", linewidth=0.6)

        if metric_label == "Frequency Bias":
            ax.axhline(1.0, color="black", lw=1.4, ls="--", zorder=5)
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(0, 1)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label}  (R95p / R99p,  window={window_size})", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

        # Frequency Bias 서브플롯에만 perfect 선 범례 표시
        if metric_label == "Frequency Bias":
            perf_handle = [plt.Line2D([0], [0], color="black", lw=1.4, ls="--", label="Perfect (=1)")]
            ax.legend(handles=perf_handle, fontsize=9, loc="upper right")

        # 각 bar 위에 값 표시
        for rect in ax.patches:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    # 공통 범례: 모델(색상) + 임계값(해치 패턴) — figure 하단 중앙
    model_handles = [mpatches.Patch(color=COLORS[m], label=m) for m in model_names]
    thr_handles = [
        mpatches.Patch(facecolor="lightgrey", hatch="",    edgecolor="black", label="R95p"),
        mpatches.Patch(facecolor="lightgrey", hatch="///", edgecolor="black", label="R99p"),
    ]
    fig.legend(handles=model_handles + thr_handles,
               fontsize=10, ncol=len(model_names) + 2,
               loc="lower center", bbox_to_anchor=(0.5, 0.01))

    thr_str = "  |  ".join(f"{k} = {v:.2f} mm/day" for k, v in thresholds.items())
    fig.text(0.5, 0.10, f"Thresholds (from Obs):  {thr_str}",
             ha="center", fontsize=9, color="gray")

    fig.suptitle("Frequency Bias & FSS — High Precipitation Thresholds",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.14, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "freq_bias_fss_r95p_r99p.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FreqBias/FSS] Saved → {out_path}")
    return out_path


# ============================================================
# Regional CDF plot: MSE / FACL / WFCL vs Observation
# ============================================================
def plot_extreme_tail(
    all_obs, all_mse, all_facl, all_wfcl,
    out_dir: str,
    sig_tp: float = 0.005952117739409208,
    mu_tp:  float = 0.002479317132383585,
    dpi: int = 150,
):
    """
    극한 강수 평가를 위한 두 가지 시각화:
      Row 1 - Survival Function (1-CDF) on log-y scale
              : R90p 이상 구간을 log scale로 펼쳐 극한 꼬리 차이를 명확히 보여줌
      Row 2 - Q-Q plot (90th ~ 99.9th percentile)
              : 관측 분위수 vs 모델 분위수를 직접 비교, 대각선 = 완벽한 예측
    """
    lats = np.arange(-89.5, 90.0, 1.0)
    lons = np.arange(0.5,  360.0, 1.0)

    REGIONS = {
        "W-C Tropical Pacific\n(120°E–220°E, 25°S–35°N)": {
            "lat_range": (-25.0, 35.0),
            "lon_range": (120.0, 220.0),
        },
        "W North Atlantic\n(260°E–340°E, 20°S–40°N)": {
            "lat_range": (-20.0, 40.0),
            "lon_range": (260.0, 340.0),
        },
    }

    MODEL_STYLES = {
        "MSE":  dict(color="#1f77b4", lw=1.8, ls="--", label="MSE"),
        "FACL": dict(color="#ff7f0e", lw=1.8, ls="-.", label="FACL"),
        "WFCL": dict(color="#d62728", lw=1.8, ls=":",  label="WFCL"),
    }

    def _region_mask(lat_range, lon_range):
        lat_idx = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
        lon_idx = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
        return lat_idx, lon_idx

    def _denorm(z):
        return np.maximum(z * sig_tp + mu_tp, 0.0) * 1000.0

    def region_vals(arr, lat_idx, lon_idx):
        sub = arr[:, :, :, lat_idx[:, None], lon_idx[None, :]]
        return _denorm(sub.ravel())

    region_list = list(REGIONS.items())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # quantile levels for Q-Q plot: 90th ~ 99.9th (100 points, log-spaced in exceedance)
    qq_probs = 1.0 - np.logspace(np.log10(0.1), np.log10(0.001), 80)  # 0.90 ~ 0.999

    for col, (region_name, region_info) in enumerate(region_list):
        lat_idx, lon_idx = _region_mask(region_info["lat_range"], region_info["lon_range"])

        vals = {
            "Obs":  region_vals(all_obs,  lat_idx, lon_idx),
            "MSE":  region_vals(all_mse,  lat_idx, lon_idx),
            "FACL": region_vals(all_facl, lat_idx, lon_idx),
            "WFCL": region_vals(all_wfcl, lat_idx, lon_idx),
        }

        r90p = float(np.percentile(vals["Obs"], 90))
        x_max = float(np.percentile(vals["Obs"], 99.9))
        x_grid = np.linspace(r90p, x_max, 500)

        # ── Row 0: Survival Function (1-CDF) log-y ──────────────
        ax_sf = axes[0, col]

        for name in ["Obs"] + list(MODEL_STYLES.keys()):
            v = np.sort(vals[name])
            sf = 1.0 - np.searchsorted(v, x_grid, side="right") / len(v)
            sf = np.clip(sf, 1e-5, 1.0)
            if name == "Obs":
                ax_sf.semilogy(x_grid, sf, color="black", lw=2.2, ls="-", label="Obs", zorder=5)
            else:
                ax_sf.semilogy(x_grid, sf, **MODEL_STYLES[name], zorder=4)

        ax_sf.set_xlabel("Precipitation (mm/day)", fontsize=11)
        ax_sf.set_ylabel("Exceedance Probability (1−CDF)", fontsize=10)
        short = region_name.splitlines()[0]
        ax_sf.set_title(f"{short} – Survival Function (log scale)", fontsize=11)
        ax_sf.legend(fontsize=9)
        ax_sf.grid(True, which="both", alpha=0.3)
        ax_sf.set_xlim(r90p, x_max)

        # R90p / R95p / R99p 수직선
        for pct, lbl in [(90, "R90p"), (95, "R95p"), (99, "R99p")]:
            thresh = float(np.percentile(vals["Obs"], pct))
            if r90p <= thresh <= x_max:
                ax_sf.axvline(thresh, color="grey", lw=0.9, ls="--", alpha=0.6)
                ax_sf.text(thresh, ax_sf.get_ylim()[0] * 3, lbl,
                           fontsize=7, color="grey", ha="center", va="bottom")

        # ── Row 1: Q-Q plot (90th ~ 99.9th percentile) ──────────
        ax_qq = axes[1, col]

        obs_qq = np.quantile(vals["Obs"], qq_probs)

        # 1:1 reference line
        ax_qq.plot(obs_qq, obs_qq, color="black", lw=1.5, ls="-", label="1:1", zorder=5)

        for name, style in MODEL_STYLES.items():
            mod_qq = np.quantile(vals[name], qq_probs)
            ax_qq.plot(obs_qq, mod_qq, **style, zorder=4)

        ax_qq.set_xlabel("Observed Quantile (mm/day)", fontsize=11)
        ax_qq.set_ylabel("Model Quantile (mm/day)", fontsize=11)
        ax_qq.set_title(f"{short} – Q-Q Plot (90th–99.9th percentile)", fontsize=11)
        ax_qq.legend(fontsize=9)
        ax_qq.grid(True, alpha=0.3)

        # R95p / R99p 수직·수평 참조선
        for pct, lbl in [(95, "R95p"), (99, "R99p")]:
            thresh = float(np.percentile(vals["Obs"], pct))
            ax_qq.axvline(thresh, color="grey", lw=0.9, ls="--", alpha=0.6)
            ax_qq.text(thresh, ax_qq.get_ylim()[0] if ax_qq.get_ylim()[0] > 0 else obs_qq[0],
                       lbl, fontsize=7, color="grey", ha="center", va="bottom")

    fig.suptitle("Extreme Precipitation Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "extreme_tail_survival_qq.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Extreme Tail] Saved → {out_path}")
    return out_path


def plot_regional_cdf(
    out_dir: str,
    cfg_mse_path:  str = "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
    cfg_facl_path: str = "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
    cfg_wfcl_path: str = "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    dpi: int = 150,
):
    """
    두 지역(Western-Central Tropical Pacific, Western North Atlantic)에 대해
    MSE / FACL / WFCL 모델과 관측(Obs)의 강수 CDF를 한 그래프에 그린다.

    lat: -89.5 ~ 89.5 (step 1), lon: 0.5 ~ 359.5 (step 1)
    Western-Central Tropical Pacific : lon 120~220°E, lat -25~35°N  (red box)
    Western North Atlantic            : lon 260~340°E, lat -20~40°N  (green box)
    """
    import yaml
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 좌표 그리드 ─────────────────────────────────────────────
    lats = np.arange(-89.5, 90.0, 1.0)   # shape (180,)
    lons = np.arange(0.5,  360.0, 1.0)   # shape (360,)

    REGIONS = {
        "W-C Tropical Pacific\n(120°E–220°E, 25°S–35°N)": {
            "lat_range": (-25.0, 35.0),
            "lon_range": (120.0, 220.0),
            "color_box": "red",
        },
        "W North Atlantic\n(260°E–340°E, 20°S–40°N)": {
            "lat_range": (-20.0, 40.0),
            "lon_range": (260.0, 340.0),
            "color_box": "green",
        },
    }

    def _region_mask(lat_range, lon_range):
        lat_idx = np.where((lats >= lat_range[0]) & (lats <= lat_range[1]))[0]
        lon_idx = np.where((lons >= lon_range[0]) & (lons <= lon_range[1]))[0]
        return lat_idx, lon_idx

    def _load_model(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg).to(device).float()
        mon_key  = cfg.get("monitor", {}).get("key", "val/mse")
        mon_name = mon_key.split("/")[-1]
        ckpt_dir = cfg["paths"]["ckpt_dir"]
        ckpt_path = cfg.get("ckpt", os.path.join(ckpt_dir, f"best_{mon_name}.pt"))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        load_checkpoint_for_eval(ckpt_path, model, device=device)
        model.eval()
        return model, cfg

    def _extract_xy(batch):
        if isinstance(batch, dict):
            if all(k in batch for k in ("X1", "X2", "Y")):
                return (batch["X1"].to(device), batch["X2"].to(device)), batch["Y"].to(device)
            for k_in in ("input","inputs","x","src","data"):
                if k_in in batch:
                    x = batch[k_in]; break
            for k_tg in ("target","targets","y","label","labels"):
                if k_tg in batch:
                    y = batch[k_tg]; break
            return x.to(device), y.to(device)
        x, y = batch
        return x.to(device), y.to(device)

    def _forward(model, x):
        if isinstance(x, (list, tuple)):
            return model(*x)
        return model(x)

    def _denorm(z):
        # z-score → mm/day (clamp negatives to 0)
        return np.maximum(z * sig_tp + mu_tp, 0.0) * 1000.0

    # ── 모델 로드 ────────────────────────────────────────────────
    print("[CDF] Loading models ...")
    model_mse,  cfg_mse  = _load_model(cfg_mse_path)
    model_facl, cfg_facl = _load_model(cfg_facl_path)
    model_wfcl, cfg_wfcl = _load_model(cfg_wfcl_path)

    test_loader = get_test_loader(cfg_mse)

    # ── 전체 테스트셋 인퍼런스 ───────────────────────────────────
    print("[CDF] Running inference on test set ...")
    all_obs, all_mse, all_facl, all_wfcl = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y = _extract_xy(batch)
            p_mse  = _forward(model_mse,  x)
            p_facl = _forward(model_facl, x)
            p_wfcl = _forward(model_wfcl, x)
            # shape: (B, C, T, H, W) or (B, C, H, W) → ensure 5D
            def to5d(t):
                if t.ndim == 4: t = t.unsqueeze(2)
                return t
            all_obs.append(to5d(y).cpu().float().numpy())
            all_mse.append(to5d(p_mse).cpu().float().numpy())
            all_facl.append(to5d(p_facl).cpu().float().numpy())
            all_wfcl.append(to5d(p_wfcl).cpu().float().numpy())

    all_obs  = np.concatenate(all_obs,  axis=0)  # (N, C, T, H, W)
    all_mse  = np.concatenate(all_mse,  axis=0)
    all_facl = np.concatenate(all_facl, axis=0)
    all_wfcl = np.concatenate(all_wfcl, axis=0)

    # ── Frequency Bias & FSS ─────────────────────────────────────
    plot_freq_bias_fss(
        all_obs, all_mse, all_facl, all_wfcl,
        out_dir=out_dir, sig_tp=sig_tp, mu_tp=mu_tp, dpi=dpi,
    )

    # ── Extreme tail: Survival Function + Q-Q plot ───────────────
    plot_extreme_tail(
        all_obs, all_mse, all_facl, all_wfcl,
        out_dir=out_dir, sig_tp=sig_tp, mu_tp=mu_tp, dpi=dpi,
    )

    # ── 지역별 CDF 플롯 ──────────────────────────────────────────
    MODEL_STYLES = {
        "Obs":  dict(color="black",  lw=2.0, ls="-",  zorder=5, label="Obs"),
        "MSE":  dict(color="#1f77b4", lw=1.8, ls="--", zorder=4, label="MSE"),
        "FACL": dict(color="#ff7f0e", lw=1.8, ls="-.", zorder=3, label="FACL"),
        "WFCL": dict(color="#d62728", lw=1.8, ls=":",  zorder=3, label="WFCL"),
    }
    DIFF_STYLES = {
        "MSE":  dict(color="#1f77b4", lw=1.8, ls="--", label="MSE−Obs"),
        "FACL": dict(color="#ff7f0e", lw=1.8, ls="-.", label="FACL−Obs"),
        "WFCL": dict(color="#d62728", lw=1.8, ls=":",  label="WFCL−Obs"),
    }

    region_list = list(REGIONS.items())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (region_name, region_info) in enumerate(region_list):
        lat_idx, lon_idx = _region_mask(region_info["lat_range"], region_info["lon_range"])

        def region_vals(arr):
            sub = arr[:, :, :, lat_idx[:, None], lon_idx[None, :]]
            return _denorm(sub.ravel())

        vals = {
            "Obs":  region_vals(all_obs),
            "MSE":  region_vals(all_mse),
            "FACL": region_vals(all_facl),
            "WFCL": region_vals(all_wfcl),
        }

        from scipy.ndimage import gaussian_filter1d

        x_max  = float(np.percentile(vals["Obs"], 99.5))
        N_BINS = 80
        bins   = np.linspace(1.0, x_max, N_BINS + 1)   # 1 mm/day 이상만 (zero spike 제거)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # R95p / R99p 임계값 (Obs 기준)
        obs_pos = vals["Obs"][vals["Obs"] > 1.0]
        r95p = float(np.percentile(obs_pos, 95))
        r99p = float(np.percentile(obs_pos, 99))

        # ── 상단: PDF (log-y scale) ──────────────────────────────
        ax_pdf = axes[0, col]
        pdf_dict = {}
        for name, style in MODEL_STYLES.items():
            v_pos = vals[name][vals[name] > 1.0]
            pdf, _ = np.histogram(v_pos, bins=bins, density=True)
            pdf_smooth = gaussian_filter1d(pdf.astype(float), sigma=1.5)
            pdf_dict[name] = pdf_smooth
            ax_pdf.semilogy(bin_centers, pdf_smooth + 1e-8, **style)

        ax_pdf.set_xlabel("Precipitation (mm/day)", fontsize=11)
        ax_pdf.set_ylabel("Probability Density (log)", fontsize=11)
        ax_pdf.set_title(region_name, fontsize=12)
        ax_pdf.legend(fontsize=10)
        ax_pdf.grid(True, which="both", alpha=0.3)
        ax_pdf.set_xlim(1.0, x_max)

        # R95p / R99p 수직 참조선
        ylo, yhi = ax_pdf.get_ylim()
        for thresh, lbl in [(r95p, "R95p"), (r99p, "R99p")]:
            ax_pdf.axvline(thresh, color="grey", lw=0.9, ls="--", alpha=0.7)
            ax_pdf.text(thresh, yhi * 0.4, lbl,
                        fontsize=7, color="grey", ha="center", va="bottom")

        # ── 하단: ΔPDF (model − Obs) ─────────────────────────────
        ax_diff = axes[1, col]
        obs_pdf = pdf_dict["Obs"]
        for name, style in DIFF_STYLES.items():
            delta = pdf_dict[name] - obs_pdf
            ax_diff.plot(bin_centers, delta, **style)

        ax_diff.axhline(0, color="black", lw=1.2, ls="-")
        for thresh, lbl in [(r95p, "R95p"), (r99p, "R99p")]:
            ax_diff.axvline(thresh, color="grey", lw=0.9, ls="--", alpha=0.7)
        ax_diff.set_xlabel("Precipitation (mm/day)", fontsize=11)
        ax_diff.set_ylabel("ΔPDF (Model − Obs)", fontsize=11)
        ax_diff.set_title(f"{region_name.splitlines()[0]} – PDF Difference", fontsize=12)
        ax_diff.legend(fontsize=10)
        ax_diff.grid(True, alpha=0.3)
        ax_diff.set_xlim(1.0, x_max)

    fig.suptitle("Regional Precipitation PDF", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "regional_pdf_mse_facl_wfcl.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[PDF] Saved → {out_path}")

    # ── 지역별 CDF 플롯 (inset: CDF > 0.90) ─────────────────────
    fig_cdf, axes_cdf = plt.subplots(2, 2, figsize=(14, 10))

    for col, (region_name, region_info) in enumerate(region_list):
        lat_idx, lon_idx = _region_mask(region_info["lat_range"], region_info["lon_range"])

        def region_vals_cdf(arr):
            sub = arr[:, :, :, lat_idx[:, None], lon_idx[None, :]]
            return _denorm(sub.ravel())

        vals_cdf = {
            "Obs":  region_vals_cdf(all_obs),
            "MSE":  region_vals_cdf(all_mse),
            "FACL": region_vals_cdf(all_facl),
            "WFCL": region_vals_cdf(all_wfcl),
        }

        x_max_cdf = float(np.percentile(vals_cdf["Obs"], 99.5))
        x_grid_cdf = np.linspace(0.0, x_max_cdf, 1000)

        obs_pos_cdf = vals_cdf["Obs"][vals_cdf["Obs"] > 1.0]
        r95p_cdf = float(np.percentile(obs_pos_cdf, 95))
        r99p_cdf = float(np.percentile(obs_pos_cdf, 99))

        # ── 상단: 전체 CDF ──
        ax_cdf = axes_cdf[0, col]
        sorted_cdfs = {}
        for name, style in MODEL_STYLES.items():
            v = np.sort(vals_cdf[name])
            cdf = np.searchsorted(v, x_grid_cdf, side="right") / len(v)
            sorted_cdfs[name] = cdf
            ax_cdf.plot(x_grid_cdf, cdf, **style)

        # R95p / R99p 수직 참조선
        for thresh, lbl in [(r95p_cdf, "R95p"), (r99p_cdf, "R99p")]:
            ax_cdf.axvline(thresh, color="grey", lw=0.9, ls="--", alpha=0.7)
            ax_cdf.text(thresh, 0.05, lbl, fontsize=7, color="grey",
                        ha="center", va="bottom")

        ax_cdf.set_xlabel("Precipitation (mm/day)", fontsize=11)
        ax_cdf.set_ylabel("CDF", fontsize=11)
        ax_cdf.set_title(region_name, fontsize=12)
        ax_cdf.legend(fontsize=10)
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_xlim(0.0, x_max_cdf)
        ax_cdf.set_ylim(0.0, 1.0)

        # inset: 꼬리 영역 (CDF > 0.90)
        cdf_obs = sorted_cdfs["Obs"]
        tail_start_x = x_grid_cdf[np.searchsorted(cdf_obs, 0.90)]
        ax_ins = ax_cdf.inset_axes([0.52, 0.05, 0.44, 0.44])
        for name, style in MODEL_STYLES.items():
            ax_ins.plot(x_grid_cdf, sorted_cdfs[name], **style)
        ax_ins.set_xlim(tail_start_x, x_max_cdf)
        ax_ins.set_ylim(0.90, 1.00)
        ax_ins.grid(True, alpha=0.3)
        ax_ins.tick_params(labelsize=7)
        ax_ins.set_title("tail (CDF > 0.90)", fontsize=7, pad=2)
        for thresh in [r95p_cdf, r99p_cdf]:
            ax_ins.axvline(thresh, color="grey", lw=0.8, ls="--", alpha=0.6)

        # ── 하단: ΔCDF (model − Obs) ──
        ax_diff_cdf = axes_cdf[1, col]
        for name, style in DIFF_STYLES.items():
            diff = sorted_cdfs[name] - sorted_cdfs["Obs"]
            ax_diff_cdf.plot(x_grid_cdf, diff, **style)

        ax_diff_cdf.axhline(0, color="black", lw=1.2, ls="-")
        for thresh, lbl in [(r95p_cdf, "R95p"), (r99p_cdf, "R99p")]:
            ax_diff_cdf.axvline(thresh, color="grey", lw=0.9, ls="--", alpha=0.7)
        ax_diff_cdf.set_xlabel("Precipitation (mm/day)", fontsize=11)
        ax_diff_cdf.set_ylabel("ΔCDF (Model − Obs)", fontsize=11)
        ax_diff_cdf.set_title(f"{region_name.splitlines()[0]} – CDF Difference", fontsize=12)
        ax_diff_cdf.legend(fontsize=10)
        ax_diff_cdf.grid(True, alpha=0.3)
        ax_diff_cdf.set_xlim(0.0, x_max_cdf)

    fig_cdf.suptitle("Regional Precipitation CDF", fontsize=14, fontweight="bold")
    fig_cdf.tight_layout(rect=[0, 0, 1, 0.97])

    out_path_cdf = os.path.join(out_dir, "regional_cdf_mse_facl_wfcl.png")
    fig_cdf.savefig(out_path_cdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig_cdf)
    print(f"[CDF] Saved → {out_path_cdf}")

    return out_path


# ============================================================
# 전체 모델 음수 강수 비교 분석
# ============================================================
def compare_neg_precip_all_models(
    out_log_dir: str,
    out_eval_dir: str,
    logger=None,
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
):
    """
    6개 모델 전체의 음수 강수 통계를 계산하고 CSV + 히스토그램을 저장.
    모델을 순차적으로 로드/추론/삭제해서 GPU 메모리를 절약한다.

    저장 결과
    ---------
    out_log_dir/neg_precip_all_models.csv          : 모델×리드타임별 통계
    out_eval_dir/neg_precip_hist_all_models.png    : 모델×리드타임 히스토그램 (6×3)
    out_eval_dir/neg_precip_ratio_all_models.png   : 비율 비교 바차트
    """
    import csv as _csv_mod
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from utils.config_utils import load_config
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    PROJECT_ROOT = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2"

    MODEL_SPECS = [
        {"label": "MSE",             "cfg": "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml"},
        {"label": "FACL",            "cfg": "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml"},
        {"label": "FACL+WACL Lv1",  "cfg": "configs/configs_3d/facl_wacl_lv1_r90p_r95p/facl_wacl_lv1_r90p_r95p.yaml"},
        {"label": "FACL+WACL Lv2",  "cfg": "configs/configs_3d/facl_wacl_lv2_r90p_r95p/facl_wacl_lv2_r90p_r95p.yaml"},
        {"label": "FACL+WACL Lv3",  "cfg": "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml"},
        {"label": "WACL Low",        "cfg": "configs/configs_3d/wacl_lv3_low/wacl_lv3_low.yaml"},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_log_dir,  exist_ok=True)
    os.makedirs(out_eval_dir, exist_ok=True)

    all_rows  = []                   # CSV 행 누적
    neg_store = {}                   # label -> {lead_idx: np.array of neg mm values}

    for spec in MODEL_SPECS:
        label    = spec["label"]
        cfg_path = os.path.join(PROJECT_ROOT, spec["cfg"])
        _log(f"[NegPrecip-All] Loading {label} ...")

        cfg_m = load_config(cfg_path)
        norm  = cfg_m.get("normalization", {}).get("tp", {})
        mu    = float(norm.get("mean", mu_tp))
        sig   = float(norm.get("std",  sig_tp))

        model = build_model(cfg_m).to(device).float().eval()
        mon_key  = cfg_m.get("monitor", {}).get("key", "val/mse")
        ckpt_dir = cfg_m["paths"]["ckpt_dir"]
        ckpt_path = cfg_m.get("ckpt",
                    os.path.join(ckpt_dir, f"best_{mon_key.split('/')[-1]}.pt"))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        load_checkpoint_for_eval(ckpt_path, model, device=device)

        loader = get_test_loader(cfg_m)

        neg_per_lead   = {}   # lead_idx -> list of neg-mm arrays
        total_per_lead = {}

        with torch.no_grad():
            for batch in loader:
                x1 = batch["X1"].to(device).float()
                x2 = batch["X2"].to(device).float()
                pred = model(x1, x2).cpu().float().numpy()
                if pred.ndim == 5:
                    pred = pred[:, 0]          # (B,T,H,W)
                T_len = pred.shape[1]
                for t in range(T_len):
                    mm = (pred[:, t] * sig + mu) * 1000.0   # mm/day
                    total_per_lead[t] = total_per_lead.get(t, 0) + mm.size
                    neg_per_lead.setdefault(t, []).append(mm[mm < 0.0])

        del model
        torch.cuda.empty_cache()

        neg_store[label] = {}
        for t in sorted(neg_per_lead.keys()):
            neg_arr  = np.concatenate(neg_per_lead[t]) if neg_per_lead[t] else np.array([])
            n_neg    = int(neg_arr.size)
            n_total  = int(total_per_lead[t])
            ratio    = n_neg / n_total * 100 if n_total > 0 else 0.0
            min_val  = float(neg_arr.min())  if n_neg > 0 else 0.0
            mean_val = float(neg_arr.mean()) if n_neg > 0 else 0.0
            neg_store[label][t] = neg_arr
            all_rows.append({
                "model":         label,
                "lead_time":     t + 1,
                "neg_count":     n_neg,
                "total_count":   n_total,
                "neg_ratio_pct": round(ratio,    6),
                "min_mm":        round(min_val,  6),
                "mean_mm":       round(mean_val, 6),
            })
            _log(f"  [{label}] lead t+{t+1}: {n_neg:,}/{n_total:,} ({ratio:.3f}%)"
                 f"  min={min_val:.4f}  mean={mean_val:.4f} mm/day")

    # ── CSV 저장 ──────────────────────────────────────────────────────────
    csv_path = os.path.join(out_log_dir, "neg_precip_all_models.csv")
    with open(csv_path, "w", newline="") as f:
        writer = _csv_mod.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    _log(f"[NegPrecip-All] Saved CSV → {csv_path}")

    labels   = [s["label"] for s in MODEL_SPECS]
    lead_ids = sorted(next(iter(neg_store.values())).keys())
    n_models = len(labels)
    n_leads  = len(lead_ids)
    COLORS   = ["#457b9d", "#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#264653"]

    # ── 히스토그램 (n_leads 행 × n_models 열) ─────────────────────────────
    fig, axes = plt.subplots(n_leads, n_models,
                             figsize=(3.2 * n_models, 3.0 * n_leads), dpi=150)
    axes = np.array(axes).reshape(n_leads, n_models)

    for i, t in enumerate(lead_ids):
        for j, label in enumerate(labels):
            ax      = axes[i, j]
            neg_arr = neg_store[label][t]
            row     = next(r for r in all_rows
                           if r["model"] == label and r["lead_time"] == t + 1)
            if neg_arr.size > 0:
                ax.hist(neg_arr, bins=50, color=COLORS[j],
                        edgecolor="white", linewidth=0.3)
            else:
                ax.text(0.5, 0.5, "No negatives", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{label}  lead t+{t+1}\n"
                         f"{row['neg_ratio_pct']:.3f}%  (n={row['neg_count']:,})",
                         fontsize=7.5)
            ax.set_xlabel("mm/day", fontsize=7)
            ax.set_ylabel("count",  fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Negative Precipitation Distribution — All Models × Lead Times",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    hist_path = os.path.join(out_eval_dir, "neg_precip_hist_all_models.png")
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)
    _log(f"[NegPrecip-All] Saved histogram → {hist_path}")

    # ── 비율 바차트 (리드타임별 모델 비교) ────────────────────────────────
    x = np.arange(n_models)
    fig2, axes2 = plt.subplots(1, n_leads,
                               figsize=(4.5 * n_leads, 4), dpi=150, sharey=True)
    axes2 = np.array(axes2).reshape(-1)

    for i, t in enumerate(lead_ids):
        ax     = axes2[i]
        ratios = [next(r["neg_ratio_pct"] for r in all_rows
                       if r["model"] == lbl and r["lead_time"] == t + 1)
                  for lbl in labels]
        bars = ax.bar(x, ratios, color=COLORS, edgecolor="white", width=0.6)
        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(ratios) * 0.01,
                    f"{v:.3f}%", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"lead t+{t+1}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Negative ratio (%)", fontsize=9)

    fig2.suptitle("Negative Precipitation Ratio by Model & Lead Time",
                  fontsize=11, y=1.02)
    plt.tight_layout()
    bar_path = os.path.join(out_eval_dir, "neg_precip_ratio_all_models.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close(fig2)
    _log(f"[NegPrecip-All] Saved bar chart → {bar_path}")


# ============================================================
# facl_wacl 분석: 음수 강수 비율 표
# ============================================================
def make_negative_precip_table(
    out_dir: str,
    cfg_mse_path:  str = "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
    cfg_facl_path: str = "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
    cfg_wfcl_path: str = "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
):
    """
    각 모델 예측값에서 음수 강수(raw z-score → mm 변환 전)가 차지하는 비율을 계산해
    CSV 파일과 콘솔 표로 출력한다.

    '음수 강수'란 모델이 z-score 공간에서 예측한 값을 물리 단위(mm/day)로
    역정규화했을 때 z*sig+mu < 0 인 픽셀을 의미한다.
    """
    import yaml, csv
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg).to(device).float()
        mon_key  = cfg.get("monitor", {}).get("key", "val/mse")
        mon_name = mon_key.split("/")[-1]
        ckpt_dir = cfg["paths"]["ckpt_dir"]
        ckpt_path = cfg.get("ckpt", os.path.join(ckpt_dir, f"best_{mon_name}.pt"))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        load_checkpoint_for_eval(ckpt_path, model, device=device)
        model.eval()
        return model, cfg

    def _extract_xy(batch):
        if isinstance(batch, dict):
            if all(k in batch for k in ("X1", "X2", "Y")):
                return (batch["X1"].to(device), batch["X2"].to(device)), batch["Y"].to(device)
        x, y = batch
        return x.to(device), y.to(device)

    def _forward(model, x):
        if isinstance(x, (list, tuple)):
            return model(*x)
        return model(x)

    print("[NegPrecip] Loading models ...")
    model_mse,  cfg_mse  = _load_model(cfg_mse_path)
    model_facl, cfg_facl = _load_model(cfg_facl_path)
    model_wfcl, cfg_wfcl = _load_model(cfg_wfcl_path)

    test_loader = get_test_loader(cfg_mse)

    neg_counts = {"MSE": 0, "FACL": 0, "WFCL": 0}
    total_pixels = 0

    print("[NegPrecip] Running inference ...")
    with torch.no_grad():
        for batch in test_loader:
            x, y = _extract_xy(batch)
            p_mse  = _forward(model_mse,  x).cpu().float().numpy()
            p_facl = _forward(model_facl, x).cpu().float().numpy()
            p_wfcl = _forward(model_wfcl, x).cpu().float().numpy()

            # mm 단위로 변환 (clamping 없이 음수 그대로)
            mm_mse  = p_mse  * sig_tp + mu_tp
            mm_facl = p_facl * sig_tp + mu_tp
            mm_wfcl = p_wfcl * sig_tp + mu_tp

            neg_counts["MSE"]  += int((mm_mse  < 0).sum())
            neg_counts["FACL"] += int((mm_facl < 0).sum())
            neg_counts["WFCL"] += int((mm_wfcl < 0).sum())
            total_pixels += mm_mse.size

    rows = []
    print("\n[음수 강수 비율 표]")
    print(f"{'Model':<10} {'Neg Pixels':>12} {'Total Pixels':>14} {'Neg Ratio (%)':>15}")
    print("-" * 55)
    for name, cnt in neg_counts.items():
        ratio = 100.0 * cnt / max(1, total_pixels)
        print(f"{name:<10} {cnt:>12,} {total_pixels:>14,} {ratio:>14.4f}%")
        rows.append({"model": name, "neg_pixels": cnt,
                     "total_pixels": total_pixels, "neg_ratio_pct": round(ratio, 6)})

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "negative_precip_ratio_facl_wacl.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[NegPrecip] Saved → {csv_path}")
    return csv_path


# ============================================================
# facl_wacl 분석: drizzle fraction 표
# ============================================================
def make_drizzle_fraction_table(
    out_dir: str,
    cfg_mse_path:  str = "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
    cfg_facl_path: str = "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
    cfg_wfcl_path: str = "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    drizzle_lo: float = 0.0,
    drizzle_hi: float = 1.0,
):
    """
    drizzle fraction: 전체 픽셀 중 강수량이 (drizzle_lo, drizzle_hi] mm/day 구간인
    픽셀의 비율을 Obs / MSE / FACL / WFCL 각각에 대해 계산한다.

    기본 drizzle 기준: 0 < precip ≤ 1 mm/day
    """
    import yaml, csv
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg).to(device).float()
        mon_key  = cfg.get("monitor", {}).get("key", "val/mse")
        mon_name = mon_key.split("/")[-1]
        ckpt_dir = cfg["paths"]["ckpt_dir"]
        ckpt_path = cfg.get("ckpt", os.path.join(ckpt_dir, f"best_{mon_name}.pt"))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        load_checkpoint_for_eval(ckpt_path, model, device=device)
        model.eval()
        return model, cfg

    def _extract_xy(batch):
        if isinstance(batch, dict):
            if all(k in batch for k in ("X1", "X2", "Y")):
                return (batch["X1"].to(device), batch["X2"].to(device)), batch["Y"].to(device)
        x, y = batch
        return x.to(device), y.to(device)

    def _forward(model, x):
        if isinstance(x, (list, tuple)):
            return model(*x)
        return model(x)

    def _to_mmday(z):
        return np.maximum(z * sig_tp + mu_tp, 0.0) * 1000.0

    print("[Drizzle] Loading models ...")
    model_mse,  cfg_mse  = _load_model(cfg_mse_path)
    model_facl, cfg_facl = _load_model(cfg_facl_path)
    model_wfcl, cfg_wfcl = _load_model(cfg_wfcl_path)

    test_loader = get_test_loader(cfg_mse)

    drizzle_counts = {"Obs": 0, "MSE": 0, "FACL": 0, "WFCL": 0}
    total_pixels = 0

    print("[Drizzle] Running inference ...")
    with torch.no_grad():
        for batch in test_loader:
            x, y = _extract_xy(batch)
            obs_mm  = _to_mmday(y.cpu().float().numpy())
            mse_mm  = _to_mmday(_forward(model_mse,  x).cpu().float().numpy())
            facl_mm = _to_mmday(_forward(model_facl, x).cpu().float().numpy())
            wfcl_mm = _to_mmday(_forward(model_wfcl, x).cpu().float().numpy())

            def _drizzle_mask(arr):
                return ((arr > drizzle_lo) & (arr <= drizzle_hi)).sum()

            drizzle_counts["Obs"]  += int(_drizzle_mask(obs_mm))
            drizzle_counts["MSE"]  += int(_drizzle_mask(mse_mm))
            drizzle_counts["FACL"] += int(_drizzle_mask(facl_mm))
            drizzle_counts["WFCL"] += int(_drizzle_mask(wfcl_mm))
            total_pixels += obs_mm.size

    rows = []
    print(f"\n[Drizzle Fraction 표]  (기준: {drizzle_lo} < precip ≤ {drizzle_hi} mm/day)")
    print(f"{'Model':<10} {'Drizzle Px':>12} {'Total Px':>12} {'Drizzle (%)':>13}")
    print("-" * 52)
    for name, cnt in drizzle_counts.items():
        ratio = 100.0 * cnt / max(1, total_pixels)
        print(f"{name:<10} {cnt:>12,} {total_pixels:>12,} {ratio:>12.4f}%")
        rows.append({"model": name, "drizzle_pixels": cnt,
                     "total_pixels": total_pixels, "drizzle_fraction_pct": round(ratio, 6)})

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "drizzle_fraction_facl_wacl.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Drizzle] Saved → {csv_path}")
    return csv_path


# ============================================================
# facl_wacl 분석: 0–1 mm/day 구간 히스토그램
# ============================================================
def plot_01mm_histogram(
    out_dir: str,
    cfg_mse_path:  str = "configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml",
    cfg_facl_path: str = "configs/configs_3d/facl_r90p_r95p/facl_r90p_r95p.yaml",
    cfg_wfcl_path: str = "configs/configs_3d/facl_wacl_lv3_r90p_r95p/facl_wacl_lv3_r90p_r95p.yaml",
    mu_tp:  float = 0.002479317132383585,
    sig_tp: float = 0.005952117739409208,
    n_bins: int = 50,
    dpi: int = 150,
):
    """
    0–1 mm/day 구간의 강수 분포를 Obs / MSE / FACL / WFCL 별로 히스토그램으로 그린다.

    상단 패널: 절대 빈도(count) 히스토그램
    하단 패널: 각 모델 히스토그램 – Obs 히스토그램 (차이)
    """
    import yaml
    from utils.save_utils import load_checkpoint_for_eval
    from datasets.dataloader import get_test_loader
    from models.get_model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg).to(device).float()
        mon_key  = cfg.get("monitor", {}).get("key", "val/mse")
        mon_name = mon_key.split("/")[-1]
        ckpt_dir = cfg["paths"]["ckpt_dir"]
        ckpt_path = cfg.get("ckpt", os.path.join(ckpt_dir, f"best_{mon_name}.pt"))
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "last.pt")
        load_checkpoint_for_eval(ckpt_path, model, device=device)
        model.eval()
        return model, cfg

    def _extract_xy(batch):
        if isinstance(batch, dict):
            if all(k in batch for k in ("X1", "X2", "Y")):
                return (batch["X1"].to(device), batch["X2"].to(device)), batch["Y"].to(device)
        x, y = batch
        return x.to(device), y.to(device)

    def _forward(model, x):
        if isinstance(x, (list, tuple)):
            return model(*x)
        return model(x)

    def _to_mmday(z):
        return np.maximum(z * sig_tp + mu_tp, 0.0) * 1000.0

    print("[0-1mm Hist] Loading models ...")
    model_mse,  cfg_mse  = _load_model(cfg_mse_path)
    model_facl, cfg_facl = _load_model(cfg_facl_path)
    model_wfcl, cfg_wfcl = _load_model(cfg_wfcl_path)

    test_loader = get_test_loader(cfg_mse)

    vals = {"Obs": [], "MSE": [], "FACL": [], "WFCL": []}

    print("[0-1mm Hist] Running inference ...")
    with torch.no_grad():
        for batch in test_loader:
            x, y = _extract_xy(batch)
            obs_mm  = _to_mmday(y.cpu().float().numpy()).ravel()
            mse_mm  = _to_mmday(_forward(model_mse,  x).cpu().float().numpy()).ravel()
            facl_mm = _to_mmday(_forward(model_facl, x).cpu().float().numpy()).ravel()
            wfcl_mm = _to_mmday(_forward(model_wfcl, x).cpu().float().numpy()).ravel()

            # 0-1 mm/day 구간만 추출 (0 포함, 1 이하)
            vals["Obs"].append(obs_mm[(obs_mm >= 0.0) & (obs_mm <= 1.0)])
            vals["MSE"].append(mse_mm[(mse_mm >= 0.0) & (mse_mm <= 1.0)])
            vals["FACL"].append(facl_mm[(facl_mm >= 0.0) & (facl_mm <= 1.0)])
            vals["WFCL"].append(wfcl_mm[(wfcl_mm >= 0.0) & (wfcl_mm <= 1.0)])

    for k in vals:
        vals[k] = np.concatenate(vals[k])

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    MODEL_STYLES = {
        "Obs":  dict(color="black",   lw=2.0, ls="-",  zorder=5, label="Obs"),
        "MSE":  dict(color="#1f77b4", lw=1.8, ls="--", zorder=4, label="MSE"),
        "FACL": dict(color="#ff7f0e", lw=1.8, ls="-.", zorder=3, label="FACL"),
        "WFCL": dict(color="#d62728", lw=1.8, ls=":",  zorder=3, label="WFCL"),
    }
    DIFF_STYLES = {
        "MSE":  dict(color="#1f77b4", lw=1.8, ls="--", label="MSE−Obs"),
        "FACL": dict(color="#ff7f0e", lw=1.8, ls="-.", label="FACL−Obs"),
        "WFCL": dict(color="#d62728", lw=1.8, ls=":",  label="WFCL−Obs"),
    }

    # 히스토그램 계산 (density=True: 확률 밀도)
    hists = {}
    for name, v in vals.items():
        h, _ = np.histogram(v, bins=bins, density=True)
        hists[name] = h

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8),
                                          gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle("Precipitation Distribution: 0–1 mm/day\n(FACL / WFCL vs MSE vs Obs)",
                 fontsize=14, fontweight="bold")

    # 상단: PDF
    for name, style in MODEL_STYLES.items():
        ax_top.plot(bin_centers, hists[name], **style)
    ax_top.set_xlabel("Precipitation (mm/day)", fontsize=12)
    ax_top.set_ylabel("Probability Density", fontsize=12)
    ax_top.set_xlim(0.0, 1.0)
    ax_top.legend(fontsize=11)
    ax_top.grid(True, alpha=0.35)

    # 하단: 차이 (model – Obs)
    for name, style in DIFF_STYLES.items():
        ax_bot.plot(bin_centers, hists[name] - hists["Obs"], **style)
    ax_bot.axhline(0, color="black", lw=1.2)
    ax_bot.set_xlabel("Precipitation (mm/day)", fontsize=12)
    ax_bot.set_ylabel("ΔPDF (Model − Obs)", fontsize=12)
    ax_bot.set_xlim(0.0, 1.0)
    ax_bot.legend(fontsize=11)
    ax_bot.grid(True, alpha=0.35)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hist_0to1mm_facl_wacl.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[0-1mm Hist] Saved → {out_path}")
    return out_path







