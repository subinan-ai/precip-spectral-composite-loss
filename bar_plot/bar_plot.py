# plot_metrics_grid.py
# VSCode에서 실행: python plot_metrics_grid.py
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 설정
# --------------------------
FILE_MSE  = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_uv_mse/logs/train_val_metrics.csv"
FILE_FACL = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_uv_facl/logs/train_val_metrics.csv"

DISPLAY_ORDER = [
    ("RHD",       "rhd"),
    ("LPIPS",     "lpips"),
    ("CPBD",      "cpbd"),
    ("CSI-r90p",  "csi_r90p"),
    ("CSI-r95p",  "csi_r95p"),
    ("Phase",     "phase"),
    ("Amp",       "amp"),
]

SYNONYMS = {
    "epoch":    ["epoch", "ep"],
    "lpips":    ["lpips", "val_lpips", "lpips_val", "lpips_mean", "val_lpips_mean"],
    "rhd":      ["rhd", "regional_histogram_divergence"],
    "cpbd":     ["cpbd", "val_cpbd", "cpbd_mean"],
    "csi_r90p": ["csi-r90p", "csi_r90p", "csi90p", "csi_r90", "csi90"],
    "csi_r95p": ["csi-r95p", "csi_r95p", "csi95p", "csi_r95", "csi95"],
    "phase":    ["phase", "phase_sim", "phase_similarity", "pearson_phase", "phase_corr"],
    "amp":      ["amp", "amplitude_mse", "amplitude", "amp_mse"],
}

# 표준편차/분산 후보 접미사
STD_SUFFIXES = ["_std", "_stdev", "_stddev"]
VAR_SUFFIXES = ["_var", "_variance"]

# --- 막대폭과 에러바 스타일 (요청 반영) ---
BAR_WIDTH = 0.38
ERROR_KW = dict(ecolor="black", elinewidth=1.6, capsize=5, capthick=1.6)

# --------------------------
# 유틸
# --------------------------
def find_col(df: pd.DataFrame, keys):
    cols = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in cols:
            return cols[k]
    for k in keys:
        k2 = k.replace("_", "").replace("-", "")
        for lc, orig in cols.items():
            if k2 in lc.replace("_", "").replace("-", ""):
                return orig
    return None

def find_stat_col(df: pd.DataFrame, base_key: str):
    """값 컬럼을 기준으로 *_std 또는 *_var 열을 찾아 반환."""
    val_col = find_col(df, SYNONYMS.get(base_key, [base_key]))
    if val_col is None:
        return None, None

    base = val_col
    candidates = []
    for suf in STD_SUFFIXES:
        candidates.append(base + suf)
    for suf in VAR_SUFFIXES:
        candidates.append(base + suf)

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c_lower = cand.lower()
        if c_lower in cols_lower:
            c_name = cols_lower[c_lower]
            is_var = any(c_lower.endswith(s.lower()) for s in VAR_SUFFIXES)
            return c_name, ("var" if is_var else "std")
    return None, None

def get_value_and_std_at_index(df: pd.DataFrame, idx: int, metric_key: str):
    """특정 행(idx)에서 metric 값과 (분산→)표준편차를 반환."""
    val_col = find_col(df, SYNONYMS.get(metric_key, [metric_key]))
    if val_col is None:
        return np.nan, 0.0
    val = float(df.loc[idx, val_col])

    std_col, kind = find_stat_col(df, metric_key)
    if std_col is None:
        std = 0.0
    else:
        stat_val = float(df.loc[idx, std_col])
        std = math.sqrt(stat_val) if kind == "var" else stat_val
    return val, std

def select_lpips_min_idx(df: pd.DataFrame):
    col_lpips = find_col(df, SYNONYMS["lpips"])
    if col_lpips is None:
        raise ValueError(f"LPIPS 컬럼을 찾을 수 없습니다. 존재 열: {list(df.columns)}")
    return df[col_lpips].astype(float).idxmin()

def load_model_row(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    idx = select_lpips_min_idx(df)
    return df, idx

# --------------------------
# 데이터 로드
# --------------------------
df_mse,  idx_mse  = load_model_row(FILE_MSE)
df_facl, idx_facl = load_model_row(FILE_FACL)

# --------------------------
# 플롯
# --------------------------
def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.ravel()

    color_mse  = "#9e9e9e"   # 회색
    color_facl = "#42a5f5"   # 파란색

    for i, (disp_name, key) in enumerate(DISPLAY_ORDER):
        ax = axes[i]

        val_mse,  std_mse  = get_value_and_std_at_index(df_mse,  idx_mse,  key)
        val_facl, std_facl = get_value_and_std_at_index(df_facl, idx_facl, key)

        x = np.array([0, 1], dtype=float)
        # 막대 간격을 넓히고 각 막대 폭을 줄이기 위해 offset 사용
        bars = []
        bars.append(ax.bar(x[0] - BAR_WIDTH/2, val_mse,  width=BAR_WIDTH,
                           yerr=std_mse,  error_kw=ERROR_KW,
                           color=color_mse,  edgecolor="black", label="MSE"))
        bars.append(ax.bar(x[1] - BAR_WIDTH/2, val_facl, width=BAR_WIDTH,
                           yerr=std_facl, error_kw=ERROR_KW,
                           color=color_facl, edgecolor="black", label="FACL"))

        ax.set_xticks(x, ["MSE", "FACL"])
        ax.set_title(disp_name, fontsize=12)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        # 막대 위 값 표시
        for b, v in zip([bars[0][0], bars[1][0]], [val_mse, val_facl]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # 범례 전용 패널
    ax_leg = axes[-1]
    ax_leg.axis("off")
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_mse,  edgecolor="black", label="MSE (LPIPS-min epoch)"),
        Patch(facecolor=color_facl, edgecolor="black", label="FACL (LPIPS-min epoch)"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", fontsize=11, frameon=True)

    fig.suptitle("Metrics @ LPIPS-min Epoch (MSE vs FACL)", fontsize=15)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])

    out_path = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/bar_plot/metrics_grid_lpipsmin.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
