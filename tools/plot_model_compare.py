# save as: tools/plot_compare_faceted.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR_DEFAULT = "results/sample_wise_out"

def load_vec(base_dir, name):
    npy = os.path.join(base_dir, f"{name}.npy")
    csv = os.path.join(base_dir, f"{name}.csv")
    if os.path.isfile(npy):
        v = np.load(npy).reshape(-1)
        return v[~np.isnan(v)]
    if os.path.isfile(csv):
        df = pd.read_csv(csv)
        col = name if name in df.columns else df.columns[0]
        v = df[col].to_numpy().reshape(-1)
        return v[~np.isnan(v)]
    raise FileNotFoundError(f"Missing {name}.npy/.csv in {base_dir}")

def mean_ci90(x):
    """t-기반 90% CI (평균의 신뢰구간)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    m = x.mean()
    s = x.std(ddof=1) if n > 1 else 0.0
    try:
        from scipy.stats import t
        tcrit = t.ppf(0.95, max(n-1,1))
    except Exception:
        tcrit = 1.645
    se = s / np.sqrt(max(n,1))
    return m, (m - tcrit*se, m + tcrit*se)

def mean_ci_bootstrap(x, level=0.90, iters=10000, seed=None):
    """
    부트스트랩 percentile CI (평균의 신뢰구간).
    level: 0~1, iters: 재표본 횟수
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan, (np.nan, np.nan)
    m = x.mean()
    if n == 1:  # 표본 1개면 CI의 의미가 없으므로 그대로 반환
        return m, (m, m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(iters), n))
    bs_means = x[idx].mean(axis=1)
    q_low  = (1 - level) / 2 * 100.0
    q_high = (1 + level) / 2 * 100.0
    lo, hi = np.percentile(bs_means, [q_low, q_high])
    return m, (lo, hi)

import numpy as np
import matplotlib.pyplot as plt
import os

def mean_ci_bootstrap(x, level=0.95, iters=10000, seed=0):
    """
    Bootstrap percentile CI
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(iters):
        sample = rng.choice(x, size=n, replace=True)
        boots.append(sample.mean())
    lo = np.percentile(boots, (1 - level) / 2 * 100)
    hi = np.percentile(boots, (1 + level) / 2 * 100)
    return x.mean(), (lo, hi)


def plot_facets(out_dir,
                metrics=("mae","fss","lpips"),
                labels=("MSE","FACL+WACL (lv3)"),
                save_name="comparison_facets_ci.png",
                err_mode="boot",         # 'boot' | 'ci90' | 'std' | 'pctl'
                p_low=10, p_high=90,
                boot_level=0.90, boot_iters=10000, boot_seed=0,
                ylim_mode="tight_ci",    # 'tight_ci' | 'zero_based' | 'auto'
                pad_frac=0.12,           
                show_values=True):       
    """
    err_mode:
      - 'boot' : bootstrap CI
      - 'ci90' : t-분포 기반 90% CI
      - 'std'  : ±1 표준편차
      - 'pctl' : 분위수 범위
    """

    os.makedirs(out_dir, exist_ok=True)

    # ---- 데이터 로드
    data = {}
    for m in metrics:
        try:
            a = load_vec(out_dir, f"{m}_mse")
            b = load_vec(out_dir, f"{m}_facl")
            data[m] = (a, b)
        except FileNotFoundError:
            print(f"[warn] skip {m}: file not found")
    if not data:
        raise RuntimeError("No metric files found.")

    # ---- 통계 헬퍼
    def mean_and_err(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        if err_mode == "std":
            m = x.mean(); s = x.std(ddof=1)
            return m, (m - s, m + s)
        elif err_mode == "pctl":
            m = x.mean(); lo, hi = np.percentile(x, [p_low, p_high])
            return m, (lo, hi)
        elif err_mode == "ci90":
            return mean_ci90(x)
        else:  # 'boot'
            return mean_ci_bootstrap(x, level=boot_level, iters=boot_iters, seed=boot_seed)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]

    fig, axes = plt.subplots(1, len(data), figsize=(4.8*len(data), 3.8), constrained_layout=False)
    if len(data) == 1:
        axes = [axes]
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.88])

    for ax, (m, (a, b)) in zip(axes, data.items()):
        m1, (l1, h1) = mean_and_err(a)
        m2, (l2, h2) = mean_and_err(b)

        # 막대
        w = 0.65
        ax.bar([0], [m1], width=w, color=colors[0], zorder=2, label=labels[0])
        ax.bar([1], [m2], width=w, color=colors[1], zorder=2, label=labels[1])

        
        # 에러바 (강조 스타일)
        if np.isfinite(m1):
            ax.errorbar([0], [m1], yerr=[[m1-l1],[h1-m1]],
                        fmt="none", ecolor="k", elinewidth=2.5,
                        capsize=10, capthick=2.5, zorder=3)
        if np.isfinite(m2):
            ax.errorbar([1], [m2], yerr=[[m2-l2],[h2-m2]],
                        fmt="none", ecolor="k", elinewidth=2.5,
                        capsize=10, capthick=2.5, zorder=3)

        # 축/제목
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        arrow = "↑ better" if m.lower()=="fss" else "↓ better"
        ax.set_title(f"{m.upper()}  ({arrow})")
        ax.grid(axis="y", alpha=0.25, zorder=0)

        # y축 설정
        if ylim_mode == "tight_ci":
            low  = float(np.nanmin([l1, l2]))
            high = float(np.nanmax([h1, h2]))
            if np.isfinite(low) and np.isfinite(high) and high > low:
                span = high - low
                pad = pad_frac * span
                ax.set_ylim(low - pad*0.3, high + pad)
        elif ylim_mode == "zero_based":
            high = float(np.nanmax([h1, h2, m1, m2]))
            if np.isfinite(high):
                pad = pad_frac * max(high, 1e-12)
                ax.set_ylim(0.0, high + pad)

        # Δ 주석
        if np.isfinite(m1) and np.isfinite(m2):
            delta = m2 - m1
            ax.text(0.5, 0.95, f"Δ={delta:.5g}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9)

        # 값 라벨
        if show_values:
            if np.isfinite(m1): ax.text(0, m1, f"{m1:.4g}", ha="center", va="bottom", fontsize=8)
            if np.isfinite(m2): ax.text(1, m2, f"{m2:.4g}", ha="center", va="bottom", fontsize=8)

    # 공통 레전드/제목
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False)
    tag = {
        "boot": f"{int(boot_level*100)}% bootstrap CI (B={boot_iters})",
        "ci90": "90% CI of mean",
        "std": "±1 SD",
        "pctl": f"{p_low}–{p_high} percentile",
    }[err_mode]
    fig.suptitle(f"MSE vs FACL+WACL (lv3) — {tag}", fontsize=13, y=0.985)

    out_path = os.path.join(out_dir, save_name)
    fig.savefig(out_path, dpi=300)   # 논문용 → 300 dpi
    print(f"[ok] saved {out_path}")


def plot_delta_only(out_dir=OUT_DIR_DEFAULT,
                    metrics=("mae","fss","lpips"),
                    save_name="comparison_delta_ci.png",
                    err_mode="boot",
                    p_low=10, p_high=90,
                    boot_level=0.90, boot_iters=10000, boot_seed=0):
    """
    Δ = FACL − MSE 에 대한 에러바(부트스트랩/CI/SD/분위수) 막대.
    """
    os.makedirs(out_dir, exist_ok=True)
    cats, means, lo, hi = [], [], [], []

    def mean_and_err_for_delta(d):
        d = np.asarray(d, float); d = d[~np.isnan(d)]
        if err_mode == "std":
            m = d.mean(); s = d.std(ddof=1)
            return m, (m - s, m + s)
        elif err_mode == "pctl":
            m = d.mean(); lo, hi = np.percentile(d, [p_low, p_high])
            return m, (lo, hi)
        elif err_mode == "ci90":
            return mean_ci90(d)
        else:
            return mean_ci_bootstrap(d, level=boot_level, iters=boot_iters, seed=boot_seed)

    for m in metrics:
        try:
            a = load_vec(out_dir, f"{m}_mse")
            b = load_vec(out_dir, f"{m}_facl")
        except FileNotFoundError:
            print(f"[warn] skip {m}: file not found")
            continue
        d = b - a
        md, (ld, hd) = mean_and_err_for_delta(d)
        cats.append(m.upper())
        means.append(md)
        lo.append(md - ld)
        hi.append(hd - md)
    if not cats:
        raise RuntimeError("No metric files found.")

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(4.2*len(cats), 3.2), constrained_layout=True)
    ax.bar(x, means, yerr=[lo, hi], width=0.65, capsize=6, ecolor="k", linewidth=1.1, zorder=2)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    tag = {
        "boot": f"{int(boot_level*100)}% bootstrap CI",
        "ci90": "90% CI",
        "std": "±1 SD",
        "pctl": f"{p_low}–{p_high} pct",
    }[err_mode]
    ax.set_ylabel("Δ (FACL − MSE)")
    ax.set_title(f"Difference ({tag})")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    out_path = os.path.join(out_dir, save_name)
    fig.savefig(out_path, dpi=220)
    print(f"[ok] saved {out_path}")

if __name__ == "__main__":
    out_dir = OUT_DIR_DEFAULT
    metrics = ("mae", "fss", "lpips")

    plot_facets(out_dir,
            metrics=("mae","fss","lpips"),
            err_mode="boot",
            boot_level=0.99, boot_iters=10000, boot_seed=0,
            ylim_mode="tight_ci",
            save_name="facets_boot_tight.png")



    plot_delta_only(out_dir, metrics=metrics,
                    err_mode="boot", boot_level=0.90, boot_iters=10000, boot_seed=0,
                    save_name="delta_boot.png")

    # 참고: 필요시 다른 표현도 가능
    # plot_facets(out_dir, metrics=metrics, err_mode="ci90", ylim_mode="zero_based",
    #             save_name="facets_ci_zero.png")
    # plot_facets(out_dir, metrics=metrics, err_mode="std",  ylim_mode="tight_ci",
    #             save_name="facets_sd_tight.png")
