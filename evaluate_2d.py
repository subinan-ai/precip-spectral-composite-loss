# evaluate_2d.py
import os
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import set_seed
from utils.save_utils import load_checkpoint_for_eval
from models.get_model import build_model
from metrics import get_metric_fns
from datasets.dataloader import get_test_loader

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

    # 6) 시각화 (첫 샘플)
    if len(preds) > 0:
        vis_dir = cfg["paths"]["eval_dir"]
        os.makedirs(vis_dir, exist_ok=True)

        exp = cfg.get("experiment_name", cfg.get("exp_name", "exp"))
        loss_name = cfg["loss"]["name"]
        base_time = str(datetimes[0]) if len(datetimes) else "unknown"
        save_path = os.path.join(vis_dir, f"{exp}_{loss_name}_{base_time}_ep{epoch}.png")

        visualize_prediction(
            pred=preds[0],
            true=targets[0],
            base_time=base_time,
            title_suffix=f"[{exp} | {loss_name}]",
            save_path=save_path,
            vmin_percentile=1.0,
            vmax_percentile=99.0,
            cmap="plasma",
        )
        logger.info(f"[Eval-2D] Saved visualization → {save_path}")
    else:
        logger.warning("[Eval-2D] No predictions collected; skipping visualization.")

    return avg_metrics


@torch.no_grad()
def evaluate_epoch(model, loader, metric_fns, device, metric_params=None):
    """
    metric_fns: dict[str -> callable], (pred, y, **metric_params) -> float (or scalar tensor)
    반환: (평균 metric dict, preds list[tensor], targets list[tensor], datetimes list[str])
    """
    model.eval()
    metrics = defaultdict(float)
    count = 0
    preds, targets, datetimes = [], [], []

    for batch in loader:
        x1 = batch["X1"].to(device, non_blocking=True).float()
        x2 = batch["X2"].to(device, non_blocking=True).float()
        y  = batch["Y"].to(device,  non_blocking=True).float()
        dt = batch.get("time", None)

        pred = model(x1, x2)

        # metric 계산
        for name, fn in metric_fns.items():
            try:
                val = fn(pred, y, **(metric_params or {}))
            except TypeError:
                val = fn(pred, y)
            val = float(val.item()) if hasattr(val, "item") else float(val)
            if not torch.isfinite(torch.tensor(val)):
                val = 0.0
            metrics[name] += val

        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())
        if dt is not None:
            if isinstance(dt, (list, tuple)):
                datetimes.extend([str(d).split("T")[0] for d in dt])
            else:
                datetimes.append(str(dt).split("T")[0])

        count += 1

    avg_metrics = {k: v / max(1, count) for k, v in metrics.items()}
    return avg_metrics, preds, targets, datetimes


def visualize_prediction(pred, true, base_time="", title_suffix="", save_path=None,
                         vmin_percentile=1.0, vmax_percentile=99.0, cmap="plasma"):
    """
    pred/true: Tensor (B, C=1, T, H, W) 가정. 첫 배치만 시각화.
    """
    pred = pred[0].numpy()
    true = true[0].numpy()
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