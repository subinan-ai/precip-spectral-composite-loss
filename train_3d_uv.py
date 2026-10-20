# train_3d.py
import os
from collections import defaultdict

import torch
from utils import Logger, save_checkpoint, set_seed
from utils.train_utils import build_optimizer, build_scheduler

from models.get_model import build_model
from metrics import get_metric_fns  
from losses import get_loss
from datasets.dataloader import get_loaders

def run(cfg, out_dir, logger):
    """
    진입점: main.py에서 호출 가정
    cfg: utils.config.load_config() 리턴
    out_dir: 실험 루트 디렉토리 (experiments/runs/<exp_name>)
    logger: get_logger(out_dir) 등 외부 로거 (여기서는 텍스트 로그만 사용)
    """
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_clip = float(cfg["trainer"].get("grad_clip", 0.0))
    accum_steps = int(cfg["trainer"].get("accumulate_steps", 1))
    log_interval = int(cfg["trainer"].get("log_interval", 50))
    val_interval = int(cfg["trainer"].get("val_interval", 1))
    save_per_metric = bool(cfg.get("save_per_metric", True))  # metric별 best 저장 여부

    # === 0) 준비 (경로/로거)
    tb_dir  = os.path.join(cfg["paths"]["log_dir"], "tensorboard")
    csv_path = os.path.join(cfg["paths"]["log_dir"], "train_val_metrics.csv")
    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    log = Logger(tb_dir, csv_path)

    # === 1) 데이터 
    train_loader, val_loader = get_loaders(cfg)  

    # === 2) 모델
    model = build_model(cfg).to(device)

    loss_name = cfg["loss"]["name"]
    loss_kwargs = cfg["loss"].get("kwargs", {})
    loss_fn = get_loss(cfg["model"]["version"], loss_name)
    metric_fns = get_metric_fns(cfg)

    optimizer = build_optimizer(cfg, model)    
    scheduler = build_scheduler(cfg, optimizer)  

    # === 3) 모니터/베스트 관리
    mon_key = cfg["monitor"]["key"]          # e.g., "val/phase"
    mon_mode = cfg["monitor"]["mode"]        # "min" | "max"
    mon_name = mon_key.split("/")[-1]        # e.g., "phase"
    best_main = float("inf") if mon_mode == "min" else -float("inf")
    best_scores = {}     # metric별 최고치

    # === 4) 루프 준비
    epochs = int(cfg["trainer"]["epochs"])
    total_steps = len(train_loader) * epochs
    global_step = 0

    logger.info(f"Start training (epochs={epochs}, loss={loss_name}, monitor={mon_key} {mon_mode})")

    try:
        for epoch in range(1, epochs + 1):
            # ----- Train -----
            model.train()
            running_loss = 0.0
            step_in_epoch = 0

            optimizer.zero_grad(set_to_none=True)

            for batch in train_loader:
                x1 = batch["X1"].to(device, non_blocking=True).float()
                x2 = batch["X2"].to(device, non_blocking=True).float()
                x3 = batch["X3"].to(device, non_blocking=True).float()
                x4 = batch["X4"].to(device, non_blocking=True).float()
                y  = batch["Y"].to(device,  non_blocking=True).float()

                pred = model(x1, x2, x3, x4)
                loss = _call_loss(loss_fn, loss_kwargs, pred=pred, y=y,
                                step=global_step, total_steps=total_steps)

                if torch.isnan(loss) or torch.isinf(loss):
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

                # scale by accum for stable LR (선택)
                loss.backward()

                step_in_epoch += 1
                global_step   += 1
                running_loss  += float(loss.item())

                if (step_in_epoch % accum_steps) == 0:
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if (global_step % log_interval) == 0:
                    avg_so_far = running_loss / max(1, step_in_epoch)
                    log.log_step(global_step, {"loss": avg_so_far}, prefix="train")

            # 잔여 그라디언트 처리(배치 수가 accum의 배수가 아닐 때)
            if (step_in_epoch % accum_steps) != 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss = running_loss / max(1, len(train_loader))

            if scheduler is not None:
                scheduler.step()

            # ----- Validation -----
            do_validate = (epoch % val_interval == 0)
            if not do_validate:
                continue

            model.eval()
            val_loss, metric_sum, n = 0.0, defaultdict(float), 0
            with torch.no_grad():
                for batch in val_loader:
                    x1 = batch["X1"].to(device, non_blocking=True).float()
                    x2 = batch["X2"].to(device, non_blocking=True).float()
                    x3 = batch["X3"].to(device, non_blocking=True).float()
                    x4 = batch["X4"].to(device, non_blocking=True).float()
                    y  = batch["Y"].to(device,  non_blocking=True).float()

                    pred = model(x1, x2, x3, x4)
                    loss = _call_loss(loss_fn, loss_kwargs, pred=pred, y=y,
                                    step=global_step, total_steps=total_steps)
                    val_loss += float(loss.item())

                    for name, fn in metric_fns.items():
                        val = fn(pred, y, **cfg["metrics"].get("params", {}))
                        metric_sum[name] += float(val.item() if hasattr(val, "item") else val)
                    n += 1

            val_loss = val_loss / max(1, n)
            val_metrics = {"train_loss": train_loss, "val_loss": val_loss}
            for name in metric_sum:
                val_metrics[f"val_{name}"] = metric_sum[name] / max(1, n)

            # ---- EPOCH LOG (TB + CSV) ----
            log.log_epoch(epoch, val_metrics)

            # ----- Best 저장: 대표 모니터 기준 -----
            main_current = val_metrics.get(f"val_{mon_name}")
            if main_current is not None:
                improved = (main_current < best_main) if mon_mode == "min" else (main_current > best_main)
                if improved:
                    best_main = main_current
                    save_path = os.path.join(ckpt_dir, f"best_{mon_name}.pt")
                    save_checkpoint(save_path, model, optimizer, epoch, val_metrics, cfg, scheduler)
                    logger.info(f"[BEST main] {mon_name}={main_current:.6f} @epoch {epoch}")

            # ----- (옵션) metric별 베스트 저장 -----
            if save_per_metric:
                for metric_key, current in val_metrics.items():
                    if not metric_key.startswith("val_") or metric_key in ("val_loss",):
                        continue

                    # 작을수록 좋은 metric
                    minimize = metric_key in ("val_mse", "val_mae", "val_rhd", "val_amp", "val_lpips")
                    better = (current < best_scores.get(metric_key, float("inf"))) if minimize \
                             else (current > best_scores.get(metric_key, -float("inf")))
                    if better:
                        best_scores[metric_key] = current
                        save_path = os.path.join(ckpt_dir, f"best_{metric_key}.pt")
                        save_checkpoint(save_path, model, optimizer, epoch, val_metrics, cfg, scheduler)

            # 항상 last 저장 
            if epoch % 5 == 0:
                save_checkpoint(os.path.join(ckpt_dir, f"epoch{epoch}.pt"), model, optimizer, epoch, val_metrics, cfg, scheduler)

    finally:
        log.close()
        logger.info("Training finished.")


def _call_loss(loss_fn, loss_kwargs, pred, y, step=None, total_steps=None):
    """
    다양한 인터페이스를 가진 loss_fn을 하나로 흡수:
      - return (per_t, loss, name) or (per_t, loss) or (loss, aux) or loss
      - 인자도 (pred, y) 또는 (pred, y, step, total_steps)
    """
    try:
        out = loss_fn(pred, y, step=step, total_steps=total_steps, **loss_kwargs)
    except TypeError:
        out = loss_fn(pred, y, **loss_kwargs)

    if isinstance(out, tuple):
        # (per_t, loss, name), (per_t, loss) 스타일
        if len(out) >= 2 and torch.is_tensor(out[1]):
            return out[1]
        # (loss, aux) 스타일
        if len(out) >= 1 and torch.is_tensor(out[0]):
            return out[0]
        raise RuntimeError("loss_fn 반환값을 해석할 수 없습니다.")
    elif torch.is_tensor(out):
        return out
    else:
        raise RuntimeError("loss_fn은 torch.Tensor를 포함하여 반환해야 합니다.")