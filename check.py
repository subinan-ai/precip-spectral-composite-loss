from netCDF4 import Dataset, num2date

nc_path = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/data/1979_2023_MaySep_prec.nc"

with Dataset(nc_path, "r") as nc:
    print("Variables:", list(nc.variables.keys()))

    time_var = nc.variables["valid_time"]
    print("time units:", getattr(time_var, "units", None))
    print("time calendar:", getattr(time_var, "calendar", "standard"))

    times = num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )

idx = 6488                     
t = times[idx]

print("Index:", idx)
print("Datetime:", t)

# ✅ cftime 객체에서도 안전한 '날짜만' 출력
print("Date only:", f"{t.year:04d}-{t.month:02d}-{t.day:02d}")

'''
"""
Debug script to verify:
1) pred / GT shapes during evaluation
2) whether T dimension corresponds to lead time (1/2/3-day)
3) whether lead_idx selection is consistent

Run once with a trained checkpoint.
"""
print("🔥 DEBUG SCRIPT STARTED 🔥")

import torch
from utils.config_utils import load_config
from utils.save_utils import load_checkpoint_for_eval
from datasets.dataloader import get_test_loader
from models.get_model import build_model


@torch.no_grad()
def check_leadtime_shapes(
    cfg_path: str,
    ckpt_path: str,
    device: str = "cuda",
    lead_offsets=(1, 2, 3),
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load config & model
    # ------------------------------------------------------------------
    cfg = load_config(cfg_path)
    model = build_model(cfg).to(device).eval()
    load_checkpoint_for_eval(ckpt_path, model, device=device)

    # ------------------------------------------------------------------
    # 2. Build test loader with MULTI-lead GT
    # ------------------------------------------------------------------
    cfg_dbg = cfg.copy()
    cfg_dbg.setdefault("data", {})
    cfg_dbg["data"]["label_offsets"] = list(lead_offsets)

    loader = get_test_loader(cfg_dbg)

    # ------------------------------------------------------------------
    # 3. Take ONE batch only
    # ------------------------------------------------------------------
    batch = next(iter(loader))
    x1 = batch["X1"].to(device).float()
    x2 = batch["X2"].to(device).float()
    y  = batch["Y"].to(device).float()

    # ------------------------------------------------------------------
    # 4. Forward
    # ------------------------------------------------------------------
    pred = model(x1, x2)

    # ------------------------------------------------------------------
    # 5. Print shapes
    # ------------------------------------------------------------------
    print("\n================= SHAPE CHECK =================")
    print(f"X1 shape  : {tuple(x1.shape)}")
    print(f"X2 shape  : {tuple(x2.shape)}")
    print(f"GT shape  : {tuple(y.shape)}")
    print(f"PRED shape: {tuple(pred.shape)}")

    assert pred.shape == y.shape, \
        "❌ pred / GT shape mismatch! Lead-time semantics may be broken."

    B, C, T, H, W = pred.shape
    print(f"\n✔ Batch={B}, Channel={C}, Lead(T)={T}, H={H}, W={W}")

    # ------------------------------------------------------------------
    # 6. Check lead-wise alignment numerically (rough)
    # ------------------------------------------------------------------
    print("\n============= LEAD-WISE LOSS CHECK =============")
    for t in range(T):
        mse_t = torch.mean((pred[:, :, t] - y[:, :, t]) ** 2).item()
        print(f"Lead t={t} (offset={lead_offsets[t]}-day): MSE = {mse_t:.6e}")

    print("\n✔ If each t corresponds to a different lead time, "
          "loss values should differ meaningfully.\n")

    print("✅ Lead-time semantics look CONSISTENT.")
if __name__ == "__main__":
    print("🚀 ENTERED MAIN 🚀")

    CFG_PATH  = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/configs/configs_3d/mse_r90p_r95p/mse_r90p_r95p.yaml"
    CKPT_PATH = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/experiments/subin_an/conv3d_mse_r90p_r95p/checkpoints/best_val_lpips.pt"

    check_leadtime_shapes(
        cfg_path=CFG_PATH,
        ckpt_path=CKPT_PATH,
        device="cuda",
        lead_offsets=(1, 2, 3),
    )
'''