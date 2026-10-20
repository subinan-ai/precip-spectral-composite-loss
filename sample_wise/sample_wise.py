"""
sample_wise/sample_wise.py
==================================
- 두 checkpoint(MSE vs FACL+WACL) 로드 (파일 또는 디렉터리)
- 디렉터리인 경우 LPIPS 최소 epoch의 체크포인트를 자동 선택
- test 데이터셋 전체 inference
- config.metrics.names(또는 use_*)에 지정된 메트릭들을 샘플 단위로 계산
- NPY/CSV/JSON 저장 (+ 실행 정보 run_info.json 항상 저장)
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from utils.save_utils import load_checkpoint_for_eval
from models.get_model import build_model
from datasets.dataloader import get_test_loader
from utils.config_utils import load_config, dump_config

# metrics: 메타까지 함께 받는 API 우선 시도, 실패 시 기본 API 사용
try:
    from metrics.get_metrics import get_metric_fns_with_info as _get_metrics_api
    _GETS_INFO = True
except Exception:
    from metrics.get_metrics import get_metric_fns as _get_metrics_api  # type: ignore
    _GETS_INFO = False


# ----------------------- helpers -----------------------
def pick_xy_from_batch(batch):
    """기존 파이프라인과 동일한 키/순서로 입력을 추출."""
    if isinstance(batch, dict):
        norm = {str(k).strip().lower(): k for k in batch.keys()}
        for ks in (("x1", "x2", "y"), ("X1", "X2", "Y")):
            if all(k.lower() in norm for k in ks):
                return batch[norm[ks[0].lower()]], batch[norm[ks[1].lower()]], batch[norm[ks[2].lower()]]
        raise KeyError(f"[pick_xy_from_batch] Cannot parse keys: {list(batch.keys())}")
    if isinstance(batch, (list, tuple)):
        tens = [it for it in batch if torch.is_tensor(it)]
        if len(tens) >= 3:
            return tens[0], tens[1], tens[-1]
        raise ValueError(f"[pick_xy_from_batch] Need >=3 tensors, got {len(tens)}")
    raise TypeError(f"[pick_xy_from_batch] Unsupported type={type(batch)}")


def reduce_time_dim(x: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """
    (B,C,T,H,W) → (B,C,H,W) 로 통일.
    """
    if x.dim() == 5:
        if how == "mean":
            return x.mean(dim=2)
        elif how == "last":
            return x[:, :, -1]
        else:
            raise ValueError("--time_reduce must be 'mean' or 'last'")
    elif x.dim() == 4:
        return x
    elif x.dim() == 3:
        return x.unsqueeze(1)
    else:
        raise ValueError(f"[reduce_time_dim] Unexpected shape {tuple(x.shape)}")


def _to_numpy_1d(v, batch_size: int):
    """
    다양한 반환 타입을 (B,) numpy 로 표준화.
    """
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        if v.dim() == 1 and v.shape[0] == batch_size:
            return v.numpy()
        return v.view(batch_size, -1).mean(dim=1).numpy()
    if isinstance(v, (float, int)):
        return np.full(batch_size, float(v), dtype=np.float32)
    if isinstance(v, (list, tuple)):
        return np.asarray(v)
    return None


# ---------- checkpoint selection by LPIPS ----------
def _safe_torchload_meta(path: str) -> Dict[str, Any]:
    """
    체크포인트에서 epoch/val_lpips 같은 메타를 최대한 안전하게 추출.
    """
    meta = {}
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            # 흔한 키 패턴들 탐색
            for k in ["epoch", "global_epoch", "e", "iter", "step"]:
                if k in ckpt and isinstance(ckpt[k], (int, float)):
                    meta["epoch"] = int(ckpt[k])
                    break
            # val_lpips 후보
            candidates = [
                ("val_lpips", ckpt.get("val_lpips")),
                ("metrics.val_lpips", ckpt.get("metrics", {}).get("val_lpips") if isinstance(ckpt.get("metrics"), dict) else None),
                ("best_val_lpips", ckpt.get("best_val_lpips")),
                ("valid_lpips", ckpt.get("valid_lpips")),
            ]
            for _, val in candidates:
                if isinstance(val, (int, float)):
                    meta["val_lpips"] = float(val)
                    break
    except Exception:
        pass
    return meta


_RE_EPOCH = re.compile(r"(?:epoch[_=]?|e)(\d+)", re.IGNORECASE)
_RE_LPIPS = re.compile(r"lpips[_=]?([0-9]*\.?[0-9]+)", re.IGNORECASE)

def _parse_meta_from_name(path: str) -> Dict[str, Any]:
    base = os.path.basename(path)
    meta: Dict[str, Any] = {}
    m = _RE_EPOCH.search(base)
    if m:
        try: meta["epoch"] = int(m.group(1))
        except Exception: pass
    m = _RE_LPIPS.search(base)
    if m:
        try: meta["val_lpips"] = float(m.group(1))
        except Exception: pass
    return meta


def _pick_best_lpips_ckpt(ckpt_dir: str) -> Tuple[str, Dict[str, Any]]:
    """
    디렉터리에서 LPIPS 최소 체크포인트 선택.
    반환: (선택된 경로, 추정메타{epoch/val_lpips})
    """
    # 1) 고정 이름 우선
    for name in ["best_val_lpips.pt", "best_lpips.pt"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            meta = _safe_torchload_meta(p) or _parse_meta_from_name(p)
            return p, meta

    # 2) 후보 수집
    pts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No .pt in {ckpt_dir}")

    best_path = None
    best_val = None
    best_meta = {}

    # 2-1) 파일 내용에서 val_lpips 읽기
    for p in pts:
        meta = _safe_torchload_meta(p)
        if "val_lpips" in meta:
            v = meta["val_lpips"]
            if best_val is None or v < best_val:
                best_val, best_path, best_meta = v, p, meta

    if best_path:
        return best_path, best_meta

    # 2-2) 파일명에서 lpips=... 파싱
    for p in pts:
        meta = _parse_meta_from_name(p)
        if "val_lpips" in meta:
            v = meta["val_lpips"]
            if best_val is None or v < best_val:
                best_val, best_path, best_meta = v, p, meta

    if best_path:
        return best_path, best_meta

    # 3) 최종 폴백: 수정시간 최신
    pts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    p = pts[0]
    meta = _safe_torchload_meta(p) or _parse_meta_from_name(p)
    if "val_lpips" not in meta:
        meta["val_lpips"] = None
    return p, meta


def _resolve_ckpt(path_or_dir: str, prefer: str = "lpips") -> Tuple[str, Dict[str, Any]]:
    """
    인자가 파일이면 그대로, 디렉터리면 prefer 기준으로 선택.
    """
    if os.path.isdir(path_or_dir):
        if prefer == "lpips":
            return _pick_best_lpips_ckpt(path_or_dir)
        else:
            raise ValueError(f"Unknown prefer mode: {prefer}")
    if os.path.isfile(path_or_dir):
        meta = _safe_torchload_meta(path_or_dir) or _parse_meta_from_name(path_or_dir)
        return path_or_dir, meta
    raise FileNotFoundError(f"Checkpoint not found: {path_or_dir}")


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mse_ckpt", required=True,
                    help="파일(.pt) 또는 디렉터리(자동 선택)")
    ap.add_argument("--facl_ckpt", required=True,
                    help="파일(.pt) 또는 디렉터리(자동 선택)")
    ap.add_argument("--select_by", choices=["lpips"], default="lpips",
                    help="디렉터리 주어졌을 때 선택 기준")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/sample_wise_out")
    ap.add_argument("--debug_keys", action="store_true")
    ap.add_argument("--time_reduce", choices=["mean", "last"], default="mean",
                    help="입력이 5D(B,C,T,H,W)일 때 T를 4D로 줄이는 방식")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) config 로드/저장
    cfg = load_config(args.config)
    dump_config(cfg, args.out)

    # 2) 체크포인트 경로 확정 (파일이면 그대로, 디렉터리면 LPIPS 최솟값 자동 선택)
    mse_ckpt_path, mse_meta  = _resolve_ckpt(args.mse_ckpt,  prefer=args.select_by)
    facl_ckpt_path, facl_meta = _resolve_ckpt(args.facl_ckpt, prefer=args.select_by)
    print(f"[INFO] MSE ckpt:  {mse_ckpt_path}  meta={mse_meta}")
    print(f"[INFO] FACL ckpt: {facl_ckpt_path} meta={facl_meta}")

    # 3) 모델 준비
    device = torch.device(args.device)
    m_mse = build_model(cfg).to(device)
    m_facl = build_model(cfg).to(device)
    load_checkpoint_for_eval(mse_ckpt_path,  m_mse,  device=device)
    load_checkpoint_for_eval(facl_ckpt_path, m_facl, device=device)
    m_mse.eval(); m_facl.eval()

    # 4) 데이터 로더
    loader = get_test_loader(cfg)

    # 5) 메트릭 함수들 로드 (+메타)
    if _GETS_INFO:
        metric_fns, metrics_info = _get_metrics_api(cfg)  # dict(name->fn), meta
    else:
        metric_fns = _get_metrics_api(cfg)                # dict(name->fn)
        metrics_info = {"selected_names": list(metric_fns.keys())}

    metric_names = list(metric_fns.keys())
    print("[INFO] metrics used:", metric_names)

    # 6) 결과 컨테이너 동적 생성
    acc: Dict[str, list] = {}
    for m in metric_names:
        acc[f"{m}_mse"] = []
        acc[f"{m}_facl"] = []

    num_batches = 0
    num_samples = 0

    # 7) 메인 루프
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.debug_keys:
                print("[DEBUG] keys:", list(batch.keys()) if isinstance(batch, dict) else type(batch))
                return

            x1, x2, y = pick_xy_from_batch(batch)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = y.to(device,  non_blocking=True)

            x1 = reduce_time_dim(x1, how=args.time_reduce)
            x2 = reduce_time_dim(x2, how=args.time_reduce)
            y4 = reduce_time_dim(y,  how=args.time_reduce)

            pm = m_mse(x1, x2)
            pf = m_facl(x1, x2)

            pm4 = reduce_time_dim(pm, how=args.time_reduce)
            pf4 = reduce_time_dim(pf, how=args.time_reduce)

            B = pm4.shape[0]
            num_batches += 1
            num_samples += int(B)

            for mname, mfn in metric_fns.items():
                try:
                    vm = mfn(pm4, y4)
                    vf = mfn(pf4, y4)
                except Exception as e:
                    print(f"[ERROR] metric '{mname}' failed at batch {i}: {e}")
                    raise

                vm_np = _to_numpy_1d(vm, B)
                vf_np = _to_numpy_1d(vf, B)

                if vm_np is not None:
                    acc[f"{mname}_mse"].append(vm_np)
                if vf_np is not None:
                    acc[f"{mname}_facl"].append(vf_np)

    # 8) 실행 정보 저장(항상)
    run_info: Dict[str, Any] = {
        "metrics": metric_names,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "device": str(device),
        "checkpoints": {
            "mse":  {"path": mse_ckpt_path,  **mse_meta},
            "facl": {"path": facl_ckpt_path, **facl_meta},
            "select_by": args.select_by
        }
    }
    if metrics_info:
        run_info["metrics_info"] = metrics_info
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    # 9) numpy로 합치기
    for k, v in acc.items():
        acc[k] = np.concatenate(v) if len(v) > 0 else np.array([])

    # 10) 개별 NPY/CSV 저장
    for mname in metric_names:
        for which in ("mse", "facl"):
            key = f"{mname}_{which}"
            vals = acc.get(key, np.array([]))
            if vals.size > 0:
                np.save(os.path.join(args.out, f"{key}.npy"), vals)
                pd.DataFrame({key: vals}).to_csv(os.path.join(args.out, f"{key}.csv"), index=False)

    # 11) 합본 CSV (있을 때만)
    df_cols = {k: v for k, v in acc.items() if v.size > 0}
    if len(df_cols) > 0:
        maxlen = max(len(v) for v in df_cols.values())
        df = pd.DataFrame({
            k: (np.pad(v, (0, maxlen - len(v)), constant_values=np.nan) if len(v) < maxlen else v)
            for k, v in df_cols.items()
        })
        df.to_csv(os.path.join(args.out, "metrics_all.csv"), index=False)

    # 12) summary.json은 항상 저장 (없으면 null)
    summary = {k: (float(np.nanmean(v)) if v.size > 0 else None) for k, v in acc.items()}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("DONE. Results in", args.out)


if __name__ == "__main__":
    main()
