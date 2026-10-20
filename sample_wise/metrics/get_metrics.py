# metrics/get_metrics.py
from functools import partial

# 버전에 따라 2D/3D 구현 선택 (ALL_METRICS, csi/fss 등 함수가 들어있어야 함)
def _import_metric_impl(version: str):
    v = str(version).lower()
    if v == "3d" or v == "3d_uv":
        from .metric_utils_3d import ALL_METRICS, csi, fss, rhd, get_lpips_model, lpips
    elif v == "2d":
        from .metric_utils_2d import ALL_METRICS, csi, fss, rhd, get_lpips_model, lpips
    else:
        raise ValueError(f"Unsupported model.version: {version}")
    return ALL_METRICS, csi, fss, rhd, get_lpips_model, lpips

def get_metric_fns(config):
    """
    최종 메트릭 선택 규칙:
      1) config.metrics.names가 있으면 그 목록을 우선 사용.
         단, 동일 이름의 use_* 플래그가 있으면 False인 항목은 제외.
      2) names가 없으면 use_*: true 항목들로 자동 구성.
      3) 아무것도 없으면 ['mse'].
    """
    version = config["model"]["version"]
    ALL_METRICS, csi_fn, fss_fn, rhd_fn, get_lpips_model, lpips_fn = _import_metric_impl(version)

    mcfg = config.get("metrics", {})
    names_cfg = list(mcfg.get("names", []) or [])
    # legacy use_* 플래그 수집
    use_true = [k for k, v in mcfg.items() if k.startswith("use_") and bool(v)]

    # names 우선, use_*가 있으면 names에서 false 항목 제거
    if names_cfg:
        names = []
        for n in names_cfg:
            key = f"use_{n}"
            if (key in mcfg) and (not mcfg[key]):
                continue  # use_n: false면 제외
            names.append(n)
    else:
        # names가 없으면 use_*로 자동 구성
        names = [k.replace("use_", "") for k in use_true]

    if not names:
        names = ["mse"]  # 최소 기본값

    # 정규화 파라미터(임계치 변환용)
    norm_tp = config.get("normalization", {}).get("tp", {})
    mean = float(norm_tp.get("mean", 0.0))
    std  = float(norm_tp.get("std", 1.0)) or 1.0

    device = config.get("device", "cuda")

    # ALL_METRICS는 보통 {'use_mse': mse, ...} 형태이므로 심플 맵으로 변환
    NAME_MAP = {k.replace("use_", ""): v for k, v in ALL_METRICS.items()}

    metric_fns = {}

    for name in names:
        n = name.lower()

        if n == "csi":
            raw_th = float(mcfg.get("csi_threshold", 1)) * 1e-3
            window = int(mcfg.get("csi_window", 5))
            norm_th = (raw_th - mean) / std
            metric_fns["csi"] = partial(csi_fn, threshold=norm_th, window_size=window)

        elif n == "fss":
            raw_th = float(mcfg.get("fss_threshold", 1)) * 1e-3
            window = int(mcfg.get("fss_window", 5))
            norm_th = (raw_th - mean) / std
            metric_fns["fss"] = partial(fss_fn, threshold=norm_th, window_size=window)

        elif n == "rhd":
            psz  = int(mcfg.get("rhd_psz", 16))
            bins = int(mcfg.get("rhd_bins", 15))
            metric_fns["rhd"] = partial(rhd_fn, patch_size=psz, num_bins=bins)

        elif n == "lpips":
            lpips_model = get_lpips_model(device=device)
            metric_fns["lpips"] = partial(lpips_fn, model=lpips_model)

        else:
            fn = NAME_MAP.get(n)
            if fn is None:
                raise KeyError(f"Unknown metric name: {n}")
            metric_fns[n] = fn

    return metric_fns
