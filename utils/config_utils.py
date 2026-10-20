# utils/config_utils.py
import os
import yaml
from typing import Any, Dict, Tuple, Union
import torch

def load_config(path: str) -> Dict[str, Any]:
    """
    YAML 파일을 dict로 로드.
    - ~, 환경변수(expandvars) 처리
    """
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # 경로 필드에 대해 ~, $HOME 등 확장 (있을 때만)
    def _expand(v):
        if isinstance(v, str):
            return os.path.expandvars(os.path.expanduser(v))
        return v

    if "paths" in cfg and isinstance(cfg["paths"], dict):
        cfg["paths"] = {k: _expand(v) for k, v in cfg["paths"].items()}

    return cfg

def dump_config(cfg: Dict[str, Any], out_dir: str, filename: str = "config.resolved.yaml") -> str:
    """
    현재 설정을 out_dir에 저장하고 경로를 리턴.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


# ===== ▼ 여기부터 추가: 범용 체크포인트 로더 =====
def _import_obj(module: str, name: str):
    import importlib
    mod = importlib.import_module(module)
    try:
        return getattr(mod, name)
    except AttributeError:
        raise AttributeError(f"[load_model_from_ckpt] '{module}'에 '{name}'(이)가 없습니다.")

def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # DataParallel / DDP로 저장된 키 'module.' 접두사 제거
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):] : v for k, v in state_dict.items()}
    return state_dict

def _pick_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """
    다양한 포맷을 지원:
      - 'state_dict', 'model_state', 'model_state_dict', 'model', 'net', 'state' 등
    """
    for k in ("state_dict", "model_state", "model_state_dict", "model", "net", "state"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # torch.save(model.state_dict())처럼 곧장 저장한 경우
    if all(isinstance(v, (bytes, bytearray)) or hasattr(v, "size") for v in ckpt.values()):
        return ckpt  # heuristic
    raise KeyError("[load_model_from_ckpt] 체크포인트에서 state_dict를 찾지 못했습니다.")

def _map_location_from_device(device: Union[str, Union[str, torch.device]]):
    # torch를 함수 내부에서 임포트 (의존성 최소화)
    import torch
    if device is None:
        return None
    if isinstance(device, str):
        d = device.lower()
        if d == "cpu":
            return torch.device("cpu")
        # 'cuda', 'cuda:0' 등
        if d.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(d)
            else:
                return torch.device("cpu")
    return device

def load_model_from_ckpt(
    cfg_or_path: Union[str, Dict[str, Any]],
    ckpt_path: str = None,
    *,
    device: str = "cuda",
    strict: bool = False,
    model_section: str = "model",
    module_key: str = "module",     # cfg['model']['module']
    class_key: str  = "class",      # cfg['model']['class']
    init_key: str   = "init_args",  # cfg['model']['init_args']
) -> Tuple["torch.nn.Module", Dict[str, Any], str]:
    """
    cfg(경로 또는 dict)와 ckpt 경로로부터 모델을 생성하고 state_dict를 로드해서 반환.
    Returns: (model, meta, used_ckpt_path)
      - model: 로드된 nn.Module (device로 이동)
      - meta : ckpt의 부가 정보 (epoch, step, loss 등 존재하면 포함)
      - used_ckpt_path: 최종 사용된 ckpt 경로

    ckpt_path가 None이면 다음 순서로 추론:
      1) cfg['paths']['ckpt'] 또는 cfg['paths']['ckpt_path']
      2) cfg['model']['ckpt'] 또는 cfg['model']['ckpt_path']
    """
    import torch  # 내부 임포트
    # 1) cfg 로드/정리
    cfg = load_config(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path
    if not isinstance(cfg, dict):
        raise TypeError("[load_model_from_ckpt] cfg_or_path는 str 또는 dict여야 합니다.")

    # 2) ckpt 경로 결정
    if ckpt_path is None:
        paths = cfg.get("paths", {}) or {}
        model_cfg = cfg.get(model_section, {}) or {}
        ckpt_path = paths.get("ckpt") or paths.get("ckpt_path") \
                    or model_cfg.get("ckpt") or model_cfg.get("ckpt_path")
        if ckpt_path is None:
            raise ValueError("[load_model_from_ckpt] ckpt_path 인자를 주거나 cfg에 paths.ckpt / model.ckpt를 지정하세요.")
    ckpt_path = os.path.expanduser(os.path.expandvars(ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[load_model_from_ckpt] 체크포인트가 없습니다: {ckpt_path}")

    # 3) 모델 클래스 동적 import 및 인스턴스 생성
    model_cfg = cfg.get(model_section, {}) or {}
    module = model_cfg.get(module_key, None)
    clsname = model_cfg.get(class_key,  None)
    init_args = model_cfg.get(init_key, {}) or {}
    if module is None or clsname is None:
        raise KeyError(f"[load_model_from_ckpt] cfg['{model_section}']에 '{module_key}'와 '{class_key}'가 필요합니다.")

    ModelCls = _import_obj(module, clsname)
    model = ModelCls(**init_args)

    # 4) 체크포인트 로드
    map_loc = _map_location_from_device(device)
    ckpt = torch.load(ckpt_path, map_location=map_loc)
    state = _pick_state_dict(ckpt)
    state = _strip_module_prefix(state)

    # 5) state_dict 주입
    missing, unexpected = model.load_state_dict(state, strict=strict)
    # strict=False면 (missing, unexpected)는 리스트 반환; strict=True면 여기 오기전 예외 발생
    if strict is False and (missing or unexpected):
        print(f"[load_model_from_ckpt] missing_keys={missing}, unexpected_keys={unexpected}")

    # 6) 디바이스 이동
    if isinstance(map_loc, torch.device):
        model.to(map_loc)

    # 7) 메타 정보 정리
    meta = {}
    for k in ("epoch", "step", "global_step", "best_metric", "loss", "loss_name", "trainer_state"):
        if k in ckpt:
            meta[k] = ckpt[k]

    model.eval()  # 평가용 기본값
    return model, meta, ckpt_path
# ===== ▲ 추가 끝 =====
