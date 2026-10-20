# utils/config_utils.py
import os
import yaml
from typing import Any, Dict

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
