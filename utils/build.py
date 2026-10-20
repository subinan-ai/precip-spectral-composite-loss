# utils/build.py
import importlib
from typing import Any, Dict, Union

def _import_obj(module: str, name: str):
    mod = importlib.import_module(module)
    return getattr(mod, name)

def _ensure_cfg(cfg_or_path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    # utils/__init__.py에서 load_config를 export했다고 가정
    from utils import load_config
    if isinstance(cfg_or_path, str):
        return load_config(cfg_or_path)
    if isinstance(cfg_or_path, dict):
        return cfg_or_path
    raise TypeError(f"[build] cfg_or_path must be str or dict, got {type(cfg_or_path)}")

def _resolve_dataset_spec(dcfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    다양한 스키마를 허용:
      1) 평평한 스키마:
         data.module, data.class, data.val.args
      2) 중첩 스키마:
         data.dataset.module, data.dataset.class, data.dataset.args
      3) 기타 별칭:
         data.target = "module:Class"  (예: 'datasets.precip:PrecipitationDataset')
    """
    mod = dcfg.get("module")
    cls = dcfg.get("class")
    args = ((dcfg.get("val") or {}).get("args") or {}).copy()

    # (2) nested dataset.*
    if not (mod and cls):
        ds = dcfg.get("dataset") or {}
        mod = mod or ds.get("module")
        cls = cls or ds.get("class")
        if not args:
            args = (ds.get("args") or {}).copy()

    # (3) target = "pkg.mod:Class"
    if not (mod and cls) and "target" in dcfg and isinstance(dcfg["target"], str):
        target = dcfg["target"]
        if ":" in target:
            mod, cls = target.split(":", 1)

    return {"module": mod, "class": cls, "args": args}

def build_val_dataset_from_cfg(cfg_or_path: Union[str, Dict[str, Any]]):
    cfg = _ensure_cfg(cfg_or_path)
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise KeyError("[build] cfg['data'] 섹션이 없습니다.")

    dcfg = cfg["data"]
    spec = _resolve_dataset_spec(dcfg)
    module, clsname, args = spec["module"], spec["class"], spec["args"]

    if not module or not clsname:
        # 디버깅 도움: 실제 들어온 키들을 보여주기
        raise KeyError(
            "[build] cfg['data']에는 'module'과 'class'가 필요합니다. "
            f"받은 keys={list(dcfg.keys())}, data={dcfg}"
        )

    args.setdefault("split", "val")  # 기본 split

    DatasetCls = _import_obj(module, clsname)
    try:
        return DatasetCls(**args)
    except TypeError as e:
        raise TypeError(f"[build] {module}.{clsname}(**args) 생성자 인자 오류: {e}\nargs={args}")
