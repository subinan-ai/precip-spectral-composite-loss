# losses/__init__.py

def get_loss(version: str, name: str):
    """
    train_2d.py / train_3d.py에서 공통 사용:
      loss_fn = get_loss(cfg["model"]["version"], cfg["loss"]["name"])
    loss 함수는 (pred, y, step=None, total_steps=None, **kwargs) 시그니처를 권장.
    """
    v = str(version).lower()
    n = str(name).lower()
    if v == "3d" or v == "3d_uv":
        from .loss_3d.registry import get_loss_3d
        return get_loss_3d(n)
    elif v == "2d":
        from .loss_2d.registry import get_loss_2d
        return get_loss_2d(n)
    else:
        raise ValueError(f"Unsupported model.version: {version}")
