# utils/train_utils.py
import torch

def build_optimizer(cfg, model):
    opt = cfg["opt"]
    lr = float(opt.get("lr", 1e-3))
    wd = float(opt.get("weight_decay", 0.0))
    betas = tuple(opt.get("betas", (0.9, 0.999)))
    eps = float(opt.get("eps", 1e-8))
    amsgrad = bool(opt.get("amsgrad", False))
    try:
        return torch.optim.Adam(
            model.parameters(),
            lr=lr, weight_decay=wd, betas=betas, eps=eps, amsgrad=amsgrad, fused=True
        )
    except TypeError:
        # PyTorch 버전에 따라 fused 인자 없을 수 있음
        return torch.optim.Adam(
            model.parameters(),
            lr=lr, weight_decay=wd, betas=betas, eps=eps, amsgrad=amsgrad
        )

def build_scheduler(*args, **kwargs):
    return None  # 스케줄러 미사용
