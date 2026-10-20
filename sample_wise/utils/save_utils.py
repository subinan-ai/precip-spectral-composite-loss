# utils/save_utils.py
import os
import torch

def safe_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def save_checkpoint(path, model, optimizer, epoch, metrics, config, scheduler=None):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config
    }
    safe_torch_save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

def load_checkpoint_for_eval(path, model, device="cpu", strict=True):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint
