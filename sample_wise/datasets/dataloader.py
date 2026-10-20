# datasets/dataloader.py
from torch.utils.data import DataLoader
from utils import seed_worker
import torch

from datasets.geodata.geodata_2d import GeoTwo2D
from datasets.geodata.geodata_3d import GeoTwo3D
from datasets.geodata.geodata_3d_uv import GeoTwo3D_UV

def _path_dict(cfg):
    p = cfg["paths"]
    if str(cfg["model"]["version"]).lower() =="3d_uv":
        return {"tp": p["tp_file"], "q850": p["q850_file"], "q1000": p["q1000_file"],
                "u850": p["u850_file"], "u1000": p["u1000_file"], "v850": p["v850_file"], "v1000": p["v1000_file"]}
    else:
        return {"tp": p["tp_file"], "q850": p["q850_file"], "q1000": p["q1000_file"]}

def get_loaders(cfg):
    data_path = _path_dict(cfg) # get data path
    bs = int(cfg["trainer"]["batch_size"]) # get batch size
    nw = int(cfg.get("data", {}).get("num_workers", 4)) # get num workers
    pin = bool(cfg.get("data", {}).get("pin_memory", True)) # get pin memory
    g = torch.Generator().manual_seed(int(cfg.get("seed", 42))) # get seed

    label_offsets = tuple(cfg.get("data", {}).get("label_offsets", [1,2,3])) # get lead time

    # Get Dataset
    if str(cfg["model"]["version"]).lower() == "3d":
        DS = GeoTwo3D
    elif str(cfg["model"]["version"]).lower() == "2d":
        DS = GeoTwo2D
    else:
        DS = GeoTwo3D_UV

    train_set = DS(data_path, mode="train", label_offsets=label_offsets)
    val_set   = DS(data_path, mode="vali",  label_offsets=label_offsets)

    # Get DataLoader
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_set, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              worker_init_fn=seed_worker, generator=g)
    return train_loader, val_loader

def get_test_loader(cfg):
    data_path = _path_dict(cfg) # get data path
    bs = int(cfg["trainer"]["batch_size"]) # get batch size
    nw = int(cfg.get("data", {}).get("num_workers", 4)) # get num workers
    pin = bool(cfg.get("data", {}).get("pin_memory", True)) # get pin memory
    g = torch.Generator().manual_seed(int(cfg.get("seed", 42))) # get seed

    label_offsets = tuple(cfg.get("data", {}).get("label_offsets", [1,2,3])) # get lead time

    # Get Dataset
    if str(cfg["model"]["version"]).lower() == "3d":
        DS = GeoTwo3D
    elif str(cfg["model"]["version"]).lower() == "2d":
        DS = GeoTwo2D
    else:
        DS = GeoTwo3D_UV

    test_set = DS(data_path, mode="test", label_offsets=label_offsets)

    # Get DataLoader
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=pin,
                             worker_init_fn=seed_worker, generator=g)
    return test_loader