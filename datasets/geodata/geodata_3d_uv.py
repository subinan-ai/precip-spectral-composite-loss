# datasets/geodata/v3d.py
import numpy as np
import torch
from torch.utils.data import Dataset
from .common_uv import load_raw_arrays

class GeoTwo3D_UV(Dataset):
    """
    Return:
      X1: (1,H,W)       # t시점 절대습도장
      X2: (1,H,W)       # t시점 강수장
      X3: (1,H,W)       # t시점 u 바람장
      X4: (1,H,W)       # t시점 v 바람장
      Y : (3,H,W)       # t+1/2/3 시점 강수장
      time: str
    """
    def __init__(self, data_path: dict, mode="train",
                 label_offsets=(1,2,3),
                 train_end="2016-01-01", val_end="2018-01-01",
                 engine=None):
        assert mode in ("train", "vali", "test")
        self.mode = mode

        # Load data from nc files
        self.X1, self.X2, self.X3, self.X4, self.Y, self.times, self.lat, self.lon = load_raw_arrays(
            data_path, label_offsets=label_offsets, engine=engine
        )

        # split train/val/test dataset
        train_end = np.datetime64(train_end)
        val_end   = np.datetime64(val_end)
        if mode == "train":
            mask = self.times < train_end
        elif mode == "vali":
            mask = (self.times >= train_end) & (self.times < val_end)
        else:
            mask = self.times >= val_end

        self.idx = np.where(mask)[0]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        X1 = torch.tensor(self.X1[j], dtype=torch.float32).unsqueeze(0)    # (1,H,W)
        X2 = torch.tensor(self.X2[j], dtype=torch.float32).unsqueeze(0)    # (1,H,W)
        X3 = torch.tensor(self.X3[j], dtype=torch.float32).unsqueeze(0)    # (1,H,W)
        X4 = torch.tensor(self.X4[j], dtype=torch.float32).unsqueeze(0)    # (1,H,W)
        Y  = torch.tensor(self.Y[j],  dtype=torch.float32).unsqueeze(0)    # (1,3,H,W)
        t  = str(self.times[j])
        return {"X1": X1, "X2": X2, "X3": X3, "X4": X4, "Y": Y, "time": t}
