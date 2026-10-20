import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset

class PrecipitationDataset(Dataset):
    """
    단일 강수 NetCDF 파일에서 (입력 t) -> (라벨 t+offset들) 샘플을 생성하는 간단 예시.
    - tp_file: 강수 NetCDF 경로
    - label_offsets: [1,2,3] 이면 t+1, t+2, t+3 시점 예측
    - engine: "netcdf4" (xarray open_dataset 엔진)
    - split: "val" (호환용; 실제 분할은 여기선 단순)
    """
    def __init__(self,
                 tp_file: str,
                 label_offsets=(1,2,3),
                 engine="netcdf4",
                 split="val",
                 # 필요시 추가 인자들...
                 ):
        super().__init__()
        if not os.path.exists(tp_file):
            raise FileNotFoundError(f"tp_file not found: {tp_file}")

        self.ds = xr.open_dataset(tp_file, engine=engine)
        # 변수명은 파일에 맞게 조정하세요 (예: 'tp', 'precip', 'prec' 등)
        # 아래는 가장 흔한 ERA5 강수 누적 변수명 'tp' 가정
        var_name = None
        for cand in ("tp", "prec", "precip", "pr"):
            if cand in self.ds.data_vars:
                var_name = cand
                break
        if var_name is None:
            raise KeyError(f"강수 변수(tp/prec/precip/pr)를 파일에서 못 찾았습니다. vars={list(self.ds.data_vars)}")

        # (T, H, W) 형태로 가져오기 (필요시 squeeze/transpose)
        arr = self.ds[var_name].load().values
        arr = np.asarray(arr)
        # 차원 순서 보정: (time, lat, lon) 가정 → (T,H,W)
        # ERA5 tp는 종종 (time, lat, lon)
        if arr.ndim == 3:
            T, H, W = arr.shape
        elif arr.ndim == 4:
            # (time, level, lat, lon) 같은 경우 첫 레벨만 사용
            arr = arr[:, 0]
            T, H, W = arr.shape
        else:
            raise ValueError(f"예상 못한 배열 차원: {arr.shape}")

        self.x = arr.astype(np.float32)
        self.label_offsets = list(label_offsets)
        self.max_offset = max(self.label_offsets) if self.label_offsets else 0

        # 사용할 유효한 t 범위: 0 .. T-1-max_offset (입력 t에서 모든 라벨이 존재하도록)
        self.valid_T = T - self.max_offset
        if self.valid_T <= 0:
            raise ValueError(f"시간 축이 너무 짧습니다: T={T}, max_offset={self.max_offset}")

    def __len__(self):
        return self.valid_T

    def __getitem__(self, idx):
        # 입력: t 시점의 2D 강수장 (H,W)
        x_t = self.x[idx]  # (H,W)
        # 라벨: t+offset들의 2D 강수장들을 쌓아서 (C=오프셋수, H, W)
        ys = [self.x[idx + off] for off in self.label_offsets]
        y = np.stack(ys, axis=0)  # (C,H,W)

        # 채널 차원 추가해서 (1,H,W)로 입력 만들기
        x_t = np.expand_dims(x_t, axis=0)  # (1,H,W)

        # torch tensor 변환
        x_t = torch.from_numpy(x_t)       # (1,H,W)
        y   = torch.from_numpy(y)         # (C,H,W)

        return x_t, y
