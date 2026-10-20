# datasets/geodata/common_uv.py
import numpy as np
import xarray as xr

# 시간 추출 helper 함수
def _pick_time_coord(ds):
    for cand in ("time", "valid_time", "datetime", "date"):
        if cand in ds.coords:
            return cand
    for k, v in ds.coords.items():
        if np.issubdtype(np.array(v.values).dtype, np.datetime64):
            return k
    raise KeyError("시간 좌표(time/valid_time 등)를 찾지 못했습니다.")

# nc 파일 열기 helper 함수
def _open_ds(path, engine=None):
    if engine is not None:
        return xr.open_dataset(path, engine=engine)
    for eng in (None, "netcdf4", "h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=eng) if eng else xr.open_dataset(path)
        except Exception:
            continue
    raise RuntimeError(f"xarray로 파일을 열 수 없습니다: {path}")

# 학습 데이터 형식으로 변환 helper 함수
def load_raw_arrays(data_path: dict,
                    label_offsets=(1, 2, 3),
                    engine=None):
    """
    data_path: {"tp": tp.nc, "q850": q850.nc, "q1000": q1000.nc, "u850": u850.nc, "u1000": u1000.nc, "v850": v850.nc, "v1000": v1000.nc}
    label_offsets: 라벨로 사용할 미래 시점의 오프셋 리스트/튜플 (예: (1,3,5))
                   ※ 오프셋은 1 이상의 정수여야 하며, 순서대로 라벨 채널 축에 반영됨.
    반환:
      X1_raw: (N,H,W)         # t시점 절대습도장 
      X2_raw: (N,H,W)         # t시점 강수장
      X3_raw: (N,H,W)         # t시점 u 바람장
      X4_raw: (N,H,W)         # t시점 v 바람장
      Y_raw : (N,K,H,W)       # t+1/2/3 시점 강수장
      times : (N,) np.datetime64  # 입력 기준 시각 t
      lat, lon
    """
    # nc 파일 불러오기
    ds_tp    = _open_ds(data_path["tp"], engine=engine)
    ds_q850  = _open_ds(data_path["q850"], engine=engine)
    ds_q1000 = _open_ds(data_path["q1000"], engine=engine)
    ds_u850 = _open_ds(data_path["u850"], engine=engine)
    ds_u1000 = _open_ds(data_path["u1000"], engine=engine)
    ds_v850 = _open_ds(data_path["v850"], engine=engine)
    ds_v1000 = _open_ds(data_path["v1000"], engine=engine)

    t_tp = _pick_time_coord(ds_tp)

    tp    = ds_tp["tp"].values
    q850  = ds_q850["q"].values
    q1000 = ds_q1000["q"].values
    u850 = ds_u850["u"].values
    u1000 = ds_u1000["u"].values
    v850 = ds_v850["v"].values
    v1000 = ds_v1000["v"].values

    if q850.ndim == 4 and q850.shape[1] == 1:   q850  = q850.squeeze(1)
    if q1000.ndim == 4 and q1000.shape[1] == 1: q1000 = q1000.squeeze(1)
    if u850.ndim == 4 and u850.shape[1] == 1:   u850  = u850.squeeze(1)
    if u1000.ndim == 4 and u1000.shape[1] == 1: u1000 = u1000.squeeze(1)
    if v850.ndim == 4 and v850.shape[1] == 1:   v850  = v850.squeeze(1)
    if v1000.ndim == 4 and v1000.shape[1] == 1: v1000 = v1000.squeeze(1)

    # lead time 설정
    if isinstance(label_offsets, int):
        label_offsets = tuple(range(1, label_offsets + 1))
    offsets = tuple(label_offsets)
    if len(offsets) == 0:
        raise ValueError("label_offsets가 비어 있습니다.")
    if any((not isinstance(o, (int, np.integer))) or o <= 0 for o in offsets):
        raise ValueError(f"label_offsets는 1 이상의 정수만 가능합니다: {label_offsets}")

    max_off = int(max(offsets))
    T = tp.shape[0]
    N = T - max_off
    if N <= 0:
        raise ValueError(f"시간축 길이 T={T}가 최대 오프셋 {max_off}보다 커야 합니다.")

    # Build input, label
    qavg = (q850 + q1000) / 2.0
    uavg = (u850 + u1000) / 2.0
    vavg = (v850 + v1000) / 2.0
    X1_raw = qavg[:N]       # (N,H,W)
    X2_raw = tp[:N]         # (N,H,W)
    X3_raw = uavg[:N]       # (N,H,W)
    X4_raw = vavg[:N]       # (N,H,W)

    # Y_raw: (N,T,H,W), T=len(offsets), 각 채널이 t+offset
    Ys = [tp[o:o+N] for o in offsets]   # 각 (N,H,W)
    Y_raw = np.stack(Ys, axis=1)

    times = ds_tp[t_tp].values[:N]
    lat   = ds_tp["lat"].values
    lon   = ds_tp["lon"].values

    ds_tp.close(); ds_q850.close(); ds_q1000.close(); ds_u850.close(); ds_u1000.close(); ds_v850.close(); ds_v1000.close()
    return X1_raw, X2_raw, X3_raw, X4_raw, Y_raw, times, lat, lon
