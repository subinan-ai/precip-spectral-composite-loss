import xarray as xr
import numpy as np
import os

def compute_percentile_map_from_nc(nc_path, varname="tp", 
                                   percentile=90, wet_day_min=1.0, out_path=None):
    """
    NetCDF 파일에서 R90p/R95p 격자 맵 계산 후 저장.

    Parameters
    ----------
    nc_path : str
        NetCDF 파일 경로
    varname : str
        강수 변수 이름 (ex: 'prec')
    percentile : int
        90 → R90p, 95 → R95p
    wet_day_min : float
        wet-day 기준 (mm)
    out_path : str
        저장할 .npy 파일 경로 (없으면 저장 안 함)

    Returns
    -------
    th_map : np.ndarray (H,W)
    """
    ds = xr.open_dataset(nc_path)
    da = ds[varname]  # (time, lat, lon) 가정

    # wet-day 마스크
    da = da.where(da >= wet_day_min)

    # 격자별 퍼센타일 계산
    th_map = np.nanpercentile(da.values, percentile, axis=0)  # (H,W)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, th_map)
        print(f"✅ Saved {percentile}p map to {out_path}")

    return th_map


if __name__ == "__main__":
    nc_file = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/data/1979_2023_MaySep_prec.nc"

    # R90p
    R90p_map = compute_percentile_map_from_nc(
        nc_file, varname="tp", percentile=90, wet_day_min=1.0,
        out_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/threshold_map/R90p_map.npy"
    )

    # R95p
    R95p_map = compute_percentile_map_from_nc(
        nc_file, varname="tp", percentile=95, wet_day_min=1.0,
        out_path="/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/threshold_map/R95p_map.npy"
    )
