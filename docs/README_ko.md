# Loss Experiment Framework (2D/3D)

## 디렉토리 구조
```
loss_experiment/
├─ configs/
│  └─ conv3d_baseline.yaml
├─ datasets/
│  ├─ dataloader.py
│  └─ geodata/
│     ├─ common.py
│     ├─ v2d.py
│     └─ v3d.py
├─ metrics/
│  ├─ __init__.py
│  ├─ get_metrics.py
│  ├─ metric_utils_2d.py
│  └─ metric_utils_3d.py
├─ losses/
│  ├─ loss_2d/ ...
│  ├─ loss_3d/
│  │  └─ loss_utils.py
│  └─ get_loss.py
├─ models/
│  ├─ get_model.py
│  ├─ model_2d/ model.py
│  └─ model_3d/ model.py
├─ utils/
│  ├─ __init__.py
│  ├─ logger_utils.py
│  ├─ save_utils.py
│  ├─ train_utils.py
│  ├─ seed_utils.py
│  └─ resource.py
├─ train_2d.py
├─ train_3d.py
├─ evaluate_2d.py
├─ evaluate_3d.py
└─ main.py
```
## YAML 스키마 예시 (configs/conv3d_baseline.yaml)
```
experiment_name: conv3d_baseline
seed: 42

paths:
  # 데이터 경로 (디렉토리 구조 맞게 수정)
  tp_file: "./data/1979_2023_MaySep_prec.nc"
  q850_file: "./data/1979_2023_MaySep_q850.nc"
  q1000_file: "./data/1979_2023_MaySep_q1000.nc"
  # 저장 경로 (디렉토리 구조 맞게 수정)
  log_dir: "./experiments/<user>/conv3d_baseline/logs"
  ckpt_dir: "./experiments/<user>/conv3d_baseline/checkpoints"

model:
  version: "3d"
  name: "geo_unet"
  x1_channels: 1
  x2_channels: 1
  hidden_channels: 24

data:
  label_offsets: [1, 2, 3]    # ← t 기준 t+1,t+2,t+3 예측
  num_workers: 4
  pin_memory: true
  engine: "netcdf4"           # xarray 엔진 이슈 있으면 "h5netcdf"로 바꿔보기

trainer:
  epochs: 80                 
  amp: true
  grad_clip: 0.0
  accumulate_steps: 1
  log_interval: 50            # 로그하는 mini-batch 개수
  val_interval: 1             # validation 하는 epoch 개수
  save_per_metric: true       # 모든 메트릭별로 최고 성능 epoch 저장

monitor:
  key: "val/mse"              # val_mse 기준 최고 성능 epoch 저장
  mode: "min"                 # min : 작을수록 좋은 메트릭 / max : 클수록 좋은 메트릭  

opt:
  name: Adam
  lr: 1.0e-4
  weight_decay: 0.0
  betas: [0.9, 0.999]
  eps: 1.0e-8
  amsgrad: false

loss:
  name: "facl_dtcwt_lv1"      
  kwargs:
    alpha: 0.1
    beta: 1.0
    gamma: 1.0

metrics:
  # 1) names 기반(권장): 여기에 적은 목록만 계산/로깅
  names: [mse, mae, ssim, csi, fss, rhd, phase, amp, lpips]

  # 2) true인 것만 계산(있으면 names와 함께 필터됨)
  use_mae: true
  use_mse: true
  use_ssim: true
  use_psnr: false
  use_csi: true
  use_fss: true
  use_rhd: true
  use_phase: true
  use_amp: true
  use_lpips: true

  # 추가 파라미터(정규화 사용 시 mean/std로 임계치 변환)
  csi_threshold: 1
  csi_window: 5
  fss_threshold: 1
  fss_window: 5
  rhd_psz: 16
  rhd_bins: 15

normalization: 
  tp:
    mean: 0.0024735306   # 정규화한 값에 맞게 수정
    std: 0.00593467
```
## 실행 방법

# 학습(3D 예시)
```
python main.py --config configs/conv3d_baseline.yaml
```
# 평가(3D 예시)
```
python evaluate_3d.py --ckpt ./experiments/conv3d_baseline/checkpoints/best_mse.pt \
                      --config configs/conv3d_baseline.yaml
```

## 문제 해결
```
- xarray가 파일을 못 염
  - data.engine을 "netcdf4" 또는 "h5netcdf"로 지정
  - 의존성 설치: conda-forge의 netcdf4/h5netcdf/libnetcdf/hdf5 또는 pip의 netCDF4/h5netcdf

- shape mismatch
  - 3D: pred/target ≈ (B,1,T,H,W)
  - 2D: pred/target ≈ (B,K,H,W)
  - 데이터셋 반환 shape와 모델 출력 shape 일치 확인

- 메트릭 미출력
  - metrics.names/use_*가 실제 계산과 일치하는지 확인
  - monitor.key가 계산되는 메트릭인지 확인 (예: "val/mse" → val_mse)
```

## 확장 가이드
```
- 새 모델: models/model_2d|3d/model.py에 클래스 추가 → models/get_model.py 등록
- 새 메트릭: metrics/metric_utils_2d|3d.py에 함수 추가 → ALL_METRICS 매핑
- 새 손실: losses/loss_2d|3d/ 구현 → losses/get_loss.py에 스위치 추가
- 새 데이터: datasets/geodata/common.py 확장(엔진/좌표 정합/보간 등)
```