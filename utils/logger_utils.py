# utils/logger_utils.py
import csv
import os
from typing import Dict, Iterable, Optional
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    - step 로깅: 텐서보드에만 기록 (CSV는 안 씀)
    - epoch 로깅: 텐서보드 + CSV (한 줄)
    - CSV 헤더는 첫 epoch 로깅 시점의 metrics 키로 자동 생성
    """
    def __init__(self, log_dir: str, csv_path: str):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.csv_path = csv_path
        self.csv_file = open(csv_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self._headers: Optional[Iterable[str]] = None  # ['epoch', 'train_loss', 'val_loss', 'val_mse', ...]
        self._metric_order: Optional[Iterable[str]] = None  # ['train_loss', 'val_loss', 'val_mse', ...]

    # ---------- step ----------
    def log_step(self, step: int, scalars: Dict[str, float], prefix: str = "train"):
        """스텝 단위: 텐서보드에만 기록"""
        for k, v in scalars.items():
            self.writer.add_scalar(f"{prefix}/{k}", float(v), step)

    # ---------- epoch ----------
    def _ensure_headers(self, metric_keys: Iterable[str]):
        if self._headers is not None:
            return
        # 고정 순서: ['epoch', <metric_keys...>]
        self._metric_order = list(metric_keys)
        self._headers = ["epoch"] + self._metric_order
        self.csv_writer.writerow(self._headers)
        self.csv_file.flush()

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        에폭 단위: 텐서보드 + CSV
        metrics 예시: {'train_loss':..., 'val_loss':..., 'val_mse':..., ...}
        """
        # TB
        for k, v in metrics.items():
            self.writer.add_scalar(f"epoch/{k}", float(v), epoch)

        # CSV
        self._ensure_headers(metrics.keys())
        row = [epoch] + [metrics.get(k, "") for k in self._metric_order]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        try:
            self.writer.flush()
        except Exception:
            pass
        self.writer.close()
        self.csv_file.close()
