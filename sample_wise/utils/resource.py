# utils/resource.py
import os

_PIN_STATE = {"done": False, "gpu": None, "threads": None}

def pin_resources(gpu: str = "3", threads: int = 5, quiet: bool = False):
    """
    연구실 리소스 고정:
      - CUDA_VISIBLE_DEVICES 고정
      - torch/BLAS 스레드 수 제한
      - 정보 출력(옵션)
    여러 번 호출돼도 최초 1회만 실제로 적용/출력됩니다.
    """
    global _PIN_STATE
    if _PIN_STATE["done"]:
        # 이미 설정되어 있으면 조용히 리턴
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    try:
        import torch
        torch.set_num_threads(int(threads))
        if not quiet:
            print(f"[INFO] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"[INFO] Num GPUs = {torch.cuda.device_count()}")
    finally:
        _PIN_STATE.update({"done": True, "gpu": str(gpu), "threads": int(threads)})
