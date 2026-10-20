import os
import random
import numpy as np

def set_seed(seed: int = 42, deterministic: bool = False):
    """
    재현성용 시드 고정.
    deterministic=True면 cudnn을 결정적으로 설정(속도 약간↓).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                # PyTorch >=1.8 권장(일부 연산에서 에러 발생 시 except)
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except ImportError:
        pass


def seed_worker(worker_id: int):
    """
    DataLoader용 worker 시드 고정.
    DataLoader(..., worker_init_fn=seed_worker, generator=g) 와 함께 사용.
    """
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
