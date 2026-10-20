# losses/loss_3d/losses.py
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DTCWTForward

# ---------- 기본 3D: MSE/MAE ----------

def loss_mse(outputs, labels):

    loss_per_batch = F.mse_loss(outputs, labels, reduction="none") # (B, C, T, H, W)
    loss_per_t = loss_per_batch.mean(dim=[0,1,3,4]) # (T,)

    return loss_per_t, loss_per_t.mean()

def loss_mae(outputs, labels):

    loss_per_batch = F.l1_loss(outputs, labels, reduction="none") # (B, C, T, H, W)
    loss_per_t = loss_per_batch.mean(dim=[0,1,3,4]) # (T,)

    return loss_per_t, loss_per_t.mean()

# ---------- FFT 기반: FAL/FCL/FACL ----------

def loss_fal(outputs, labels):
    """
    Fourier Amplitude Loss (FAL)
    - 출력 및 정답을 푸리에 변환 후, amplitude 차이의 MSE를 계산
    - 반환: (T별 loss, 전체 평균)
    """
    pred_fft = torch.fft.fft2(outputs, norm="ortho")  # (B, C, T, H, W)
    target_fft = torch.fft.fft2(labels, norm="ortho")

    pred_amp = torch.abs(pred_fft) # (B, C, T, H, W)
    target_amp = torch.abs(target_fft)

    fal_per_t = ((pred_amp - target_amp) ** 2).mean(dim=(0, 1, 3, 4))  # (T,)
    return fal_per_t, fal_per_t.mean()


def loss_fcl(outputs, labels, eps=1e-8):
    """
    Fourier Coherence Loss (FCL)
    - 푸리에 도메인에서 정규화된 내적을 기반으로 코히어런스 손실 계산
    - 반환: (T별 loss, 전체 평균)
    """
    pred_fft = torch.fft.fft2(outputs, norm="ortho")  # (B, C, T, H, W)
    target_fft = torch.fft.fft2(labels, norm="ortho")

    num = torch.real(pred_fft * target_fft.conj() + target_fft * pred_fft.conj()).sum(dim=(-2, -1))  # (B, C, T)
    denom = torch.sqrt(
        (torch.abs(pred_fft) ** 2).sum(dim=(-2, -1)) *
        (torch.abs(target_fft) ** 2).sum(dim=(-2, -1)) + eps
    )  # (B, C, T)

    corr = num / denom
    fcl_per_t = 1.0 - 0.5 * corr.mean(dim=(0, 1))  # (T,)
    return fcl_per_t, fcl_per_t.mean()


def loss_facl(outputs, labels, step, total_steps, alpha=0.1):
    """
    Fourier Amplitude & Coherence Loss (FACL)
    - 학습 단계에 따라 FAL 또는 FCL을 확률적으로 선택
    - P_t = max(0, 1 - step / ((1 - alpha) * total_steps))
    - 초기에는 FCL 중심, 후기에는 FAL 중심으로 전환
    - 반환: (T별 loss, 전체 평균, 선택된 loss 이름)
    """
    P_t = max(0.0, 1.0 - float(step) / ((1 - alpha) * float(total_steps)))  # 논문 기반 스케줄

    if torch.rand(1).item() > P_t:
        loss_per_t, loss = loss_fal(outputs, labels)
        return loss_per_t, loss, "FAL"
    else:
        loss_per_t, loss = loss_fcl(outputs, labels)
        return loss_per_t, loss, "FCL"
    
def loss_facl_phase(outputs, labels, step, total_steps, alpha=0.1):
    """
    Fourier Amplitude & Coherence Loss (FACL)
    - 학습 단계에 따라 FAL 또는 FCL을 확률적으로 선택
    - P_t = max(0, 1 - step / ((1 - alpha) * total_steps))
    - 초기에는 FCL 중심, 후기에는 FAL 중심으로 전환 [Phase signal은 계속 주입]
    - 반환: (T별 loss, 전체 평균, 선택된 loss 이름)
    """
    P_t = max(0.0, 1.0 - float(step) / ((1 - alpha) * float(total_steps)))  # 논문 기반 스케줄

    if torch.rand(1).item() > P_t:
        loss_per_t_amp, loss_amp = loss_fal(outputs, labels)
        loss_per_t_phase, loss_phase = loss_fcl(outputs, labels)

        loss_per_t = loss_per_t_amp + 0.2 * loss_per_t_phase
        loss = loss_amp + 0.2 * loss_phase
        return loss_per_t, loss, "FAL"
    else:
        loss_per_t, loss = loss_fcl(outputs, labels)
        return loss_per_t, loss, "FCL"

# ---------- DWT (Haar, J=1) ----------

def extract_wavelet_subbands(tensor):
    """
    (B, C, T, H, W) → 서브밴드 추출
    Returns:
        LL: (B, C, T, H//2, W//2)
        LH, HL, HH: same shape
    """
    device = tensor.device
    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)
    B, C, T, H, W = tensor.shape
    LL_list, LH_list, HL_list, HH_list = [], [], [], []

    for t in range(T):
        x_t = tensor[:, :, t]  # (B, C, H, W)
        Yl, Yh = dwt(x_t)      # Yh[0] = (B, C, 3, H//2, W//2)
        LL_list.append(Yl)
        LH_list.append(Yh[0][:, :, 0])
        HL_list.append(Yh[0][:, :, 1])
        HH_list.append(Yh[0][:, :, 2])

    # stack over time → (B, C, T, H//2, W//2)
    def stack_list(lst): return torch.stack(lst, dim=2)
    return stack_list(LL_list), stack_list(LH_list), stack_list(HL_list), stack_list(HH_list)

def subband_mse(pred_band, target_band):
    loss_per_batch = F.mse_loss(pred_band, target_band, reduction='none')
    loss_per_t = loss_per_batch.mean(dim=[0, 1, 3, 4])  # (T,)
    return loss_per_t, loss_per_t.mean()

def loss_fcl_ll(outputs, labels, alpha=1.0, beta=1.0):
    fcl_t, fcl = loss_fcl(outputs, labels)
    ll_pred, _, _, _ = extract_wavelet_subbands(outputs)
    ll_target, _, _, _ = extract_wavelet_subbands(labels)
    ll_t, ll = subband_mse(ll_pred, ll_target)
    total = alpha * fcl + beta * ll
    return total, {
        "loss_fcl": fcl.item(), "loss_ll": ll.item(), "loss_total": total.item()
    }

def loss_fcl_high(outputs, labels, alpha=1.0, beta=1.0):
    fcl_t, fcl = loss_fcl(outputs, labels)
    _, lh_pred, hl_pred, hh_pred = extract_wavelet_subbands(outputs)
    _, lh_target, hl_target, hh_target = extract_wavelet_subbands(labels)

    lh_t, lh = subband_mse(lh_pred, lh_target)
    hl_t, hl = subband_mse(hl_pred, hl_target)
    hh_t, hh = subband_mse(hh_pred, hh_target)

    high_loss = (lh + hl + hh) / 3.0
    total = alpha * fcl + beta * high_loss

    return total, {
        "loss_fcl": fcl.item(),
        "loss_lh": lh.item(), "loss_hl": hl.item(), "loss_hh": hh.item(),
        "loss_total": total.item()
    }

def loss_fal_ll(outputs, labels, alpha=1.0, beta=1.0):
    fal_t, fal = loss_fal(outputs, labels)
    ll_pred, _, _, _ = extract_wavelet_subbands(outputs)
    ll_target, _, _, _ = extract_wavelet_subbands(labels)

    ll_t, ll = subband_mse(ll_pred, ll_target)
    total = alpha * fal + beta * ll

    return total, {
        "loss_fal": fal.item(), "loss_ll": ll.item(), "loss_total": total.item()
    }

def loss_fal_high(outputs, labels, alpha=1.0, beta=1.0):
    fal_t, fal = loss_fal(outputs, labels)
    _, lh_pred, hl_pred, hh_pred = extract_wavelet_subbands(outputs)
    _, lh_target, hl_target, hh_target = extract_wavelet_subbands(labels)
    
    lh_t, lh = subband_mse(lh_pred, lh_target)
    hl_t, hl = subband_mse(hl_pred, hl_target)
    hh_t, hh = subband_mse(hh_pred, hh_target)

    high_loss = (lh + hl + hh) / 3.0
    total = alpha * fal + beta * high_loss

    return total, {
        "loss_fal": fal.item(),
        "loss_lh": lh.item(), "loss_hl": hl.item(), "loss_hh": hh.item(),
        "loss_total": total.item()
    }


def loss_fal_high(outputs, labels, alpha=1.0, beta=1.0):
    fal_t, fal = loss_fal(outputs, labels)
    _, lh_pred, hl_pred, hh_pred = extract_wavelet_subbands(outputs)
    _, lh_target, hl_target, hh_target = extract_wavelet_subbands(labels)

    lh_t, lh = subband_mse(lh_pred, lh_target)
    hl_t, hl = subband_mse(hl_pred, hl_target)
    hh_t, hh = subband_mse(hh_pred, hh_target)

    high_loss = (lh + hl + hh) / 3.0
    total = alpha * fal + beta * high_loss

    return total, {
        "loss_fal": fal.item(),
        "loss_lh": lh.item(), "loss_hl": hl.item(), "loss_hh": hh.item(),
        "loss_total": total.item()
    }

def loss_fcl_all(outputs, labels, alpha=1.0, beta=1.0):
    fcl_t, fcl = loss_fcl(outputs, labels)
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    ll_t, lh_tgt, hl_tgt, hh_tgt = extract_wavelet_subbands(labels)

    ll_t, ll = subband_mse(ll_p, ll_t)
    lh_t, lh = subband_mse(lh_p, lh_tgt)
    hl_t, hl = subband_mse(hl_p, hl_tgt)
    hh_t, hh = subband_mse(hh_p, hh_tgt)

    wavelet_loss = (ll + lh + hl + hh) / 4.0
    total = alpha * fcl + beta * wavelet_loss

    return total, {
        "loss_fcl": fcl.item(),
        "loss_ll": ll.item(), "loss_lh": lh.item(),
        "loss_hl": hl.item(), "loss_hh": hh.item(),
        "loss_total": total.item()
    }

def loss_fal_all(outputs, labels, alpha=1.0, beta=1.0):
    fal_t, fal = loss_fal(outputs, labels)
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    ll_t, lh_tgt, hl_tgt, hh_tgt = extract_wavelet_subbands(labels)

    ll_t, ll = subband_mse(ll_p, ll_t)
    lh_t, lh = subband_mse(lh_p, lh_tgt)
    hl_t, hl = subband_mse(hl_p, hl_tgt)
    hh_t, hh = subband_mse(hh_p, hh_tgt)

    wavelet_loss = (ll + lh + hl + hh) / 4.0
    total = alpha * fal + beta * wavelet_loss

    return total, {
        "loss_fcl": fal.item(),
        "loss_ll": ll.item(), "loss_lh": lh.item(),
        "loss_hl": hl.item(), "loss_hh": hh.item(),
        "loss_total": total.item()
    }

def loss_ll_only(outputs, labels):
    ll_p, _, _, _ = extract_wavelet_subbands(outputs)
    ll_t, _, _, _ = extract_wavelet_subbands(labels)
    return subband_mse(ll_p, ll_t)


def loss_high_only(outputs, labels):
    _, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    _, lh_tgt, hl_tgt, hh_tgt = extract_wavelet_subbands(labels)

    lh_t, lh = subband_mse(lh_p, lh_tgt)
    hl_t, hl = subband_mse(hl_p, hl_tgt)
    hh_t, hh = subband_mse(hh_p, hh_tgt)

    high_loss = (lh + hl + hh) / 3.0
    return None, high_loss

def loss_wavelet_all(outputs, labels):
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    ll_tgt, lh_tgt, hl_tgt, hh_tgt = extract_wavelet_subbands(labels)

    ll_t, ll = subband_mse(ll_p, ll_tgt)
    lh_t, lh = subband_mse(lh_p, lh_tgt)
    hl_t, hl = subband_mse(hl_p, hl_tgt)
    hh_t, hh = subband_mse(hh_p, hh_tgt)

    loss = (ll + lh + hl + hh) / 4.0
    return None, loss

def loss_wavelet_mixed(outputs, labels, alpha=1.0, beta=1.0):
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    ll_tgt, lh_tgt, hl_tgt, hh_tgt = extract_wavelet_subbands(labels)

    ll_t, ll = subband_mse(ll_p, ll_tgt)
    lh_t, lh = subband_mse(lh_p, lh_tgt)
    hl_t, hl = subband_mse(hl_p, hl_tgt)
    hh_t, hh = subband_mse(hh_p, hh_tgt)

    low_loss = ll
    high_loss = (lh + hl + hh) / 3.0

    total = alpha * low_loss + beta * high_loss
    return total, {
        "loss_ll": ll.item(), "loss_lh": lh.item(),
        "loss_hl": hl.item(), "loss_hh": hh.item(),
        "loss_total": total.item()
    }


def loss_facl_wavelet(
    outputs, labels, step, total_steps, alpha=0.1,
    w_facl=1.0, w_wave=1.0,    # 주손실(FAL/FCL)과 wavelet_all 가중치
):
    """
    FACL + Wavelet_all 결합 손실
    - P_t = max(0, 1 - step / ((1 - alpha) * total_steps)) (초기 FCL 위주 → 후기 FAL 위주)
    - 매 스텝에서 FAL/FCL 중 하나를 선택하고, wavelet_all(LL,LH,HL,HH 평균)을 가산
    - 반환: (T별 결합 손실, 결합 스칼라, 선택된 주손실 이름 "FAL+WAV"/"FCL+WAV")
    """
    # ----- FACL 스케줄: 주손실 선택 -----
    P_t = max(0.0, 1.0 - float(step) / ((1 - alpha) * float(total_steps)))
    if torch.rand(1).item() > P_t:
        base_per_t, base_loss = loss_fal(outputs, labels)  # (T,), scalar
        tag = "FAL+WAV"
    else:
        base_per_t, base_loss = loss_fcl(outputs, labels)  # (T,), scalar
        tag = "FCL+WAV"

    # ----- wavelet_all: 4개 서브밴드 평균(MSE) -----
    # (B,C,T,H//2,W//2) 형태로 서브밴드 추출
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    ll_t, lh_t, hl_t, hh_t = extract_wavelet_subbands(labels)

    # 각 밴드별 MSE (T별, 스칼라)
    ll_t_per, ll_loss = subband_mse(ll_p, ll_t)
    lh_t_per, lh_loss = subband_mse(lh_p, lh_t)
    hl_t_per, hl_loss = subband_mse(hl_p, hl_t)
    hh_t_per, hh_loss = subband_mse(hh_p, hh_t)

    wavelet_per_t = (ll_t_per + lh_t_per + hl_t_per + hh_t_per) / 4.0  # (T,)
    wavelet_loss  = (ll_loss  + lh_loss  + hl_loss  + hh_loss ) / 4.0  # scalar

    # ----- 결합 -----
    total_per_t = w_facl * base_per_t + w_wave * wavelet_per_t
    total_loss  = w_facl * base_loss  + w_wave * wavelet_loss

    return total_per_t, total_loss, tag
import torch
import torch.nn.functional as F

def loss_facl_wavelet_ll(
    outputs, labels,
    step: int, total_steps: int, alpha: float = 0.1,
    w_facl: float = 1.0, w_ll: float = 1.0,
):
    """
    FACL(= FAL/FCL 확률적 선택) + Wavelet LL(저주파) 결합
    반환: (T별 결합 손실, 스칼라, 태그 "FAL+LL"/"FCL+LL")
    """
    P_t = max(0.0, 1.0 - float(step) / ((1 - alpha) * float(total_steps)))
    if torch.rand((), device=outputs.device).item() > P_t:
        base_per_t, base_loss = loss_fal(outputs, labels); tag = "FAL+LL"
    else:
        base_per_t, base_loss = loss_fcl(outputs, labels); tag = "FCL+LL"

    ll_p, _, _, _ = extract_wavelet_subbands(outputs)
    ll_t, _, _, _ = extract_wavelet_subbands(labels)
    ll_per_t, ll_loss = subband_mse(ll_p, ll_t)

    total_per_t = w_facl * base_per_t + w_ll * ll_per_t
    total_loss  = w_facl * base_loss  + w_ll * ll_loss
    return total_per_t, total_loss, tag


def loss_facl_wavelet_high(
    outputs, labels,
    step: int, total_steps: int, alpha: float = 0.1,
    w_facl: float = 1.0, w_high: float = 1.0,
):
    """
    FACL(= FAL/FCL 확률적 선택) + Wavelet High(LH/HL/HH) 결합
    - 고주파는 (LH, HL, HH) **균등 평균**으로 고정
    반환: (T별 결합 손실, 스칼라, 태그 "FAL+HIGH"/"FCL+HIGH")
    """
    P_t = max(0.0, 1.0 - float(step) / ((1 - alpha) * float(total_steps)))
    if torch.rand((), device=outputs.device).item() > P_t:
        base_per_t, base_loss = loss_fal(outputs, labels); tag = "FAL+HIGH"
    else:
        base_per_t, base_loss = loss_fcl(outputs, labels); tag = "FCL+HIGH"

    # 고주파 서브밴드
    _, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    _, lh_t, hl_t, hh_t = extract_wavelet_subbands(labels)

    lh_per_t, lh_loss = subband_mse(lh_p, lh_t)   # (T,), scalar
    hl_per_t, hl_loss = subband_mse(hl_p, hl_t)
    hh_per_t, hh_loss = subband_mse(hh_p, hh_t)

    # **균등 평균(가중치 없음)**
    high_per_t = (lh_per_t + hl_per_t + hh_per_t) / 3.0
    high_loss  = (lh_loss  + hl_loss  + hh_loss ) / 3.0

    total_per_t = w_facl * base_per_t + w_high * high_per_t
    total_loss  = w_facl * base_loss  + w_high * high_loss
    return total_per_t, total_loss, tag

def loss_facl_dwt(outputs, labels, step, total_steps, alpha=0.1):

    # Wavelet 밴드 추출
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands(outputs)
    ll_t, lh_tgt, hl_tgt, hh_tgt = extract_wavelet_subbands(labels)

    # 각 band에 대하여 FACL 적용
    _, loss_ll, _ = loss_facl(ll_p, ll_t, step, total_steps, alpha)
    _, loss_lh, _ = loss_facl(lh_p, lh_tgt, step, total_steps, alpha)
    _, loss_hl, _ = loss_facl(hl_p, hl_tgt, step, total_steps, alpha)
    _, loss_hh, _ = loss_facl(hh_p, hh_tgt, step, total_steps, alpha)

    total = (loss_ll + loss_lh + loss_hl + loss_hh) / 4.0

    return None, total, None

# ---------- DTCWT (3D) ----------
    
def loss_wal_complex(outputs, labels):
    """
    Wavelet Amplitude Loss (WAL)
    - 출력 및 정답의 wavelet 도메인에서 amplitude 차이의 MSE를 계산
    - 반환: (T별 loss, 전체 평균)
    """
    pred_amp = torch.abs(outputs)
    target_amp = torch.abs(labels)

    wal_per_t = ((pred_amp - target_amp) ** 2).mean(dim=(0, 1, 3, 4))  # (T,)
    return wal_per_t, wal_per_t.mean()

def loss_wcl_complex(outputs, labels, eps=1e-8):
    """
    Wavelet Coherence Loss (WCL)
    - wavelet 도메인에서 코히어런스 손실 계산
    - 반환: (T별 loss, 전체 평균)
    """

    num = torch.real(outputs * labels.conj() + labels * outputs.conj()).sum(dim=(-2, -1))  # (B, C, T)
    denom = torch.sqrt(
        (torch.abs(outputs) ** 2).sum(dim=(-2, -1)) *
        (torch.abs(labels) ** 2).sum(dim=(-2, -1)) + eps
    )  # (B, C, T)

    corr = num / denom
    wcl_per_t = 1.0 - 0.5 * corr.mean(dim=(0, 1))  # (T,)
    return wcl_per_t, wcl_per_t.mean()
    
def loss_wacl_complex(outputs, labels, step, total_steps, alpha=0.1):
    """
    Wavelet Amplitude & Coherence Loss (WACL)
    - 학습 단계에 따라 WAL 또는 WCL을 확률적으로 선택
    - P_t = max(0, 1 - step / ((1 - alpha) * total_steps))
    - 초기에는 WCL 중심, 후기에는 WAL 중심으로 전환
    - 반환: (T별 loss, 전체 평균, 선택된 loss 이름)
    """
    P_t = max(0.0, 1.0 - step / ((1 - alpha) * total_steps))  # 논문 기반 스케줄

    if torch.rand(1).item() > P_t:
        loss_per_t, loss = loss_wal_complex(outputs, labels)
        return loss_per_t, loss, "WAL"
    else:
        loss_per_t, loss = loss_wcl_complex(outputs, labels)
        return loss_per_t, loss, "WCL"
    
def loss_wacl_complex_phase(outputs, labels, step, total_steps, alpha=0.1):
    """
    Wavelet Amplitude & Coherence Loss (WACL)
    - 학습 단계에 따라 WAL 또는 WCL을 확률적으로 선택
    - P_t = max(0, 1 - step / ((1 - alpha) * total_steps))
    - 초기에는 WCL 중심, 후기에는 WAL 중심으로 전환 [Phase signal은 계속 주입]
    - 반환: (T별 loss, 전체 평균, 선택된 loss 이름)
    """
    P_t = max(0.0, 1.0 - step / ((1 - alpha) * total_steps))  # 논문 기반 스케줄

    if torch.rand(1).item() > P_t:
        loss_per_t_amp, loss_amp = loss_wal_complex(outputs, labels)
        loss_per_t_phase, loss_phase = loss_wcl_complex(outputs, labels)

        loss_per_t = loss_per_t_amp + 0.2 * loss_per_t_phase
        loss = loss_amp + 0.2 * loss_phase
        return loss_per_t, loss, "WAL"
    else:
        loss_per_t, loss = loss_wcl_complex(outputs, labels)
        return loss_per_t, loss, "WCL"

def to_complex(x):
    """
    x: tensor of shape (B, C, H, W, 2), where x[..., 0] = real, x[..., 1] = imag
    returns: complex tensor of shape (B, C, H, W)
    """
    return torch.complex(x[..., 0], x[..., 1])
    
def extract_dtcwt_subbands_lv1(tensor):
    """
    (B, C, T, H, W) → 서브밴드 추출
    Returns:
        Low : (B, C, T, H, W)                # Real Number
        High[0 - 5] : (B, C, T, H//2, W//2)  # Complex Number
    """
    device = tensor.device
    dtcwt = DTCWTForward(J=1, biort="near_sym_b").to(device)
    B, C, T, H, W = tensor.shape
    Low_List, High0_list, High1_list, High2_list, High3_list, High4_list, High5_list = [], [], [], [], [], [], []

    for t in range(T):
        x_t = tensor[:, :, t] # (B, C, H, W)
        Yl, Yh = dtcwt(x_t)   # Yh[0] : (B, C, 6, H, W, 2)
        Low_List.append(Yl)
        High0_list.append(to_complex(Yh[0][:,:,0]))
        High1_list.append(to_complex(Yh[0][:,:,1]))
        High2_list.append(to_complex(Yh[0][:,:,2]))
        High3_list.append(to_complex(Yh[0][:,:,3]))
        High4_list.append(to_complex(Yh[0][:,:,4]))
        High5_list.append(to_complex(Yh[0][:,:,5]))

    # stack over time → (B, C, T, H//2, W//2)
    def stack_list(lst): return torch.stack(lst, dim=2)
    return stack_list(Low_List), stack_list(High0_list), stack_list(High1_list), stack_list(High2_list), stack_list(High3_list), stack_list(High4_list), stack_list(High5_list)

def extract_dtcwt_subbands_lv2(tensor):
    """
    DTCWT Level 2 Subband Extractor
    Input:
        tensor: (B, C, T, H, W)
    Returns:
        Low: (B, C, T, H//2, W//2)                      # Real
        High_lvl1[0~5]: (B, C, T, H//2, W//2)           # Complex
        High_lvl2[0~5]: (B, C, T, H//4, W//4)           # Complex
    """
    device = tensor.device
    dtcwt = DTCWTForward(J=2, biort="near_sym_b", qshift="qshift_b").to(device)
    B, C, T, H, W = tensor.shape

    Low_List = []
    High1_Lists = [[] for _ in range(6)]  # Level 1
    High2_Lists = [[] for _ in range(6)]  # Level 2

    for t in range(T):
        x_t = tensor[:, :, t]  # (B, C, H, W)
        Yl, Yh = dtcwt(x_t)    # Yh[0]: lvl1 (B,C,6,H/2,W/2,2), Yh[1]: lvl2 (B,C,6,H/4,W/4,2)

        Low_List.append(Yl)

        # Level 1 High subbands
        for i in range(6):
            High1_Lists[i].append(to_complex(Yh[0][:, :, i]))

        # Level 2 High subbands
        for i in range(6):
            High2_Lists[i].append(to_complex(Yh[1][:, :, i]))

    def stack_list(lst): return torch.stack(lst, dim=2)

    return (
        stack_list(Low_List),                             # (B, C, T, H//2, W//2)
        *[stack_list(High1_Lists[i]) for i in range(6)],  # (B, C, T, H//2, W//2)
        *[stack_list(High2_Lists[i]) for i in range(6)]   # (B, C, T, H//4, W//4)
    )

def extract_dtcwt_subbands_lv3(tensor):
    """
    DTCWT level 3 서브밴드 추출
    Input: (B, C, T, H, W)
    Returns:
        Low: (B, C, T, H//4, W//4)
        High L1[0~5]: (B, C, T, H//2, W//2)
        High L2[0~5]: (B, C, T, H//4, W//4)
        High L3[0~5]: (B, C, T, H//8, W//8)
    """
    device = tensor.device
    dtcwt = DTCWTForward(J=3, biort="near_sym_b", qshift="qshift_b").to(device)
    B, C, T, H, W = tensor.shape

    Low_List = []
    High1_Lists = [[] for _ in range(6)]
    High2_Lists = [[] for _ in range(6)]
    High3_Lists = [[] for _ in range(6)]

    for t in range(T):
        x_t = tensor[:, :, t]  # (B, C, H, W)
        Yl, Yh = dtcwt(x_t)    # Yh[0]: L1, Yh[1]: L2, Yh[2]: L3

        Low_List.append(Yl)

        for i in range(6):
            High1_Lists[i].append(to_complex(Yh[0][:, :, i]))
            High2_Lists[i].append(to_complex(Yh[1][:, :, i]))
            High3_Lists[i].append(to_complex(Yh[2][:, :, i]))

    def stack(lst): return torch.stack(lst, dim=2)

    return (
        stack(Low_List),
        *[stack(High1_Lists[i]) for i in range(6)],
        *[stack(High2_Lists[i]) for i in range(6)],
        *[stack(High3_Lists[i]) for i in range(6)],
    )


def loss_facl_dtcwt_lv1(outputs, labels, step, total_steps,alpha=0.1, beta=1, gamma=1):

    # DTCWT band extraction 
    # Level1
    lp, h0p, h1p, h2p, h3p, h4p, h5p = extract_dtcwt_subbands_lv1(outputs)
    lt, h0t, h1t, h2t, h3t, h4t, h5t = extract_dtcwt_subbands_lv1(labels)

    # Low Freq FACL loss
    _, low_facl_loss, _ = loss_facl(lp, lt, step=step, total_steps=total_steps, alpha=alpha)
    # High Freq FACL loss
    _, high0_facl_loss, _ = loss_wacl_complex(h0p, h0t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high1_facl_loss, _ = loss_wacl_complex(h1p, h1t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high2_facl_loss, _ = loss_wacl_complex(h2p, h2t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high3_facl_loss, _ = loss_wacl_complex(h3p, h3t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high4_facl_loss, _ = loss_wacl_complex(h4p, h4t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high5_facl_loss, _ = loss_wacl_complex(h5p, h5t, step=step,
                                              total_steps=total_steps, alpha=alpha)

    total = ((beta * low_facl_loss) +
             gamma * (high0_facl_loss + high1_facl_loss + high2_facl_loss +
             high3_facl_loss + high4_facl_loss + high5_facl_loss)) / 7.0
    return None, total, None

def loss_wacl_lv1(outputs, labels, step, total_steps,alpha=0.1, beta=1, gamma=1):

    # DTCWT band extraction 
    # Level1
    lp, h0p, h1p, h2p, h3p, h4p, h5p = extract_dtcwt_subbands_lv1(outputs)
    lt, h0t, h1t, h2t, h3t, h4t, h5t = extract_dtcwt_subbands_lv1(labels)

    # High Freq FACL loss
    _, high0_facl_loss, _ = loss_wacl_complex(h0p, h0t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high1_facl_loss, _ = loss_wacl_complex(h1p, h1t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high2_facl_loss, _ = loss_wacl_complex(h2p, h2t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high3_facl_loss, _ = loss_wacl_complex(h3p, h3t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high4_facl_loss, _ = loss_wacl_complex(h4p, h4t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high5_facl_loss, _ = loss_wacl_complex(h5p, h5t, step=step,
                                              total_steps=total_steps, alpha=alpha)

    total = (gamma * (high0_facl_loss + high1_facl_loss + high2_facl_loss +
             high3_facl_loss + high4_facl_loss + high5_facl_loss)) / 6.0
    return None, total, None
#전체에 대한 FACL loss + 고주파 6개의 밴드에 대한 WACL loss (밴드 분해는 DTCWT를 바탕으로)
def loss_facl_wacl_lv1(outputs, labels, step, total_steps,alpha=0.1, beta=1, gamma=1):

    # DTCWT band extraction 
    # Level1
    lp, h0p, h1p, h2p, h3p, h4p, h5p = extract_dtcwt_subbands_lv1(outputs)
    lt, h0t, h1t, h2t, h3t, h4t, h5t = extract_dtcwt_subbands_lv1(labels)

    # Global FACL Loss
    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    # High Freq WACL loss
    _, high0_facl_loss, _ = loss_wacl_complex(h0p, h0t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high1_facl_loss, _ = loss_wacl_complex(h1p, h1t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high2_facl_loss, _ = loss_wacl_complex(h2p, h2t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high3_facl_loss, _ = loss_wacl_complex(h3p, h3t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high4_facl_loss, _ = loss_wacl_complex(h4p, h4t, step=step,
                                              total_steps=total_steps, alpha=alpha)
    _, high5_facl_loss, _ = loss_wacl_complex(h5p, h5t, step=step,
                                              total_steps=total_steps, alpha=alpha)

    total = beta * global_loss + (gamma * (high0_facl_loss + high1_facl_loss + high2_facl_loss +
             high3_facl_loss + high4_facl_loss + high5_facl_loss)) / 6.0
    return None, total, None

def loss_facl_dtcwt_lv2(outputs, labels, step, total_steps, alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0):
    """
    DTCWT Level 2 기반 FACL Loss
    - 평균 없이 각 loss term 모두 합산
    - 고주파 term이 저주파에 비해 희석되지 않도록 구성
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: FACL soft blend factor
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)

    # Low-band loss (Real)
    _, low_loss, _ = loss_facl(lp, lt, step=step, total_steps=total_steps, alpha=alpha)

    # Level 1 High-band loss (Complex)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l

    # Level 2 High-band loss (Complex)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l

    # 전체 Loss 합산
    total_loss = (beta * low_loss + gamma1 * high1_loss + gamma2 * high2_loss) / 13.0

    return None, total_loss, None

def loss_wacl_lv2(outputs, labels, step, total_steps, alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0):
    """
    DTCWT Level 2 기반 FACL Loss
    - 평균 없이 각 loss term 모두 합산
    - 고주파 term이 저주파에 비해 희석되지 않도록 구성
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: FACL soft blend factor
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)

    # Level 1 High-band loss (Complex)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l

    # Level 2 High-band loss (Complex)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l

    # 전체 Loss 합산
    total_loss = (gamma1 * high1_loss + gamma2 * high2_loss) / 12.0

    return None, total_loss, None

def loss_facl_wacl_lv2(outputs, labels, step, total_steps, alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0):
    """
    DTCWT Level 2 기반 FACL Loss
    - 평균 없이 각 loss term 모두 합산
    - 고주파 term이 저주파에 비해 희석되지 않도록 구성
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: FACL soft blend factor
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)
    
    # Global loss
    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    # Level 1 High-band loss (Complex)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l

    # Level 2 High-band loss (Complex)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l

    # 전체 Loss 합산
    total_loss = beta * global_loss + (gamma1 * high1_loss + gamma2 * high2_loss) / 12.0

    return None, total_loss, None


def loss_facl_dtcwt_lv2_weighted(outputs, labels, step, total_steps, alpha=0.1, beta=0.5, gamma1=1.5, gamma2=1.0):
    """
    DTCWT Level 2 기반 FACL Loss
    - 평균 없이 각 loss term 모두 합산
    - 고주파 term이 저주파에 비해 희석되지 않도록 구성
    - FACL / WACL 에 phase signal 계속 주입
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: FACL soft blend factor
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)

    # Low-band loss (Real)
    _, low_loss, _ = loss_facl_phase(lp, lt, step=step, total_steps=total_steps, alpha=alpha)

    # Level 1 High-band loss (Complex)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex_phase(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l

    # Level 2 High-band loss (Complex)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex_phase(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l

    # 전체 Loss 합산
    total_loss = (beta * low_loss + gamma1 * high1_loss + gamma2 * high2_loss) / 13.0

    return None, total_loss, None

def loss_wacl_mse_lv2(outputs, labels, step, total_steps, alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0):
    """
    DTCWT Level 2 기반 WACL Loss
    - 평균 없이 각 loss term 모두 합산
    - 고주파 term이 저주파에 비해 희석되지 않도록 구성
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: WACL soft blend factor
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)

    # Low-band loss (Real)
    low_loss = F.mse_loss(lp, lt)

    # Level 1 High-band loss (Complex)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l

    # Level 2 High-band loss (Complex)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l

    # 전체 Loss 합산
    total_loss = (beta * low_loss + gamma1 * high1_loss + gamma2 * high2_loss) / 13.0

    return None, total_loss, None

import torch.nn.functional as F

def loss_wacl_mse_lv2_pixel(outputs, labels, step, total_steps,
                      alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0,
                      pixel_weight=1.0):
    """
    DTCWT Level 2 기반 WACL + Pixel MSE Loss
    - Pixel MSE 추가 (full-resolution)
    - 밴드별 normalization 포함
    - weight는 학습 단계(step)에 따라 조정 가능
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: WACL soft blend factor (phase vs amplitude)
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
        pixel_weight: Pixel-domain MSE 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)

    # Low-band loss
    low_loss = F.mse_loss(lp, lt)

    # Level 1 High-band loss (6방향 평균)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l
    high1_loss /= 6.0

    # Level 2 High-band loss (6방향 평균)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l
    high2_loss /= 6.0

    # Pixel-domain MSE (full resolution)
    pixel_loss = F.mse_loss(outputs, labels)

    # 스케줄링 예시 (선택적): 학습 후반으로 갈수록 high-band 비중↑
    progress = step / float(total_steps)
    beta_t  = beta * (1 - progress)       # low loss weight ↓
    gamma1_t = gamma1 * progress          # high1 weight ↑
    gamma2_t = gamma2 * progress          # high2 weight ↑

    # 최종 Loss 합산
    total_loss = (pixel_weight * pixel_loss
                 + beta_t * low_loss
                 + gamma1_t * high1_loss
                 + gamma2_t * high2_loss)

    return None, total_loss, None


def loss_wacl_mae_lv2(outputs, labels, step, total_steps, alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0):
    """
    DTCWT Level 2 기반 WACL Loss
    - 평균 없이 각 loss term 모두 합산
    - 고주파 term이 저주파에 비해 희석되지 않도록 구성
    Args:
        outputs, labels: (B, C, T, H, W)
        step, total_steps: scheduler용
        alpha: WACL soft blend factor
        beta: Low-band 가중치
        gamma1: Level 1 High-band 가중치
        gamma2: Level 2 High-band 가중치
    """
    # DTCWT subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2 = extract_dtcwt_subbands_lv2(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2 = extract_dtcwt_subbands_lv2(labels)

    # Low-band loss (Real)
    low_loss = F.l1_loss(lp, lt)

    # Level 1 High-band loss (Complex)
    high1_loss = 0.0
    for hp, ht in zip([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                      [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high1_loss += l

    # Level 2 High-band loss (Complex)
    high2_loss = 0.0
    for hp, ht in zip([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                      [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]):
        _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
        high2_loss += l

    # 전체 Loss 합산
    total_loss = (beta * low_loss + gamma1 * high1_loss + gamma2 * high2_loss) / 13.0

    return None, total_loss, None

def loss_facl_dtcwt_lv3(outputs, labels, step, total_steps,
                         alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0):
    """
    DTCWT level 3 기반 FACL Loss
    - 총 19개 subband
    """
    # Subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    # Low band
    _, low_loss, _ = loss_facl(lp, lt, step=step, total_steps=total_steps, alpha=alpha)

    # High bands
    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    # 전체 term 수 = 19
    total_loss = (beta * low_loss +
                  gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 19.0

    return None, total_loss, None

def loss_wacl_lv3(outputs, labels, step, total_steps,
                         alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0):
    """
    DTCWT level 3 기반 FACL Loss
    - 총 19개 subband
    """
    # Subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    # High bands
    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    # 전체 term 수 = 18
    total_loss = (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None

def loss_wacl_lv3_low(outputs, labels, step, total_steps,
                      alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0):
    """
    순수 Wavelet-only Loss (Fourier 없음) — loss_wacl_lv3의 저주파 포함 버전
    - 저주파(Low band, real): MSE
    - 고주파(18개 subband, complex): WACL
    - 총 19개 term / 19.0 정규화
    loss_facl_wacl_lv3와의 차이: global FACL(Fourier) 대신 low-band MSE 사용
    """
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    low_loss = F.mse_loss(lp, lt)

    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    total_loss = (beta * low_loss +
                  gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 19.0

    return None, total_loss, None


def loss_facl_wacl_lv3(outputs, labels, step, total_steps,
                         alpha=0.1, beta=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0):
    """
    DTCWT level 3 기반 FACL Loss
    - 총 19개 subband
    """
    # Subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    # Global loss
    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    # High bands
    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    # 전체 term 수 = 18
    total_loss = beta * global_loss + (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None


def loss_facl_wacl_lv3_w1(outputs, labels, step, total_steps,
                           alpha=0.1, beta=1.0,
                           gamma1=2.0, gamma2=0.5, gamma3=0.5):
    """
    loss_facl_wacl_lv3 + Level 1 (finest, highest-freq) 고주파 강조
    - gamma1=2.0, gamma2=0.5, gamma3=0.5 (기본값)
    """
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    total_loss = beta * global_loss + (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None


def loss_facl_wacl_lv3_w2(outputs, labels, step, total_steps,
                           alpha=0.1, beta=1.0,
                           gamma1=0.5, gamma2=2.0, gamma3=0.5):
    """
    loss_facl_wacl_lv3 + Level 2 (mid-freq) 고주파 강조
    - gamma1=0.5, gamma2=2.0, gamma3=0.5 (기본값)
    """
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    total_loss = beta * global_loss + (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None


def loss_facl_wacl_lv3_w3(outputs, labels, step, total_steps,
                           alpha=0.1, beta=1.0,
                           gamma1=0.5, gamma2=0.5, gamma3=2.0):
    """
    loss_facl_wacl_lv3 + Level 3 (coarsest, lowest-freq) 고주파 강조
    - gamma1=0.5, gamma2=0.5, gamma3=2.0 (기본값)
    """
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    total_loss = beta * global_loss + (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None


import torch.nn.functional as F

def loss_facl_wacl_lv3_mae(outputs, labels, step, total_steps,
                           alpha=0.1, beta=1.0,
                           gamma1=1.0, gamma2=1.0, gamma3=1.0,
                           w_mae=1.0):
    """
    FACL + WACL(DTCWT level-3) + Pixel-domain MAE
    - 기존 방식 유지: 고주파 18개 항(6방향 x 3레벨) 합산 후 /18.0 정규화
    """
    # Subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    # Global FACL loss
    _, facl_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    # High-band WACL 합산 (레벨/방향 모두 합, 평균 X)
    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss(
        [h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
        [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1]
    )
    high2_loss = sum_band_loss(
        [h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
        [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2]
    )
    high3_loss = sum_band_loss(
        [h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
        [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3]
    )

    # Pixel-domain MAE (전체 텐서 평균 절대오차)
    mae_loss = F.l1_loss(outputs, labels)

    # 최종 합산 (기존과 동일하게 /18.0 정규화 유지) + MAE 추가
    total_loss = (beta * facl_loss +
                  (gamma1 * high1_loss + gamma2 * high2_loss + gamma3 * high3_loss) / 18.0 +
                  w_mae * mae_loss)

    return None, total_loss, None

def loss_facl_wacl_lv3_alpha(outputs, labels, step, total_steps,
                         alpha=0.3, beta=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0):
    """
    DTCWT level 3 기반 FACL Loss
    - 총 19개 subband
    """
    # Subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    # Global loss
    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    # High bands
    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            _, l, _ = loss_wacl_complex(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    # 전체 term 수 = 18
    total_loss = beta * global_loss + (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None

def loss_wacl_complex_2(outputs, labels, step, total_steps, alpha=0.1):
    P_t = max(0.0, 1.0 - step / ((1 - alpha) * total_steps))

    # coherence
    _, loss_phase = loss_wcl_complex(outputs, labels)
    # amplitude
    _, loss_amp = loss_wal_complex(outputs, labels)

    return loss_phase, loss_amp

def loss_facl_wacl_lv3_phase_left(outputs, labels, step, total_steps,
                             alpha=0.3, beta=1.0, gamma1=1.0, gamma2=1.0, gamma3=1.0):
    """
    DTCWT level 3 기반 FACL+WACL Loss (phase 최소 비중 유지)
    - 70%까지는 phase 학습, 이후 amplitude 강화
    - 마지막까지 phase 비중을 0.2 이상 유지
    """
    # Subband 추출
    lp, h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1, \
        h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2, \
        h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3 = extract_dtcwt_subbands_lv3(outputs)

    lt, h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1, \
        h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2, \
        h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3 = extract_dtcwt_subbands_lv3(labels)

    # 스케줄링: phase 최소 0.2 유지
    progress = step / total_steps
    if progress < 0.7:
        w_phase = 1.0 - 0.3 * (progress / 0.7)  # 1.0 → 0.7
    else:
        w_phase = 0.2                           # 최소 유지
    w_amp = 1.0 - w_phase

    # Global loss (FACL에서 coherence vs amplitude 조합)
    _, global_loss, _ = loss_facl(outputs, labels, step=step, total_steps=total_steps, alpha=alpha)

    
    # High bands (WACL에서 coherence vs amplitude 조합)
    def sum_band_loss(hps, hts):
        total = 0.0
        for hp, ht in zip(hps, hts):
            loss_phase, loss_amp = loss_wacl_complex_2(hp, ht, step=step, total_steps=total_steps, alpha=alpha)
            l = w_phase * loss_phase + w_amp * loss_amp
            total += l
        return total

    high1_loss = sum_band_loss([h0p_l1, h1p_l1, h2p_l1, h3p_l1, h4p_l1, h5p_l1],
                                [h0t_l1, h1t_l1, h2t_l1, h3t_l1, h4t_l1, h5t_l1])
    high2_loss = sum_band_loss([h0p_l2, h1p_l2, h2p_l2, h3p_l2, h4p_l2, h5p_l2],
                                [h0t_l2, h1t_l2, h2t_l2, h3t_l2, h4t_l2, h5t_l2])
    high3_loss = sum_band_loss([h0p_l3, h1p_l3, h2p_l3, h3p_l3, h4p_l3, h5p_l3],
                                [h0t_l3, h1t_l3, h2t_l3, h3t_l3, h4t_l3, h5t_l3])

    # 전체 term 수 = 18
    total_loss = beta * global_loss + (gamma1 * high1_loss +
                  gamma2 * high2_loss +
                  gamma3 * high3_loss) / 18.0

    return None, total_loss, None






