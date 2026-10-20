# losses/loss_2d/losses.py
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DTCWTForward

# ---------- 기본 2D: MSE/MAE ----------

def loss_mse(outputs, labels):

    loss_per_batch = F.mse_loss(outputs, labels, reduction="none") # (B, C, H, W)
    loss_per_t = loss_per_batch.mean(dim=[0,2,3]) # (C,)

    return loss_per_t.mean()

def loss_mae(outputs, labels):

    loss_per_batch = F.l1_loss(outputs, labels, reduction="none") # (B, C, H, W)
    loss_per_t = loss_per_batch.mean(dim=[0,2,3]) # (C,)

    return loss_per_t.mean()

# ---------- FFT 기반: FAL/FCL/FACL ----------

def _fft2(x):
    return torch.fft.fft2(x, norm="ortho")

def loss_fal(outputs, labels):
    """
    Fourier Amplitude Loss (2D)
    """
    pred_fft = _fft2(outputs)          # (B,C,H,W)
    target_fft = _fft2(labels)

    pred_amp = torch.abs(pred_fft)     # (B,C,H,W)
    target_amp = torch.abs(target_fft)

    fal_per_t = ((pred_amp - target_amp) ** 2).mean(dim=(0,2,3)) # (C,)

    return fal_per_t.mean()

def loss_fcl(outputs, labels, eps=1e-8):
    """
    Fourier Coherence Loss (2D)
    """
    X = _fft2(outputs) # (B,C,H,W)
    Y = _fft2(labels)

    num = torch.real(X * Y.conj() + Y * X.conj()).sum(dim=(-2, -1))  # (B,C)
    den = torch.sqrt((torch.abs(X) ** 2).sum(dim=(-2, -1)) * (torch.abs(Y) ** 2).sum(dim=(-2, -1)) + eps)  # (B,C)

    corr = num / den  # (B,C)
    fcl_per_t = 1.0 - 0.5 * corr.mean(dim=(0))  # (C,)

    return fcl_per_t.mean()

def loss_facl(outputs, labels, step=None, total_steps=None, alpha=0.1):
    """
    2D 버전 FACL: step/total_steps 스케줄로 FCL↔FAL 확률 선택
    """
    P_t = max(0.0, 1.0 - float(step) / ((1 - alpha) * float(total_steps)))  # 논문 기반 스케줄

    if torch.rand(1).item() > P_t:
        return loss_fal(outputs, labels)  # FAL
    else:
        return loss_fcl(outputs, labels)  # FCL

# ---------- DWT (Haar, J=1) ----------

def extract_wavelet_subbands_2d(x):
    """
    x: (B,C,H,W) → (LL, LH, HL, HH), 각각 (B,C,H/2,W/2)
    """
    device = x.device
    dwt = DWTForward(J=1, mode='zero', wave='haar').to(device)

    Yl, Yh = dwt(x)  # Yh[0]: (B,C,3,H//2,W//2)
    LL = Yl
    LH = Yh[0][:, :, 0]
    HL = Yh[0][:, :, 1]
    HH = Yh[0][:, :, 2]

    return LL, LH, HL, HH # (B, C, H//2, W//2)

def _subband_mse_2d(a, b):

    return F.mse_loss(a, b)


def loss_fcl_ll(outputs, labels, alpha=1.0, beta=1.0):
    fcl = loss_fcl(outputs, labels)
    ll_p, _, _, _ = extract_wavelet_subbands_2d(outputs)
    ll_t, _, _, _ = extract_wavelet_subbands_2d(labels)

    ll = _subband_mse_2d(ll_p, ll_t)

    return alpha * fcl + beta * ll

def loss_fal_ll(outputs, labels, alpha=1.0, beta=1.0):
    fal = loss_fal(outputs, labels)
    ll_p, _, _, _ = extract_wavelet_subbands_2d(outputs)
    ll_t, _, _, _ = extract_wavelet_subbands_2d(labels)

    ll = _subband_mse_2d(ll_p, ll_t)

    return alpha * fal + beta * ll

def loss_fcl_high(outputs, labels, alpha=1.0, beta=1.0):
    fcl = loss_fcl(outputs, labels)
    _, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    _, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)

    high = (_subband_mse_2d(lh_p, lh_t) + _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 3.0

    return alpha * fcl + beta * high

def loss_fal_high(outputs, labels, alpha=1.0, beta=1.0):
    fal = loss_fal(outputs, labels)
    _, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    _, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)

    high = (_subband_mse_2d(lh_p, lh_t) + _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 3.0

    return alpha * fal + beta * high

def loss_fcl_all(outputs, labels, alpha=1.0, beta=1.0):
    fcl = loss_fcl(outputs, labels)
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    ll_t, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)

    wave = (_subband_mse_2d(ll_p, ll_t) + _subband_mse_2d(lh_p, lh_t) +
            _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 4.0
    
    return alpha * fcl + beta * wave

def loss_fal_all(outputs, labels, alpha=1.0, beta=1.0):
    fal = loss_fal(outputs, labels)
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    ll_t, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)

    wave = (_subband_mse_2d(ll_p, ll_t) + _subband_mse_2d(lh_p, lh_t) +
            _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 4.0
    
    return alpha * fal + beta * wave

def loss_ll_only(outputs, labels):
    ll_p, _, _, _ = extract_wavelet_subbands_2d(outputs)
    ll_t, _, _, _ = extract_wavelet_subbands_2d(labels)
    return _subband_mse_2d(ll_p, ll_t)

def loss_high_only(outputs, labels):
    _, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    _, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)
    return (_subband_mse_2d(lh_p, lh_t) + _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 3.0

def loss_wavelet_all(outputs, labels):
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    ll_t, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)
    return (_subband_mse_2d(ll_p, ll_t) + _subband_mse_2d(lh_p, lh_t) +
            _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 4.0

def loss_wavelet_mixed(outputs, labels, alpha=1.0, beta=1.0):
    ll_p, lh_p, hl_p, hh_p = extract_wavelet_subbands_2d(outputs)
    ll_t, lh_t, hl_t, hh_t = extract_wavelet_subbands_2d(labels)
    low = _subband_mse_2d(ll_p, ll_t)
    high = (_subband_mse_2d(lh_p, lh_t) + _subband_mse_2d(hl_p, hl_t) + _subband_mse_2d(hh_p, hh_t)) / 3.0
    return alpha * low + beta * high

# ---------- DTCWT (2D) ----------

def _to_complex(x):  # x: (...,2) → complex
    return torch.complex(x[..., 0], x[..., 1])

def extract_dtcwt_subbands_2d_lv1(x):
    """
    x: (B,C,H,W) → (Low, H0..H5)
      Low: (B,C,H,W)
      Hi : (B,C,H//2,W//2) complex (각 6방향)
    """
    device = x.device
    dt = DTCWTForward(J=1, biort="near_sym_b").to(device)

    Yl, Yh = dt(x)   # Yh[0]: (B,C,6,H/2,W/2,2)
    Low = Yl
    Hi = [_to_complex(Yh[0][:, :, k]) for k in range(6)]
    return (Low, *Hi)

def _facl_real_2d(a, b, step=None, total_steps=None, alpha=0.1):
    # real-band (LL) FACL
    return loss_facl(a, b, step=step, total_steps=total_steps, alpha=alpha)

def _facl_complex_2d(a, b, step=None, total_steps=None, alpha=0.1, eps=1e-8):
    # complex-band FACL: FAL/FCL를 complex 입력으로
    # FAL
    fal = ((torch.abs(a) - torch.abs(b)) ** 2).mean()
    # FCL
    num = torch.real(a * b.conj() + b * a.conj()).sum(dim=(-2, -1))
    den = torch.sqrt((torch.abs(a) ** 2).sum(dim=(-2, -1)) * (torch.abs(b) ** 2).sum(dim=(-2, -1)) + eps)
    corr = num / den
    fcl = 1.0 - 0.5 * corr.mean()

    P_t = max(0.0, 1.0 - float(step) / max(1.0, (1 - alpha) * float(total_steps)))

    return fal if (torch.rand(1).item() > P_t) else fcl

def loss_facl_dtcwt_lv1(outputs, labels, step=None, total_steps=None, alpha=0.1, beta=1.0, gamma=1.0):
    lp, h0p, h1p, h2p, h3p, h4p, h5p = extract_dtcwt_subbands_2d_lv1(outputs)
    lt, h0t, h1t, h2t, h3t, h4t, h5t = extract_dtcwt_subbands_2d_lv1(labels)

    low = _facl_real_2d(lp, lt, step=step, total_steps=total_steps, alpha=alpha)
    highs = [
        _facl_complex_2d(h0p, h0t, step=step, total_steps=total_steps, alpha=alpha),
        _facl_complex_2d(h1p, h1t, step=step, total_steps=total_steps, alpha=alpha),
        _facl_complex_2d(h2p, h2t, step=step, total_steps=total_steps, alpha=alpha),
        _facl_complex_2d(h3p, h3t, step=step, total_steps=total_steps, alpha=alpha),
        _facl_complex_2d(h4p, h4t, step=step, total_steps=total_steps, alpha=alpha),
        _facl_complex_2d(h5p, h5t, step=step, total_steps=total_steps, alpha=alpha),
    ]
    total = (beta * low + gamma * sum(highs)) / 7.0
    return total

