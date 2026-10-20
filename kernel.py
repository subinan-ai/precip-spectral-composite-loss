################1D kernel
import numpy as np
import matplotlib.pyplot as plt
import os

# Haar 1D 필터 정의
h = np.array([1/np.sqrt(2), 1/np.sqrt(2)])   # low-pass
g = np.array([1/np.sqrt(2), -1/np.sqrt(2)])  # high-pass

# 저장 폴더
save_dir = "loss_experiment_2/precipitation_loss_experiment-main-2/kernel"
os.makedirs(save_dir, exist_ok=True)

# Low-pass 필터 시각화 & 저장
plt.figure()
plt.stem(h, use_line_collection=True)
plt.title("Haar Low-pass (h)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.ylim(-1, 1)
plt.savefig(os.path.join(save_dir, "haar_lowpass.png"), dpi=150, bbox_inches="tight")
plt.close()

# High-pass 필터 시각화 & 저장
plt.figure()
plt.stem(g, use_line_collection=True)
plt.title("Haar High-pass (g)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.ylim(-1, 1)
plt.savefig(os.path.join(save_dir, "haar_highpass.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"저장 완료! -> {save_dir}/haar_lowpass.png, haar_highpass.png")
###########2D kernel
import os
import numpy as np
import matplotlib.pyplot as plt

# 1) Haar 1D 필터
h0 = np.array([1/np.sqrt(2),  1/np.sqrt(2)])   # low-pass
h1 = np.array([1/np.sqrt(2), -1/np.sqrt(2)])   # high-pass

def sep2d(a, b):
    """1D 필터 a(열), b(행)로 2D 분리형 커널 생성: a ⊗ b"""
    return np.outer(a, b)

# 2) 2D 커널 만들기
LL = sep2d(h0, h0)   # low-low
LH = sep2d(h1, h0)   # high-low  (세로 에지)
HL = sep2d(h0, h1)   # low-high  (가로 에지)
HH = sep2d(h1, h1)   # high-high (대각/세부)

# 3) 시각화 & 저장
save_dir = "loss_experiment_2/precipitation_loss_experiment-main-2/kernel"
os.makedirs(save_dir, exist_ok=True)

kernels = [("LL = h0 ⊗ h0", LL),
           ("LH = h1 ⊗ h0", LH),
           ("HL = h0 ⊗ h1", HL),
           ("HH = h1 ⊗ h1", HH)]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), constrained_layout=True)
for ax, (name, K) in zip(axes, kernels):
    im = ax.imshow(K, cmap="RdBu", origin="upper")
    ax.set_title(name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03)
plt.savefig(os.path.join(save_dir, "haar_2d_kernels.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"저장 완료: {os.path.join(save_dir, 'haar_2d_kernels.png')}")
###############2D frequency kernel
def freq_mag2d(K, size=64):
    # 작은 커널을 보기 좋게 패딩 후 스펙트럼
    pad = np.zeros((size, size))
    kh, kw = K.shape
    pad[:kh, :kw] = K
    F = np.fft.fftshift(np.fft.fft2(pad))
    return np.abs(F)

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), constrained_layout=True)
for ax, (name, K) in zip(axes, kernels):
    mag = freq_mag2d(K, size=64)
    im = ax.imshow(np.log1p(mag), cmap="magma", origin="upper")
    ax.set_title(name+" (|F|)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.04, pad=0.03)
plt.savefig(os.path.join(save_dir, "haar_2d_kernels_freq.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"저장 완료: {os.path.join(save_dir, 'haar_2d_kernels_freq.png')}")
############1D frequency kernel
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------
# Haar 1D filters
# -----------------
h0 = np.array([1/np.sqrt(2),  1/np.sqrt(2)])   # Low-pass (scaling)
h1 = np.array([1/np.sqrt(2), -1/np.sqrt(2)])   # High-pass (wavelet)

def freq_response(h, nfft=4096):
    """
    1D FIR 필터 h의 주파수 응답 H(e^{jw})를 numpy FFT로 계산.
    - 반환: w (0..pi, rad), |H|, angle(H) (unwrap)
    """
    # 0..pi 만 그릴 것이므로 rfft 사용 (실수 입력)
    L = len(h)
    n = max(nfft, 2**int(np.ceil(np.log2(L))))  # 충분히 촘촘한 스펙트럼
    H = np.fft.rfft(h, n=n)
    w = np.linspace(0, np.pi, H.size)          # 정규화 각주파수 (rad/sample)
    mag = np.abs(H)
    ang = np.unwrap(np.angle(H))
    return w, mag, ang

def plot_filter_and_response(name, h, save_dir="loss_experiment_2/precipitation_loss_experiment-main-2/kernel"):
    os.makedirs(save_dir, exist_ok=True)

    w, mag, ang = freq_response(h, nfft=4096)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)

    # (1) 임펄스 응답(계수)
    ax = axes[0]
    ax.stem(np.arange(len(h)), h, use_line_collection=True)
    ax.set_title(f"{name} – Taps")
    ax.set_xlabel("Index"); ax.set_ylabel("Value")
    ax.set_ylim(1.1*min(-1, h.min()), 1.1*max(1, h.max()))
    ax.grid(alpha=0.3)

    # (2) 진폭 응답 (선형 스케일)
    ax = axes[1]
    ax.plot(w/np.pi, mag)                     # 정규화 주파수 (0..1) = 0..Nyquist
    ax.set_title(f"{name} – |H(e^{{jω}})|")
    ax.set_xlabel("Normalized Frequency (×π rad/sample)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, 1); ax.grid(alpha=0.3)

    # (3) 위상 응답
    ax = axes[2]
    ax.plot(w/np.pi, ang)
    ax.set_title(f"{name} – ∠H(e^{{jω}})")
    ax.set_xlabel("Normalized Frequency (×π rad/sample)")
    ax.set_ylabel("Phase (rad)")
    ax.set_xlim(0, 1); ax.grid(alpha=0.3)

    out = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_1d_resp.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

# 실행: Haar low/high
plot_filter_and_response("Haar Low-pass (h)", h0)
plot_filter_and_response("Haar High-pass (g)", h1)
#################
import os
import numpy as np
import matplotlib.pyplot as plt
# 기존(에러)
# from pytorch_wavelets.dtcwt.transform2d import biort, qshift

# 교체(이 버전에서 동작)
from pytorch_wavelets.dtcwt.coeffs import biort, qshift

print([len(x) for x in biort("near_sym_b")])  # 예: [h0a_len, h1a_len, h0b_len, h1b_len]

# --- DTCWT biort/qshift 로더 (버전 호환) ---
import importlib
import numpy as np

def _resolve_dtcwt_loaders():
    """
    반환: (load_biort, load_qshift)  # 두 함수 핸들
    """
    candidates = [
        "pytorch_wavelets.dtcwt.transform2d",  # 최신
        "pytorch_wavelets.dtcwt.transform",    # 구버전
    ]
    last_err = None
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "biort"):
                load_biort = getattr(mod, "biort")
            else:
                continue
            # qshift는 level>=2에서만 필요하지만 함께 반환
            load_qshift = getattr(mod, "qshift", None)
            return load_biort, load_qshift
        except ImportError as e:
            last_err = e
            continue
    raise ImportError(
        "pytorch_wavelets에서 biort/qshift를 찾을 수 없습니다. "
        "패키지 버전을 확인하세요."
    ) from last_err


from pytorch_wavelets.dtcwt.coeffs import biort, qshift

def load_biort_filters(biort_name="near_sym_b", fallback="haar"):
    import numpy as np
    try:
        # 최신 버전은 함수식 → 바로 호출
        h0a, h1a, h0b, h1b = biort(biort_name)
        to_np = lambda x: np.asarray(x, dtype=float)
        return to_np(h0a), to_np(h1a), to_np(h0b), to_np(h1b)
    except Exception as e:
        if fallback is None:
            raise
        print(f"[WARN] biort('{biort_name}') 로드 실패: {e} → Haar로 폴백")
        s = 1/np.sqrt(2.0)
        h0 = np.array([ s,  s]); h1 = np.array([ s, -s])
        return h0, h1, h0, h1
h0a, h1a, h0b, h1b = load_biort_filters("near_sym_b")
print("Tree A LP:", h0a)
print("Tree A HP:", h1a)
print("Tree B LP:", h0b)
print("Tree B HP:", h1b)



def plot_tree_filters(biort="near_sym_b", save_dir="loss_experiment_2/precipitation_loss_experiment-main-2/kernel"):
    os.makedirs(save_dir, exist_ok=True)
    h0a, h1a, h0b, h1b = load_biort_filters(biort)

    filters = {
        "TreeA_LP (h0a)": h0a,
        "TreeA_HP (h1a)": h1a,
        "TreeB_LP (h0b)": h0b,
        "TreeB_HP (h1b)": h1b,
    }

    for name, f in filters.items():
        plt.figure(figsize=(4, 2))
        plt.stem(f, use_line_collection=True)
        plt.axhline(0, color="red", linewidth=0.8)
        plt.title(name)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.5)
        save_path = os.path.join(save_dir, f"{name.replace(' ','_')}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved {save_path}")
if __name__ == "__main__":
    out_root = "loss_experiment_2/precipitation_loss_experiment-main-2/kernel"
    for biort_name in ["near_sym_b", "farras"]:
        plot_tree_filters(
            biort=biort_name,
            save_dir=os.path.join(out_root, biort_name)
        )
#########################
# ======================= NEW: 박스 안에 LP/HP를 '그림'으로 꽂는 시각화 =======================
def _make_bar_image(f, height=140, width_per_tap=14, vabs=None, zero_bar_px=2):
    """
    1D 필터 f를 0선 기준 위/아래로 채운 '바 이미지'로 변환(축/눈금 없음).
    양수=위(빨강), 음수=아래(파랑) → imshow(cmap='coolwarm', vmin=-1, vmax=1)로 표시.
    """
    f = np.asarray(f, float).squeeze().reshape(-1)
    L = f.size
    H = int(height)
    W = int(L * width_per_tap)
    vmax = (np.max(np.abs(f)) if vabs is None else float(vabs)) + 1e-12

    img = np.zeros((H, W), dtype=float)  # 0=중립색
    mid = H // 2
    # 중앙 0선(얇게)
    if zero_bar_px > 0:
        img[mid-zero_bar_px//2:mid+zero_bar_px//2+1, :] = 0.0

    # 각 탭을 세로 막대로 채우기 (양수 위 / 음수 아래)
    for i, v in enumerate(f):
        w0, w1 = i * width_per_tap, (i + 1) * width_per_tap
        h = int(np.clip(abs(v) / vmax, 0, 1) * (mid - 3))
        if v >= 0:
            img[mid - h:mid, w0:w1] = (+abs(v) / vmax)
        else:
            img[mid:mid + h, w0:w1] = (-abs(v) / vmax)

    # 가장자리 패딩 약간
    img = np.pad(img, ((6, 6), (6, 6)), mode="constant", constant_values=0.0)
    return img

def _draw_tree_panel(ax, title, lp_img, hp_img, cmap="coolwarm"):
    """Tree 박스(굵은 테두리) 안에 LP(위)/HP(아래) 이미지를 인셋으로 배치."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(3.0)
        sp.set_color("#0E6071")  # 템플릿과 비슷한 청록색 테두리
    ax.set_title(title, fontsize=22, pad=10)

    # LP 위쪽, HP 아래쪽
    ax_lp = ax.inset_axes([0.05, 0.55, 0.90, 0.38])   # [left, bottom, width, height]
    ax_hp = ax.inset_axes([0.05, 0.10, 0.90, 0.38])
    for a, img, ttl in [(ax_lp, lp_img, "LP (h0)"), (ax_hp, hp_img, "HP (h1)")]:
        a.imshow(img, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        a.set_xticks([]); a.set_yticks([]); a.set_frame_on(False)
        a.text(0.5, 1.02, ttl, ha="center", va="bottom", transform=a.transAxes, fontsize=12)

def plot_tree_boxes_with_filters(
    biort_name="near_sym_b",
    save_path="loss_experiment_2/precipitation_loss_experiment-main-2/kernel/tree_panel.png",
    height=140,            # 막대 높이
    width_per_tap=14,      # 탭당 가로폭(=필터 길이 차이가 자연스레 폭으로 보임)
    cmap="coolwarm"
):
    # 1) 필터 로드 (네가 만든 로더 재사용)
    h0a, h1a, h0b, h1b = load_biort_filters(biort_name)

    # 2) 모든 필터를 같은 스케일로 보기 위해 공통 vabs 계산
    vabs = max(np.max(np.abs(h0a)), np.max(np.abs(h1a)),
               np.max(np.abs(h0b)), np.max(np.abs(h1b)))

    # 3) 바 이미지 생성
    img_h0a = _make_bar_image(h0a, height=height, width_per_tap=width_per_tap, vabs=vabs)
    img_h1a = _make_bar_image(h1a, height=height, width_per_tap=width_per_tap, vabs=vabs)
    img_h0b = _make_bar_image(h0b, height=height, width_per_tap=width_per_tap, vabs=vabs)
    img_h1b = _make_bar_image(h1b, height=height, width_per_tap=width_per_tap, vabs=vabs)

    # 4) 두 박스(좌: Tree a / 우: Tree b)에 꽂기
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), constrained_layout=True)
    _draw_tree_panel(axes[0], "Tree a", img_h0a, img_h1a, cmap=cmap)
    _draw_tree_panel(axes[1], "Tree b", img_h0b, img_h1b, cmap=cmap)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved → {save_path}")
if __name__ == "__main__":
    out_root = "loss_experiment_2/precipitation_loss_experiment-main-2/kernel"
    for biort_name in ["near_sym_b", "farras"]:
        # 기존 stem 플롯(그래프) 저장
        plot_tree_filters(
            biort=biort_name,
            save_dir=os.path.join(out_root, biort_name)
        )
        # 새 패널(그림) 저장: Tree a/b 박스 안에 LP/HP를 이미지로 배치
        panel_path = os.path.join(out_root, biort_name, "tree_panel.png")
        plot_tree_boxes_with_filters(
            biort_name=biort_name,
            save_path=panel_path,
            height=140,
            width_per_tap=14,
            cmap="coolwarm",
        )
   
# ======================= /NEW =================================================

###########################
import os
import numpy as np
import matplotlib.pyplot as plt

# 최신 pytorch_wavelets 레이아웃 (biort/qshift는 여기 있어요)
from pytorch_wavelets.dtcwt.coeffs import biort  # qshift는 레벨2+에서

# 1D 필터를 "그림"으로 저장 (축/프레임 없이)
def save_filter_strip_img(name, f, outdir,
                          height=64, repeat=10, pad=4,
                          cmap="coolwarm", vabs=None):
    """
    f: 1D numpy array (필터 계수)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # >>> 중요한 수정: 무조건 1D로 평탄화 <<<
    f = np.asarray(f, dtype=float).squeeze()
    if f.ndim != 1:
        f = f.reshape(-1)  # 어떤 형태여도 1D로 강제

    # (H, L)
    strip = np.repeat(f[np.newaxis, :], height, axis=0)
    # (H, L*repeat)
    strip = np.repeat(strip, repeat, axis=1)

    if pad > 0:
        # 2D 배열용 패딩 지정
        strip = np.pad(strip, ((0, 0), (pad, pad)), mode="constant")

    vmax = (np.max(np.abs(f)) if vabs is None else float(vabs)) + 1e-12
    path = os.path.join(outdir, f"{name}.png")
    plt.imsave(path, strip, cmap=cmap, vmin=-vmax, vmax=vmax)
    print(f"[saved] {path}")


# 2D 분리형 커널(외적) 이미지를 저장 (옵션)
def save_kernel_img(name, a, b, outdir, cmap="RdBu", vabs=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    a = np.asarray(a, float).squeeze()
    b = np.asarray(b, float).squeeze()
    if a.ndim != 1: a = a.reshape(-1)
    if b.ndim != 1: b = b.reshape(-1)

    K = np.outer(a, b)
    vmax = (np.max(np.abs(K)) if vabs is None else float(vabs)) + 1e-12
    path = os.path.join(outdir, f"{name}.png")
    plt.imsave(path, K, cmap=cmap, vmin=-vmax, vmax=vmax)
    print(f"[saved] {path}")


def render_dtcwt_level1_filters_as_images(biort_name="near_sym_b", outdir="loss_experiment_2/precipitation_loss_experiment-main-2/kernel"):
    # 1) 레벨1 biorthogonal 계수 (Tree a/b)
    h0a, h1a, h0b, h1b = biort(biort_name)  # 저역/고역 × 트리 a/b
    # 1D 스트립 공통 스케일(비교 가능하게)
    vabs_1d = max(np.max(np.abs(h0a)), np.max(np.abs(h1a)),
                  np.max(np.abs(h0b)), np.max(np.abs(h1b)))

    # 2) 1D 필터를 "그림"으로 저장
    save_filter_strip_img("TreeA_LP_h0a", h0a, outdir, vabs=vabs_1d)
    save_filter_strip_img("TreeA_HP_h1a", h1a, outdir, vabs=vabs_1d)
    save_filter_strip_img("TreeB_LP_h0b", h0b, outdir, vabs=vabs_1d)
    save_filter_strip_img("TreeB_HP_h1b", h1b, outdir, vabs=vabs_1d)

    # 3) (옵션) 레벨1 2D 커널도 같이 저장하고 싶다면:
    # LL, LH, HL, HH (Tree a/b는 1D에서 이미 보여주니, 커널은 한 세트만 저장해도 설명 충분)
    save_kernel_img("LL_h0a_otimes_h0a", h0a, h0a, outdir)  # 저역
    save_kernel_img("LH_h1a_otimes_h0a", h1a, h0a, outdir)  # 수직 에지
    save_kernel_img("HL_h0a_otimes_h1a", h0a, h1a, outdir)  # 수평 에지
    save_kernel_img("HH_h1a_otimes_h1a", h1a, h1a, outdir)  # 대각 성분

# -------- 실행 --------
if __name__ == "__main__":
    render_dtcwt_level1_filters_as_images(
        biort_name="near_sym_b",   # farras, near_sym_b 등 교체 가능
        outdir="loss_experiment_2/precipitation_loss_experiment-main-2/kernel"            # 그림이 저장될 폴더
    )
#####################
import os, numpy as np, matplotlib.pyplot as plt
from pytorch_wavelets.dtcwt.coeffs import biort  # 레벨1 biorthogonal 계수

# 1) 0선(중앙)에서 위/아래로 자라는 막대 '그림' 저장
def save_filter_bar_image(name, f, outdir,
                          height=160, width_per_tap=18,
                          cmap="coolwarm", vabs=None):
    """
    f: 1D 필터 계수 배열. 축/눈금 없이 '이미지'로 저장.
    - 중앙 가로선(0선) 기준으로 양수는 위로, 음수는 아래로 채움.
    - 막대 폭은 width_per_tap, 높이는 |계수|에 비례.
    """
    os.makedirs(outdir, exist_ok=True)
    f = np.asarray(f, float).squeeze().reshape(-1)
    L = f.size
    H = int(height)
    W = int(L * width_per_tap)

    vmax = (np.max(np.abs(f)) if vabs is None else float(vabs)) + 1e-12
    img = np.zeros((H, W), dtype=float)  # 0 → 중립색

    mid = H // 2
    # 얇은 0선(중앙 가로선)
    img[mid-1:mid+1, :] = 0.0

    for i, v in enumerate(f):
        w0, w1 = i*width_per_tap, (i+1)*width_per_tap
        h = int(np.clip(abs(v) / vmax, 0, 1) * (mid-3))
        if v >= 0:
            img[mid-h:mid, w0:w1] =  +abs(v) / vmax  # 위쪽(양수)
        else:
            img[mid:mid+h, w0:w1] =  -abs(v) / vmax  # 아래쪽(음수)

    # 가장자리 약간 여백(선택)
    pad = max(4, width_per_tap//3)
    img = np.pad(img, ((0,0),(pad,pad)), mode="constant", constant_values=0.0)

    path = os.path.join(outdir, f"{name}_bars.png")
    plt.imsave(path, img, cmap=cmap, vmin=-1, vmax=+1)
    print(f"[saved] {path}")

# 2) 1D 주파수 응답(|H(ω)|)을 스트립 이미지로 저장 (축/눈금 없음)
def save_filter_freq_strip(name, f, outdir, nfft=4096, height=80, cmap="magma"):
    """
    |FFT| (0..π) 를 가로축으로 늘여서 '그림'으로 저장.
    """
    os.makedirs(outdir, exist_ok=True)
    f = np.asarray(f, float).squeeze().reshape(-1)
    # 위상은 버리고 크기만(직관적)
    H = np.fft.rfft(f, n=nfft)
    mag = np.abs(H) / (np.max(np.abs(H)) + 1e-12)   # 정규화
    # (height x width) 스트립 이미지로 복제
    strip = np.repeat(mag[np.newaxis, :], height, axis=0)
    path = os.path.join(outdir, f"{name}_freq.png")
    plt.imsave(path, strip, cmap=cmap, vmin=0.0, vmax=1.0)
    print(f"[saved] {path}")

# 3) Tree a/b 필터를 직관 이미지로 한꺼번에 저장
def render_filters_intuitive(biort_name="near_sym_b", outdir="loss_experiment_2/precipitation_loss_experiment-main-2/kernel/filters_intuitive"):
    os.makedirs(outdir, exist_ok=True)
    h0a, h1a, h0b, h1b = biort(biort_name)

    # 모든 1D 바 이미지를 같은 스케일로 보기 위해 공통 vabs 사용
    vabs = max(np.max(np.abs(h0a)), np.max(np.abs(h1a)),
               np.max(np.abs(h0b)), np.max(np.abs(h1b)))

    for tag, h in [("TreeA_LP_h0a", h0a), ("TreeA_HP_h1a", h1a),
                   ("TreeB_LP_h0b", h0b), ("TreeB_HP_h1b", h1b)]:
        save_filter_bar_image(tag, h, outdir, vabs=vabs)     # 0선 기준 막대
        save_filter_freq_strip(tag, h, outdir)               # 주파수 응답 스트립

# ---------------- run ----------------
render_filters_intuitive("near_sym_b", outdir="loss_experiment_2/precipitation_loss_experiment-main-2/kernel/filters_intuitive")

############################
# ======================= NEW: 더 직관적인 1D(시간/주파수) 시각화 =======================
def _freq_resp_1d(h, nfft=4096):
    """0..π 구간의 1D 크기 응답 |H(ω)| (정규화)"""
    h = np.asarray(h, float).squeeze().reshape(-1)
    H = np.fft.rfft(h, n=nfft)                           # 0..π
    w = np.linspace(0, np.pi, H.shape[0])
    mag = np.abs(H)
    mag /= mag.max() + 1e-12
    return w, mag

def _nice_pi_ticks(ax):
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(["0", "π/2", "π"])

def _draw_tree_timefreq(ax_box, title, h0, h1):
    """큰 박스(ax_box) 안에 2x2 인셋: (LP/HP) x (time/freq)"""
    # 박스 스타일
    ax_box.set_xticks([]); ax_box.set_yticks([])
    for sp in ax_box.spines.values():
        sp.set_linewidth(3.0); sp.set_color("#0E6071")
    ax_box.set_title(title, fontsize=26, pad=12)

    # 레이아웃: 위/아래 2행, 좌/우 2열
    # (좌상) LP time, (우상) HP time, (좌하) LP |H(ω)|, (우하) HP |H(ω)|
    insets = {
        "lp_time": ax_box.inset_axes([0.06, 0.56, 0.40, 0.36]),
        "hp_time": ax_box.inset_axes([0.54, 0.56, 0.40, 0.36]),
        "lp_freq": ax_box.inset_axes([0.06, 0.10, 0.40, 0.36]),
        "hp_freq": ax_box.inset_axes([0.54, 0.10, 0.40, 0.36]),
    }

    # ---- 시간 영역 stem (LP)
    ax = insets["lp_time"]
    x = np.arange(len(h0))
    ax.stem(x, h0, basefmt="k-", linefmt="C0-", markerfmt="C0o", use_line_collection=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("LP  (time)", fontsize=12)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("tap"); ax.set_ylabel("value")

    # ---- 시간 영역 stem (HP)
    ax = insets["hp_time"]
    x = np.arange(len(h1))
    ax.stem(x, h1, basefmt="k-", linefmt="C1-", markerfmt="C1o", use_line_collection=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("HP  (time)", fontsize=12)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("tap"); ax.set_ylabel("value")

    # ---- 주파수 응답 (LP)
    ax = insets["lp_freq"]
    w, mag = _freq_resp_1d(h0)
    ax.plot(w, mag, lw=2)
    _nice_pi_ticks(ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("LP  |H(ω)|", fontsize=12)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("ω"); ax.set_ylabel("magnitude")

    # ---- 주파수 응답 (HP)
    ax = insets["hp_freq"]
    w, mag = _freq_resp_1d(h1)
    ax.plot(w, mag, lw=2, color="C1")
    _nice_pi_ticks(ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("HP  |H(ω)|", fontsize=12)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("ω"); ax.set_ylabel("magnitude")

def plot_tree_boxes_1d_timefreq(
    biort_name="near_sym_b",
    save_path="loss_experiment_2/precipitation_loss_experiment-main-2/kernel/tree_timefreq.png"
):
    # (네가 이미 만든 로더를 사용)
    h0a, h1a, h0b, h1b = load_biort_filters(biort_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    _draw_tree_timefreq(axes[0], "Tree a", h0a, h1a)
    _draw_tree_timefreq(axes[1], "Tree b", h0b, h1b)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved → {save_path}")
    # 기존 그래프/이미지 저장들 …
panel_path = os.path.join(out_root, biort_name, "tree_timefreq.png")
plot_tree_boxes_1d_timefreq(biort_name=biort_name, save_path=panel_path)
############################
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from pytorch_wavelets import DTCWTForward
from utils.config_utils import load_config
from datasets.dataloader import get_test_loader

# ---------------------- 유틸 함수들 ----------------------
def to_5d_BCTHW(x: torch.Tensor) -> torch.Tensor:
    """입력 텐서를 항상 (B,C,T,H,W)로 변환"""
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        B, d1, H, W = x.shape
        if d1 <= 4:  # 채널일 가능성이 높음
            return x.unsqueeze(2)   # (B,C,1,H,W)
        else:         # 시간일 가능성이 높음
            return x.unsqueeze(1)   # (B,1,T,H,W)
    raise ValueError(f"Expected 4D/5D tensor, got {x.shape}")

def to_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    if x.shape[-1] == 2:
        return torch.view_as_complex(x)
    raise ValueError(f"to_complex: unexpected input {x.shape}, {x.dtype}")

def extract_dtcwt_subbands_lv1(x, biort="near_sym_b"):
    """
    x: (B,C,T,H,W) or (B,C,H,W) or (B,T,H,W)
    return: Low, H0..H5 (모두 (B,C,T,H//2,W//2))
    """
    x = to_5d_BCTHW(x)
    device = x.device
    B, C, T, H, W = x.shape

    if (H % 2) or (W % 2):
        raise ValueError(f"DTCWT lv1 needs even H,W; got {H}, {W}")

    dtcwt = DTCWTForward(J=1, biort=biort).to(device)
    Low_list = []
    H_lists = [[] for _ in range(6)]

    for t in range(T):
        xt = x[:, :, t]   # (B,C,H,W)
        Yl, Yh = dtcwt(xt)
        h = Yh[0]         # (B,C,6,H//2,W//2,2)
        Low_list.append(Yl)
        for k in range(6):
            H_lists[k].append(to_complex(h[:, :, k]))

    stack = lambda lst: torch.stack(lst, dim=2)   # (B,C,T,H//2,W//2)
    Low = stack(Low_list)
    Highs = [stack(H_lists[k]) for k in range(6)]
    return (Low, *Highs)

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

from matplotlib.ticker import FuncFormatter

def custom_lon_formatter(x, pos):
    """
    경도 포매터:
    - 180 → '180°'
    - 0 → '0°'
    - 양수 → 'E'
    - 음수 → 'W'
    """
    if x == 180 or x == -180:
        return "180°"
    elif x == 0:
        return "0°"
    elif x > 0 and x < 180:
        return f"{int(x)}°E"
    elif x < 0 and x > -180:
        return f"{int(abs(x))}°W"
    else:
        return f"{int(x)}°"

def plot_global_map(data2d, title, cmap, vmin=None, vmax=None,
                    extent=(0,360,-90,90), central_longitude=180, save=None,
                    tick_fontsize=12, cbar_fontsize=12, title_fontsize=14):
    H, W = data2d.shape
    lons = np.linspace(extent[0], extent[1], W)
    lats = np.linspace(extent[2], extent[3], H)

    fig = plt.figure(figsize=(9,4), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    m = ax.pcolormesh(lons, lats, data2d, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax)

    ax.coastlines(linewidth=0.6, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="gray")

    # ---- gridlines ----
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = FuncFormatter(custom_lon_formatter)  # ← 여기 수정
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylocator  = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])  

    # 축 레이블 글자 크기
    gl.xlabel_style = {"size": tick_fontsize}
    gl.ylabel_style = {"size": tick_fontsize}

    # 제목
    plt.title(title, fontsize=title_fontsize)

    # 컬러바
    cbar = plt.colorbar(m, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=cbar_fontsize)

    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



# -*- coding: utf-8 -*-
"""
DTCWT 레벨1(Tree a/b) LP/HP 필터를
- 동일한 가로 스케일(가로 폭/Index 범위)로
- 개별 PNG + 패널 PNG 로 저장

필수: pip install pytorch-wavelets
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pytorch_wavelets.dtcwt.coeffs import biort  # 레벨1 계수


# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ★★★ 핵심: DTCWTForward에서 필터를 직접 꺼낸다 ★★★
from pytorch_wavelets import DTCWTForward


# ------------------------- 공통 유틸 ------------------------- #
def pad_center_to_length(f: np.ndarray, Lref: int) -> np.ndarray:
    f = np.asarray(f, float).squeeze().reshape(-1)
    if f.size == Lref:
        return f
    if f.size > Lref:
        raise ValueError(f"pad_center_to_length: len(f)={f.size} > Lref={Lref}")
    left = (Lref - f.size) // 2
    right = Lref - f.size - left
    return np.pad(f, (left, right), mode="constant")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def _xcorr_center(a: np.ndarray, b: np.ndarray) -> float:
    """중심 지연에서의 정규화 교차상관(대략적인 유사도 확인용)"""
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    L = max(len(a), len(b))
    a = pad_center_to_length(a, L)
    b = pad_center_to_length(b, L)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


# ------------------------- 필터 로더 (수정) ------------------------- #
def load_biort_filters(biort_name: str = "near_sym_b") -> Tuple[np.ndarray, ...]:
    """
    DTCWTForward 인스턴스에서 분석(analysis) 필터를 직접 추출:
      - h0a: Tree a low-pass analysis
      - h1a: Tree a high-pass analysis
      - h0b: Tree b low-pass analysis
      - h1b: Tree b high-pass analysis
    """
    # J=1로 생성하면 레벨1의 biorthogonal 필터 세트가 로딩된다.
    fw = DTCWTForward(J=1, biort=biort_name)
    # torch 텐서 → numpy
    to_np = lambda t: np.asarray(t.detach().cpu().numpy(), dtype=float).squeeze()
    h0a = to_np(fw.h0a)  # (len,)
    h1a = to_np(fw.h1a)
    h0b = to_np(fw.h0b)
    h1b = to_np(fw.h1b)

    # ---- 무결성 체크: LP와 HP가 너무 비슷하면 경고 ----
    cos_a = _cosine(h0a, h1a)
    cos_b = _cosine(h0b, h1b)
    if cos_a > 0.85 or cos_b > 0.85:
        print(f"[WARN] LP/HP look too similar (cos a={cos_a:.3f}, b={cos_b:.3f}). "
              f"Check filter names/order.")
    else:
        print(f"[OK] filters loaded. cos(LP,HP): a={cos_a:.3f}, b={cos_b:.3f}")
    return h0a, h1a, h0b, h1b


# --------------------- 저장 함수 (가로축 동일) --------------------- #
def save_stem_png(name: str, f: np.ndarray, outdir: str, Lref: int,
                  ylim: Tuple[float, float] = None) -> str:
    os.makedirs(outdir, exist_ok=True)
    f = pad_center_to_length(f, Lref)
    x = np.arange(Lref)

    plt.figure(figsize=(4.6, 2.8))
    markerline, stemlines, baseline = plt.stem(x, f, basefmt="k-", use_line_collection=True)
    plt.setp(markerline, markersize=6)
    plt.axhline(0, color="red", lw=0.9)
    plt.xlim(0, Lref - 1)
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        vmax = max(0.5, float(np.max(np.abs(f))) * 1.2)
        plt.ylim(-vmax, vmax)
    plt.xlabel("Sample Index"); plt.ylabel("Filter coefficient"); plt.grid(alpha=0.3, ls="--")
    out = os.path.join(outdir, f"{name}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def _make_bar_image(f: np.ndarray, Lref: int,
                    height: int = 140, width_per_tap: int = 16,
                    vabs: float = None, zero_bar_px: int = 2) -> np.ndarray:
    f = pad_center_to_length(f, Lref)
    L = f.size
    H = int(height)
    W = int(L * width_per_tap)
    vmax = (np.max(np.abs(f)) if vabs is None else float(vabs)) + 1e-12

    img = np.zeros((H, W), dtype=float)
    mid = H // 2
    if zero_bar_px > 0:
        img[mid-zero_bar_px//2:mid+zero_bar_px//2+1, :] = 0.0

    for i, v in enumerate(f):
        w0, w1 = i * width_per_tap, (i + 1) * width_per_tap
        h = int(np.clip(abs(v) / vmax, 0, 1) * (mid - 3))
        if v >= 0:
            img[mid - h:mid, w0:w1] = (+abs(v) / vmax)
        else:
            img[mid:mid + h, w0:w1] = (-abs(v) / vmax)

    img = np.pad(img, ((6, 6), (6, 6)), mode="constant", constant_values=0.0)
    return img


def save_tree_panel_png(outpath: str,
                        h0a: np.ndarray, h1a: np.ndarray,
                        h0b: np.ndarray, h1b: np.ndarray,
                        cmap: str = "coolwarm"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # 공통 스케일
    Lref = max(len(h0a), len(h1a), len(h0b), len(h1b))
    vabs = max(np.max(np.abs(h0a)), np.max(np.abs(h1a)),
               np.max(np.abs(h0b)), np.max(np.abs(h1b)))

    img_h0a = _make_bar_image(h0a, Lref, vabs=vabs)
    img_h1a = _make_bar_image(h1a, Lref, vabs=vabs)
    img_h0b = _make_bar_image(h0b, Lref, vabs=vabs)
    img_h1b = _make_bar_image(h1b, Lref, vabs=vabs)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), constrained_layout=True)

    def _draw(ax, title, img_lp, img_hp):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(3.0); sp.set_color("#0E6071")
        ax.set_title(title, fontsize=18, pad=8)
        ax_lp = ax.inset_axes([0.06, 0.56, 0.88, 0.36])
        ax_hp = ax.inset_axes([0.06, 0.10, 0.88, 0.36])
        for a, im, ttl in [(ax_lp, img_lp, "LP (h0)"), (ax_hp, img_hp, "HP (h1)")]:
            a.imshow(im, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
            a.set_xticks([]); a.set_yticks([]); a.set_frame_on(False)
            a.text(0.5, 1.02, ttl, ha="center", va="bottom", transform=a.transAxes, fontsize=11)

    _draw(axes[0], "Tree a", img_h0a, img_h1a)
    _draw(axes[1], "Tree b", img_h0b, img_h1b)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved → {outpath}")


# ---------------------------- 메인 ---------------------------- #
def main(biort_name: str = "near_sym_b",
         out_root: str = "loss_experiment_2/precipitation_loss_experiment-main-2/kernel"):
    # 필터 로드(정확한 순서/이름으로)
    h0a, h1a, h0b, h1b = load_biort_filters(biort_name)

    # 공통 가로 길이(Lref)
    Lref = max(len(h0a), len(h1a), len(h0b), len(h1b))

    outdir = os.path.join(out_root, biort_name)
    os.makedirs(outdir, exist_ok=True)

    # 개별 스템(동일 x축)
    save_stem_png("TreeA_LP_(h0a)", h0a, outdir, Lref=Lref)
    save_stem_png("TreeA_HP_(h1a)", h1a, outdir, Lref=Lref)
    save_stem_png("TreeB_LP_(h0b)", h0b, outdir, Lref=Lref)
    save_stem_png("TreeB_HP_(h1b)", h1b, outdir, Lref=Lref)

    # 패널
    panel_path = os.path.join(outdir, "tree_panel.png")
    save_tree_panel_png(panel_path, h0a, h1a, h0b, h1b)


if __name__ == "__main__":
    main(biort_name="near_sym_b")
    # 필요하면:
    # main(biort_name="farras")


