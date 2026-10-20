import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# ---- 라벨 텍스트 설정 ----
label = "(a) Ground Truth"

# ---- Figure 생성 ----
plt.figure(figsize=(2.5, 0.5))
plt.text(0.0, 0.5, label,
         fontsize=16,           # 글자 크기
         fontweight='bold',     # 굵게 (옵션)
         ha='left', va='center')

# ---- 테두리 / 축 제거 ----
plt.axis('off')
plt.tight_layout()

# ---- 저장 ----
plt.savefig("/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/label_png/label_groundtruth.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()
import matplotlib.pyplot as plt

# ---- 텍스트 설정 ----
label = "(a) Ground Truth"

# ---- Figure 생성 ----
plt.figure(figsize=(4, 0.8))
plt.text(0.0, 0.5, label,
         fontsize=15,               # 비슷한 크기
         fontweight='normal',       # 일반 두께 (볼드 아님)
         ha='left', va='center',    # 좌측 정렬, 수직 가운데
         fontname='DejaVu Sans')    # matplotlib 기본 폰트 (논문용과 유사)

# ---- 축, 테두리 제거 ----
plt.axis('off')
plt.tight_layout()

# ---- 저장 ----
plt.savefig("/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/label_png/label_groundtruth_style.png", dpi=300,
            bbox_inches='tight', pad_inches=0.05, transparent=True)
plt.show()

from PIL import Image

# ---- 원본 이미지 경로 ----
input_path = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/pic1.png"  # 업로드한 파일 이름
output_path = "/home/subin_an/loss_experiment_2/precipitation_loss_experiment-main-2/label_png/image_300dpi.png"  # 저장될 파일 이름

# ---- 이미지 열기 ----
img = Image.open(input_path)

# ---- 3배 확대 (고품질 보간법) ----
scale_factor = 3
new_size = (img.width * scale_factor, img.height * scale_factor)
img_upscaled = img.resize(new_size, Image.LANCZOS)

# ---- 선명도 향상 (Unsharp Mask) ----
img_sharp = img_upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))

# ---- 300 DPI로 저장 ----
img_sharp.save(output_path, dpi=(300, 300))
print("✅ 고해상도 + 선명도 향상 완료:", output_path)
