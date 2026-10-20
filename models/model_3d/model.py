import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    일반 Conv2d + BatchNorm + Activation 블록
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(0.1)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

import torch
import torch.nn as nn

class GeocyclicPadding(nn.Module):
    """
    지구 격자용 인덱싱 기반 패딩:
      - 위도(H): 양 끝에서 반사(행 뒤집기) + '반대 경도(180°)' 적용
      - 경도(W): 좌/우 cyclic padding
    입력 텐서: (B, C, T, H, W) 또는 (B, C, H, W) 모두 지원
    """
    def __init__(self, lat_pad: int = 1, lon_pad: int = 1, use_opposite_longitude: bool = True):
        super().__init__()
        self.lat_pad = int(lat_pad)
        self.lon_pad = int(lon_pad)
        self.use_opposite_longitude = bool(use_opposite_longitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 지원 차원 확인
        if x.dim() not in (4, 5):
            raise ValueError(f"GeocyclicPadding expects 4D/5D, got shape={tuple(x.shape)}")
        # 마지막 두 차원은 (H, W) 라고 가정: ... , H, W
        H = x.size(-2)
        W = x.size(-1)

        lat_pad = self.lat_pad
        lon_pad = self.lon_pad

        # ─── 1) 위도(H) 방향 패딩: 양 끝에서 '행 뒤집기' + (선택) 반대 경도 적용 ───
        if lat_pad > 0:
            # 윗쪽 lat_pad 줄 / 아랫쪽 lat_pad 줄 (H 축은 -2)
            # ellipsis(...) 로 4D/5D 모두 동일 코드 사용
            top_band = x[..., :lat_pad, :]      # (..., lat_pad, W)
            bot_band = x[..., -lat_pad:, :]     # (..., lat_pad, W)

            # 위도 뒤집기(행 반전): H 축이 -2이므로 dims=[-2]
            top_pad = torch.flip(top_band, dims=[-2])   # (..., lat_pad, W)
            bot_pad = torch.flip(bot_band, dims=[-2])   # (..., lat_pad, W)

            if self.use_opposite_longitude:
                # 반대 경도 인덱스: 항상 디바이스/long + 모듈러로 안전
                opp = torch.arange(W, device=x.device, dtype=torch.long)
                opp = (opp + (W // 2)) % W      # 0..W-1 보장
                # W축은 -1
                top_pad = top_pad.index_select(dim=-1, index=opp)
                bot_pad = bot_pad.index_select(dim=-1, index=opp)

            # 위/아래 연결 (H 축은 -2)
            x = torch.cat([top_pad, x, bot_pad], dim=-2)

        # ─── 2) 경도(W) 방향 cyclic padding ───
        if lon_pad > 0:
            # 좌/우에서 잘라 붙이기 (W 축은 -1)
            left_chunk  = x[..., -lon_pad:]  # (..., lon_pad)
            right_chunk = x[..., :lon_pad]   # (..., lon_pad)
            x = torch.cat([left_chunk, x, right_chunk], dim=-1)

        return x
# utils/layers.py (새 파일로 두거나, 기존 모듈에 추가)
import torch
import torch.nn as nn

class SpatialMaxPool(nn.Module):
    """
    4D 입력(B,C,H,W) → MaxPool2d
    5D 입력(B,C,T,H,W) → MaxPool3d(kernel_size=(1,k,k))  # 시간축은 유지, 공간만 풀링
    """
    def __init__(self, kernel_size=2, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        if stride is None:
            stride = kernel_size

        # 2D 경로
        self.pool2d = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

        # 5D 경로: 시간축 무변(1), 공간만 풀도록 설정
        if isinstance(kernel_size, int):
            k3 = (1, kernel_size, kernel_size)
        else:
            k3 = (1, kernel_size[0], kernel_size[1])

        if isinstance(stride, int):
            s3 = (1, stride, stride)
        else:
            s3 = (1, stride[0], stride[1])

        if isinstance(padding, int):
            p3 = (0, padding, padding)
        else:
            p3 = (0, padding[0], padding[1])

        self.pool3d = nn.MaxPool3d(k3, stride=s3, padding=p3, ceil_mode=ceil_mode)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            return self.pool2d(x)         # (B,C,H,W)
        elif x.dim() == 5:
            return self.pool3d(x)         # (B, C, T, H, W) -> (B,C,T,H/2,W/2)
        else:
            raise ValueError(f"SpatialMaxPool expects 4D/5D input, got {tuple(x.shape)}")


class GeocyclicConvencoder(nn.Module):
    def __init__(self, in_channels, out_channels, lat_pad=1, lon_pad=1, activation=nn.LeakyReLU(0.1)):
        super(GeocyclicConvencoder, self).__init__()
        # 지구 격자 패딩(인덱싱 방식 유지: 위도 반사 + 반대 경도, 경도 cyclic)
        self.geo_pad = GeocyclicPadding(lat_pad=lat_pad, lon_pad=lon_pad, use_opposite_longitude=True)
        # Conv2d 인코더 (패딩은 geo_pad에서 처리하므로 Conv2d padding=0 유지)
        self.encoder = ConvBlock(in_channels, out_channels, padding=0, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 또는 (B, C, T, H, W)
        Returns:
            (B, out_C, H', W')  또는  (B, out_C, T, H', W')
        계산 순서/의미:
          1) 지구 격자 패딩(위도 반사+반대 경도, 경도 cyclic) ← 기존 논리 동일
          2) Conv2d 인코더 적용 (5D면 시간축 T에 대해 독립 적용: Time-Distributed)
        """
        if x.dim() == 4:
            # (B, C, H, W)
            x = self.geo_pad(x)          # 패딩 (기존과 동일한 계산 의미)
            x = self.encoder(x)          # Conv2d
            return x

        elif x.dim() == 5:
            # (B, C, T, H, W)
            # 1) 패딩은 5D 그대로 적용 (위도 반사+반대 경도, 경도 cyclic) → 기존 의미 동일
            x = self.geo_pad(x)          # (B, C, T, Hp, Wp)

            # 2) Conv2d는 4D만 처리하므로 T 차원을 펼쳐서 Time-Distributed로 적용
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)  # (B*T, C, H, W)
            x = self.encoder(x)          # Conv2d
            C2, H2, W2 = x.size(1), x.size(2), x.size(3)
            x = x.view(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4).contiguous()  # (B, C2, T, H2, W2)
            return x

        else:
            raise ValueError(f"GeocyclicConvencoder expects 4D/5D input, got shape={tuple(x.shape)}")


class Geoupsample(nn.Module):
    def __init__(self, scale_factor=2, lat_pad=1, lon_pad=1):
        super(Geoupsample, self).__init__()
        self.geocyclic_padding = GeocyclicPadding(lat_pad=lat_pad, lon_pad=lon_pad)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.lat_pad = lat_pad
        self.lon_pad = lon_pad

    def forward(self, x):
        # Geocyclic padding
        x = self.geocyclic_padding(x)
        # Upsample
        x = self.upsample(x)
        # Apply slicing
        return x[..., 2*self.lat_pad:-2*self.lat_pad, 2*self.lon_pad:-2*self.lon_pad]  # Remove the extra padding

class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, mask):
        for module in self.modules_list:
            try:
                x, mask = module(x, mask)
            except TypeError:
                x = module(x)  # mask 없는 모듈 (Dropout 등)
        return x, mask

class Geo_UNet(nn.Module):
    def __init__(self, x1_channels=1, x2_channels=1, channels=24, z_min=-0.41654):
        super(Geo_UNet, self).__init__()
        
        # First Encoder with Conv2d
        self.encoder_x11 = nn.Sequential(GeocyclicConvencoder(x1_channels, channels),  
                                         GeocyclicConvencoder(channels, channels * 2),
                                         nn.Dropout2d(p=0.1))
        
        self.encoder_x21 = nn.Sequential(GeocyclicConvencoder(x2_channels, channels),  
                                         GeocyclicConvencoder(channels, channels * 2),
                                         nn.Dropout2d(p=0.1))

        # Seconde Encoder
        self.encoder_x12 = nn.Sequential(GeocyclicConvencoder(channels * 2, channels * 2),  
                                         GeocyclicConvencoder(channels * 2, channels * 4),
                                         nn.Dropout2d(p=0.1))
        
        self.encoder_x22 = nn.Sequential(GeocyclicConvencoder(channels * 2, channels * 2),  
                                         GeocyclicConvencoder(channels * 2, channels * 4),
                                         nn.Dropout2d(p=0.1))

        # Bottom layers
        self.bottom = nn.Sequential(GeocyclicConvencoder(channels * 8, channels * 4),  
                                    GeocyclicConvencoder(channels * 4, channels * 4),
                                    nn.Dropout2d(p=0.1))
        
        # Decoder layers
        self.decoder1 = nn.Sequential(GeocyclicConvencoder(channels * 12, channels * 2),
                                      nn.Dropout2d(p=0.1),  
                                      GeocyclicConvencoder(channels * 2, channels * 2))
        
        self.decoder2 = nn.Sequential(GeocyclicConvencoder(channels * 6, channels * 1),
                                      nn.Dropout2d(p=0.1),  
                                      GeocyclicConvencoder(channels * 1, channels * 1))
 
        # Final layers 3D
        self.time_mix = nn.Conv3d(channels // 3, channels // 6, kernel_size=(3,1,1), padding=(1,0,0))

        self.final_y1 = nn.Conv3d(
            in_channels=channels // 6,
            out_channels=1,
            kernel_size=1)

        # Pooling and Upsampling
        self.pool = SpatialMaxPool(kernel_size=2)
        self.up = Geoupsample(scale_factor=2, lat_pad=1, lon_pad=1)
        self.z_min = z_min

    def forward(self, x1, x2):

        # x1 인코딩 (PartialConv2D 사용)
        # Encoder pass 1 (180x360 -> 90x180)
        encoded_x11 = self.encoder_x11(x1)
        encoded_x21 = self.encoder_x21(x2)

        encoded_x11_pool = self.pool(encoded_x11)
        encoded_x21_pool = self.pool(encoded_x21)

        # Encoder pass 2 (90x180 -> 45x90)
        encoded_x12 = self.encoder_x12(encoded_x11_pool)
        encoded_x22 = self.encoder_x22(encoded_x21_pool)

        encoded_x12_pool = self.pool(encoded_x12)
        encoded_x22_pool = self.pool(encoded_x22)

        # Bottom layers: Combine layers (45x90)
        combined_x = self.bottom(torch.cat([encoded_x12_pool, encoded_x22_pool], dim=1)) # 습도 + 강수

        # Decoder pass 1 (45x90 -> 90x180)
        combined_up = self.up(combined_x) 
        decoded_x1 = self.decoder1(torch.cat([combined_up, encoded_x12, encoded_x22], dim=1))

        # Decoder pass 2 (90x180 -> 180x360)
        decoded_x1_up = self.up(decoded_x1)
        decoded_x2 = self.decoder2(torch.cat([decoded_x1_up, encoded_x11, encoded_x21], dim=1))

        # Convert to 3D time frame
        B, C, H, W = decoded_x2.shape
        C_in = C // 3
        decoded_x2 = decoded_x2.view(B, C_in, 3, H, W)

        # Final outputs
        decoded_x2 = F.leaky_relu(self.time_mix(decoded_x2), 0.1)
        y1 = self.final_y1(decoded_x2)
        y1 = torch.clamp(y1, min=self.z_min)

        return y1
    
### code check ###

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Geo_UNet().to(device)

# # 더미 입력도 같은 디바이스로 이동
# x1 = torch.randn(1, 1, 180, 360).to(device)
# x2 = torch.randn(1, 1, 180, 360).to(device)

# # forward
# y1 = model(x1, x2)
# print("✅ Output y1 shape:", y1.shape)

# # 파라미터 수 확인
# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"📦 Total parameters: {total:,}")
#     print(f"🛠️ Trainable parameters: {trainable:,}")

# count_parameters(model)