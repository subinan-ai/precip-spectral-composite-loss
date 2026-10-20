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

class GeocyclicPadding(nn.Module):
    def __init__(self, lat_pad=1, lon_pad=1):
        super(GeocyclicPadding, self).__init__()
        self.lat_pad = lat_pad
        self.lon_pad = lon_pad

    def forward(self, x):
        original_lon = x.size(-1)

        # 반대 경도 계산
        opposite_indices = (torch.arange(original_lon) + original_lon // 2) % original_lon

        # Top padding: 위도 순서를 뒤집고 반대 경도 적용
        top_pad = torch.flip(x[:, :, :self.lat_pad, :], dims=[-2])  # 위도 순서를 뒤집음
        top_pad = top_pad[:, :, :, opposite_indices]  # 반대 경도 적용

        # Bottom padding: 위도 순서를 뒤집고 반대 경도 적용
        bottom_pad = torch.flip(x[:, :, -self.lat_pad:, :], dims=[-2])  # 위도 순서를 뒤집음
        bottom_pad = bottom_pad[:, :, :, opposite_indices]  # 반대 경도 적용

        # 위도 방향으로 합치기
        lat_padded_x = torch.cat([top_pad, x, bottom_pad], dim=-2)

        # 경도 방향 cyclic padding
        padded_x = torch.cat([
            lat_padded_x[:, :, :, -self.lon_pad:],  # 오른쪽 경도를 왼쪽으로
            lat_padded_x,                          # 원본 데이터
            lat_padded_x[:, :, :, :self.lon_pad]   # 왼쪽 경도를 오른쪽으로
        ], dim=-1)

        return padded_x

class GeocyclicConvencoder(nn.Module):
    def __init__(self, in_channels, out_channels, lat_pad=1, lon_pad=1, activation=nn.LeakyReLU(0.1)):
        super(GeocyclicConvencoder, self).__init__()
        # Geocyclic padding + Conv2D encoder
        self.geo_pad = GeocyclicPadding(lat_pad=lat_pad, lon_pad=lon_pad)
        self.encoder = ConvBlock(in_channels, out_channels, padding=0, activation=activation)
    
    def forward(self, x):
        """
        Args: x1: Input tensor of shape (batch_size, channels, lat, lon)
        Returns: Encoded tensor
        """
        # Apply geocyclic padding + Encode with ConvBlock
        x_padded = self.geo_pad(x)
        x_encoded = self.encoder(x_padded)
        
        return x_encoded

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
    def __init__(self, x1_channels=1, x2_channels=1, u_channels=1, v_channels=1, channels=24):
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
        #u, v 인코더 추가
        self.encoder_u1 = nn.Sequential(GeocyclicConvencoder(u_channels, channels),
                                         GeocyclicConvencoder(channels, channels * 2),
                                         nn.Dropout2d(p=0.1))

        self.encoder_v1 = nn.Sequential(GeocyclicConvencoder(v_channels, channels),
                                         GeocyclicConvencoder(channels, channels * 2),
                                         nn.Dropout2d(p=0.1))

        self.encoder_u2 = nn.Sequential(GeocyclicConvencoder(channels * 2, channels * 2),
                                         GeocyclicConvencoder(channels * 2, channels * 4),
                                         nn.Dropout2d(p=0.1))

        self.encoder_v2 = nn.Sequential(GeocyclicConvencoder(channels * 2, channels * 2),
                                         GeocyclicConvencoder(channels * 2, channels * 4),
                                         nn.Dropout2d(p=0.1))

        # Bottom layers
        self.bottom = nn.Sequential(GeocyclicConvencoder(channels * 16, channels * 8), #4개 입력 * channels * 4 = channels * 16
                                    GeocyclicConvencoder(channels * 8, channels * 8),
                                    nn.Dropout2d(p=0.1))

        # Decoder layers
        # Decoder 1: combined_up(channels*8) + encoded_x12(channels*4) + encoded_x22(channels*4) + encoded_u2(channels*4) + encoded_v2(channels*4) = channels*24
        self.decoder1 = nn.Sequential(GeocyclicConvencoder(channels * 24, channels * 4),
                                      nn.Dropout2d(p=0.1),
                                      GeocyclicConvencoder(channels * 4, channels * 4))
        # Decoder 2: decoded_x1_up(channels*4) + encoded_x11(channels*2) + encoded_x21(channels*2) + encoded_u1(channels*2) + encoded_v1(channels*2) = channels*12
        self.decoder2 = nn.Sequential(GeocyclicConvencoder(channels * 12, channels * 2),
                                      nn.Dropout2d(p=0.1),
                                      GeocyclicConvencoder(channels * 2, channels * 2))

        # Final layers 3D
        self.time_mix = nn.Conv3d((channels * 2) // 3, channels // 3, kernel_size=(3,1,1), padding=(1,0,0))

        self.final_y1 = nn.Conv3d(
            in_channels=channels // 3,
            out_channels=1,
            kernel_size=1)

        # Pooling and Upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = Geoupsample(scale_factor=2, lat_pad=1, lon_pad=1)

    def forward(self, x1, x2, u, v):

        # x1 인코딩 (PartialConv2D 사용)
        # Encoder pass 1 (180x360 -> 90x180)
        encoded_x11 = self.encoder_x11(x1)
        encoded_x21 = self.encoder_x21(x2)
        encoded_u1=self.encoder_u1(u)
        encoded_v1=self.encoder_v1(v)

        encoded_x11_pool = self.pool(encoded_x11)
        encoded_x21_pool = self.pool(encoded_x21)
        encoded_u1_pool=self.pool(encoded_u1)
        encoded_v1_pool=self.pool(encoded_v1)

        # Encoder pass 2 (90x180 -> 45x90)
        encoded_x12 = self.encoder_x12(encoded_x11_pool)
        encoded_x22 = self.encoder_x22(encoded_x21_pool)
        encoded_u2=self.encoder_u2(encoded_u1_pool)
        encoded_v2=self.encoder_v2(encoded_v1_pool)

        encoded_x12_pool = self.pool(encoded_x12)
        encoded_x22_pool = self.pool(encoded_x22)
        encoded_u2_pool=self.pool(encoded_u2)
        encoded_v2_pool=self.pool(encoded_v2)

        # Bottom layers: Combine layers (45x90)
        combined_x = self.bottom(torch.cat([encoded_x12_pool, encoded_x22_pool, encoded_u2_pool, encoded_v2_pool], dim=1)) # 습도 + 강수 + u + v

        # Decoder pass 1 (45x90 -> 90x180)
        combined_up = self.up(combined_x)
        decoded_x1 = self.decoder1(torch.cat([combined_up, encoded_x12, encoded_x22, encoded_u2, encoded_v2], dim=1))

        # Decoder pass 2 (90x180 -> 180x360)
        decoded_x1_up = self.up(decoded_x1)
        decoded_x2 = self.decoder2(torch.cat([decoded_x1_up, encoded_x11, encoded_x21, encoded_u1, encoded_v1], dim=1))

        # Convert to 3D time frame
        B, C, H, W = decoded_x2.shape
        C_in = C // 3
        decoded_x2=decoded_x2.view(B, C_in, 3, H, W)

        # Final outputs
        decoded_x2 = F.leaky_relu(self.time_mix(decoded_x2), 0.1)
        y1 = self.final_y1(decoded_x2)

        return y1
    
### code check ###

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Geo_UNet().to(device)

# # 더미 입력도 같은 디바이스로 이동
# x1 = torch.randn(1, 1, 180, 360).to(device)
# x2 = torch.randn(1, 1, 180, 360).to(device)
# x3 = torch.randn(1, 1, 180, 360).to(device)
# x4 = torch.randn(1, 1, 180, 360).to(device)

# # forward
# y1 = model(x1, x2, x3, x4)
# print("✅ Output y1 shape:", y1.shape)

# # 파라미터 수 확인
# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"📦 Total parameters: {total:,}")
#     print(f"🛠️ Trainable parameters: {trainable:,}")

# count_parameters(model)