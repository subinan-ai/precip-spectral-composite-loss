# models/get_model.py
def build_model(config):
    """
    config 예:
    model:
      version: "3d"         # "2d" | "3d"
      name: "geo_unet"      # 모델 이름(확장 가능)
      x1_channels: 2
      x2_channels: 2
      hidden_channels: 64
    """
    mcfg = config["model"]
    version = str(mcfg["version"]).lower()
    name = str(mcfg.get("name", "geo_unet")).lower()
    is_model_uv = False

    if version == "3d":
        if name == "geo_unet":
            from models.model_3d import Geo_UNet
            ModelCls = Geo_UNet
        else:
            raise ValueError(f"Unknown 3D model name: {name}")
    elif version == "2d":
        if name == "geo_unet":
            from models.model_2d import Geo_UNet
            ModelCls = Geo_UNet
        else:
            raise ValueError(f"Unknown 2D model name: {name}")
    elif version == "3d_uv":
        is_model_uv = True
        if name == "geo_unet":
            from models.model_3d_uv import Geo_UNet
            ModelCls_uv = Geo_UNet
        else:
            raise ValueError(f"Unknown 3D model name: {name}")
    else:
        raise ValueError(f"Unsupported model version: {version}")

    if is_model_uv:
        return ModelCls_uv(
            x1_channels=mcfg["x1_channels"],
            x2_channels=mcfg["x2_channels"],
            u_channels=mcfg["u_channels"],
            v_channels=mcfg["v_channels"],
            channels=mcfg["hidden_channels"]
        )
    else:
        return ModelCls(
            x1_channels=mcfg["x1_channels"],
            x2_channels=mcfg["x2_channels"],
            channels=mcfg["hidden_channels"]
        )