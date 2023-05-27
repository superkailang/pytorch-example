import cv2
import numpy as np
import torch
import os
from einops import rearrange
from annotator.base_annotator import BaseProcessor
from .midas.dpt_depth import DPTDepthModel
from .midas.midas_net import MidasNet
from .midas.midas_net_custom import MidasNet_small
from .midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"


def load_midas_transform(model_type):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load transform only
    if model_type == "dpt_large":  # DPT-Large
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    elif model_type == "midas_v21_small":
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    else:
        assert False, f"model_type '{model_type}' not implemented, use: --model_type large"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return transform


class MidasProcessor(BaseProcessor):
    MODEL_TYPES_TORCH_HUB = [
        "DPT_Large",
        "DPT_Hybrid",
        "MiDaS_small"
    ]
    MODEL_TYPES_ISL = [
        "dpt_large",
        "dpt_hybrid",
        "midas_v21",
        "midas_v21_small",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_dir = os.path.join(self.models_path, "midas")
        self.model = None

    def load_model(self, model_type):
        ISL_PATHS = {
            "dpt_large": os.path.join(self.model_dir, "dpt_large-midas-2f21e586.pt"),
            "dpt_hybrid": os.path.join(self.model_dir, "dpt_hybrid-midas-501f0c75.pt"),
            "midas_v21": "",
            "midas_v21_small": "",
        }
        # https://github.com/isl-org/MiDaS/blob/master/run.py
        # load network
        model_path = ISL_PATHS[model_type]
        # old_model_path = OLD_ISL_PATHS[model_type]
        if model_type == "dpt_large":  # DPT-Large
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            if not os.path.exists(model_path):
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(remote_model_path, model_dir=self.model_dir)

            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "midas_v21":
            model = MidasNet(model_path, non_negative=True)
            net_w, net_h = 384, 384
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif model_type == "midas_v21_small":
            model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                                   non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        else:
            print(f"model_type '{model_type}' not implemented, use: --model_type large")
            assert False

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()
        self.model = model

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1, *args, **kwargs):
        if self.model is None:
            self.load_model(model_type="dpt_hybrid")
        if self.device != 'mps':
            self.model = self.model.to(self.device)
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            if self.device != 'mps':
                image_depth = image_depth.to(self.device)
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]

            return depth_image, normal_image

    def unload_midas_model(self):
        if self.model is not None:
            self.model = self.model.cpu()