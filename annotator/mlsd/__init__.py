import cv2
import numpy as np
import torch
import os

from einops import rearrange
from annotator.base_annotator import BaseProcessor
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines

remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"
old_modeldir = os.path.dirname(os.path.realpath(__file__))


class MLSDProcessor(BaseProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.model_dir = os.path.join(self.models_path, "mlsd")

    def unload_model(self):
        if self.model is not None:
            self.model = self.model.cpu()

    def load_model(self):
        model_path = os.path.join(self.model_dir, "mlsd_large_512_fp32.pth")
        # old_modelpath = os.path.join(old_modeldir, "mlsd_large_512_fp32.pth")
        # if os.path.exists(old_modelpath):
        #     modelpath = old_modelpath
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        mlsdmodel = MobileV2_MLSD_Large()
        mlsdmodel.load_state_dict(torch.load(model_path), strict=True)

        mlsdmodel = mlsdmodel.to(self.device).eval()
        self.model = mlsdmodel

    def __call__(self, input_image, thr_v= 0.1, thr_d= 0.1, **kwargs):
        # global modelpath, mlsdmodel
        if self.model is None:
            self.load_model()
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d, self.device)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass
        return img_output[:, :, 0]
