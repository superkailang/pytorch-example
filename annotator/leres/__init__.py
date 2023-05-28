import cv2
import numpy as np
import torch
import os

# AdelaiDepth/LeReS imports
from .leres.depthmap import estimateleres, estimateboost
from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import strip_prefix_if_present
from annotator.base_annotator import BaseProcessor

# pix2pix/merge net imports
from .pix2pix.options.test_options import TestOptions
from .pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

# old_modeldir = os.path.dirname(os.path.realpath(__file__))

remote_model_path_leres = "https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth"
remote_model_path_pix2pix = "https://huggingface.co/lllyasviel/Annotators/resolve/main/latest_net_G.pth"


class LeresPix2Pix(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.pix2pixmodel = None
        self.model_dir = os.path.join(self.models_path, "leres")

    def unload_model(self):
        if self.model is not None:
            self.model = self.model.cpu()
        if self.pix2pixmodel is not None:
            self.pix2pixmodel = self.pix2pixmodel.unload_network('G')

    def load_model(self):
        model_path = os.path.join(self.model_dir, "res101.pth")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path_leres, model_dir=self.model_dir)

        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        self.model = RelDepthModel(backbone='resnext101')
        self.model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
        del checkpoint

    def load_pix2pix2_model(self):
        pix2pixmodel_path = os.path.join(self.model_dir, "latest_net_G.pth")
        if not os.path.exists(pix2pixmodel_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path_pix2pix, model_dir=self.model_dir)

        opt = TestOptions().parse()
        if not torch.cuda.is_available():
            opt.gpu_ids = []  # cpu mode
        self.pix2pixmodel = Pix2Pix4DepthModel(opt)
        self.pix2pixmodel.save_dir = self.model_dir
        self.pix2pixmodel.load_networks('latest')
        self.pix2pixmodel.eval()

    def __call__(self, input_image, thr_a, thr_b, boost=False, **kwargs):
        if self.model is None:
            self.load_model()
        if boost and self.pix2pixmodel is None:
            self.load_pix2pix2_model()

        if self.device != 'mps':
            self.model = self.model.to(self.device)

        assert input_image.ndim == 3
        height, width, dim = input_image.shape

        with torch.no_grad():

            if boost:
                depth = estimateboost(input_image, self.model, 0, self.pix2pixmodel, max(width, height))
            else:
                depth = estimateleres(input_image, self.model, width, height, self.device)

            numbytes = 2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2 ** (8 * numbytes)) - 1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape)

            # single channel, 16 bit image
            depth_image = out.astype("uint16")

            # convert to uint8
            depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0 / 65535.0))

            # remove near
            if thr_a != 0:
                thr_a = ((thr_a / 100) * 255)
                depth_image = cv2.threshold(depth_image, thr_a, 255, cv2.THRESH_TOZERO)[1]

            # invert image
            depth_image = cv2.bitwise_not(depth_image)

            # remove bg
            if thr_b != 0:
                thr_b = ((thr_b / 100) * 255)
                depth_image = cv2.threshold(depth_image, thr_b, 255, cv2.THRESH_TOZERO)[1]

            return depth_image
