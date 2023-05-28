import os
import torch
import numpy as np
from einops import rearrange
from annotator.pidinet.model import pidinet
from annotator.util import safe_step
from annotator.base_annotator import BaseProcessor

remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"


class PidInet(BaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_dir = os.path.join(self.models_path, "pidinet")
        self.netNetwork = None

    def unload_model(self):
        if self.netNetwork is not None:
            self.netNetwork.cpu()

    def load_model(self):
        modelpath = os.path.join(self.model_dir, "table5_pidinet.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        self.netNetwork = pidinet()
        ckp = torch.load(modelpath)['state_dict']
        self.netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()})

    def __call__(self, input_image, is_safe=False, apply_fliter=False, **kwargs):
        if self.netNetwork is None:
            self.load_model()

        self.netNetwork = self.netNetwork.to(self.device)
        self.netNetwork.eval()
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(input_image).float().to(self.device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_fliter:
                edge = edge > 0.5
            if is_safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        return edge[0][0]
