# Uniformer
# From https://github.com/Sense-X/UniFormer
# # Apache-2.0 license

import os

from annotator.base_annotator import BaseProcessor
from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette


checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"


class UniformerDetector(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.model_dir = os.path.join(self.models_path, "uniformer")
        self.model = None

    def load_model(self):
        model_path = os.path.join(self.model_dir, "upernet_global_small.pth")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(checkpoint_file, model_dir=self.model_dir)
        file_package = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(file_package, "exp", "upernet_global_small", "config.py")
        self.model = init_segmentor(config_file, model_path, self.device).to(self.device)

    def __call__(self, img):
        if self.model is None:
            self.load_model()
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img