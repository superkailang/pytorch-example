import os.path
from os import path
from typing import List, Dict, Any, ClassVar

import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import get_texture_atlases, DensePoseOutputsTextureVisualizer, \
    DensePoseOutputsVertexVisualizer
from densepose.vis.densepose_results import DensePoseResultsContourVisualizer, \
    DensePoseResultsFineSegmentationVisualizer, DensePoseResultsUVisualizer, DensePoseResultsVVisualizer
from densepose.vis.densepose_results_textures import get_texture_atlas, DensePoseResultsVisualizerWithTexture
from densepose.vis.extractor import create_extractor, CompoundExtractor
from detectron2.config import CfgNode, get_cfg
from detectron2.engine.defaults import DefaultPredictor

from annotator.base_annotator import BaseProcessor

config_model = {
    "densepose_rcnn_R_50_FPN_s1x": {
        "yaml": 'densepose_rcnn_R_50_FPN_s1x.yaml',
        "file": "densepose_rcnn_R_50_FPN_s1x.pkl"
    }
}

default_conf = config_model["densepose_rcnn_R_50_FPN_s1x"]


class DenthPoseProcessor(BaseProcessor):
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        here = path.abspath(path.dirname(__file__))
        self.cfg, self.model_conf = os.path.join(here, default_conf["yaml"]), os.path.join(here,default_conf["file"])
        self.predictor = None

    @classmethod
    def setup_config(
            cls: type, config_fpath: str, model_fpath: str, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def create_context(cls: type, visualizations, cfg: CfgNode, arg_texture_atlas=None, arg_texture_atlases_map=None) -> Dict[
        str, Any]:
        """
            创建可视化
        """
        vis_specs = visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(arg_texture_atlas)
            texture_atlases_dict = get_texture_atlases(arg_texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "entry_idx": 0,
        }
        return context

    def execute_on_outputs(self, image, context: Dict[str, Any], outputs):
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        # image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        # zero_image = np.zeros(image.shape, dtype=np.uint8)
        image_vis = visualizer.visualize(image, data)
        return image_vis

    def __call__(self, img, visualizations, texture_atlas=None, texture_atlases_map=None):
        opts = []
        cfg = self.setup_config(config_fpath=self.cfg, model_fpath=self.model_conf, opts=opts)
        if self.predictor is None:
            self.predictor = DefaultPredictor(cfg)
        context = self.create_context(visualizations, cfg, texture_atlas, texture_atlases_map)
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            return self.execute_on_outputs(img, context, outputs)


# if __name__ == '__main__':
#     image_path = "demo.jpeg"
#     img = cv2.imread(image_path)
#     process = DenthPoseProcessor()
#     process(img, "dp_contour,bbox")
