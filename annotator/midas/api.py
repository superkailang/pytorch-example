# # based on https://github.com/isl-org/MiDaS
# # Third-party model: Midas depth estimation model.
#
# import cv2
# import torch
# import torch.nn as nn
#
#
# from torchvision.transforms import Compose
#
#
#
#
# # OLD_ISL_PATHS = {
# #     "dpt_large": os.path.join(old_modeldir, "dpt_large-midas-2f21e586.pt"),
# #     "dpt_hybrid": os.path.join(old_modeldir, "dpt_hybrid-midas-501f0c75.pt"),
# #     "midas_v21": "",
# #     "midas_v21_small": "",
# # }
#
#
# def disabled_train(self, mode=True):
#     """Overwrite model.train with this function to make sure train/eval mode
#     does not change anymore."""
#     return self
#
#
#
#
#
#
#
#
#
# class MiDaSInference(nn.Module):
#
#
#     def __init__(self, model_type):
#         super().__init__()
#         assert (model_type in self.MODEL_TYPES_ISL)
#         model, _ = load_model(model_type)
#         self.model = model
#         self.model.train = disabled_train
#
#     def forward(self, x):
#         with torch.no_grad():
#             prediction = self.model(x)
#         return prediction
