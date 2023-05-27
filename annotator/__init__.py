from .canny import CannyDetector
from .openpose import OpenposeDetector
from .binary import BinaryDetector
from .hed import HedDetector
# from .keypose import KeyPoseProcess
from .midas import MidasProcessor
from .mlsd import MLSDProcessor
from .uniformer import UniformerDetector

__all__ = [
    UniformerDetector, HedDetector, MLSDProcessor, BinaryDetector, CannyDetector, OpenposeDetector, MidasProcessor
]
#
#
# # default cache
# default_home = os.path.join(os.path.expanduser("~"), ".cache")
# model_cache_home = os.path.expanduser(
#     os.getenv(
#         "HF_HOME",
#         os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "model"),
#     )
# )
