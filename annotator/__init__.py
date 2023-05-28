from .canny import CannyDetector
from .openpose import OpenposeDetector
from .binary import BinaryDetector
from .hed import HedDetector
# from .keypose import KeyPoseProcess
from .midas import MidasProcessor
from .mlsd import MLSDProcessor
from .uniformer import UniformerDetector
from .lineart import LineArtDetector
from .lineart_anime import LineArtAnimeDetector
from .manga_line import MangaLineExtration
from .leres import LeresPix2Pix
from .mediapipe_face import MediaPipeFace
from .normalbae import NormalBaeDetector
from .pidinet import PidInet
from .shuffle import Image2MaskShuffleDetector
from .zoe import ZoeDetector
from .oneformer import OneformerDetector

__all__ = [
    UniformerDetector,
    HedDetector,
    MLSDProcessor,
    BinaryDetector,
    CannyDetector,
    OpenposeDetector,
    MidasProcessor,
    LineArtDetector,
    LineArtAnimeDetector,
    MangaLineExtration,
    LeresPix2Pix,
    MediaPipeFace,
    NormalBaeDetector,
    PidInet,
    Image2MaskShuffleDetector,
    ZoeDetector,
    OneformerDetector
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
