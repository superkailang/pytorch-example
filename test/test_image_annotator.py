import os

import cv2
from matplotlib import pyplot as plt

from Image.processor import resize_image_with_pad
from annotator import DenthPoseProcessor, MLSDProcessor, MidasProcessor,UniformerDetector,LineArtAnimeDetector,MangaLineExtration, LineArtDetector, HedDetector

# # default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
model_cache_home = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "model"),
    )
)

image_path = "demo.jpg"
img = cv2.imread(image_path)

img, remove_pad = resize_image_with_pad(img, 512)

img2 = DenthPoseProcessor()(img,"dp_contour,bbox")

# img2 = BinaryDetector()(img,bin_threshold=110)
plt.imshow(img2[:, :])
plt.axis('off')
plt.show()


detector = HedDetector(device="cpu", models_path=model_cache_home)

canvas = detector(
    img,
    include_body=True,
    include_hand=True,
    include_face=True,
    json_pose_callback=None
)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
