from .mediapipe_face_common import generate_annotation


class MediaPipeFace:
    def __call__(self, image, max_faces: int = 1, min_confidence: float = 0.5, **kwargs):
        return generate_annotation(image, max_faces, min_confidence)
