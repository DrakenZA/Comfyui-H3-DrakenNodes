"""FaceFusion-inspired XSeg face masks for ComfyUI.

This is an independent integration of the upstream models, not a copy of
FaceFusion's Python implementation. Model files are fetched from their owners
only when this node is run; they are not distributed with this package.
"""

import logging
import hashlib
import os
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


LOG = logging.getLogger(__name__)
_MODEL_LOCK = threading.RLock()
_DETECTOR_LOCK = threading.Lock()
_XSEG_RELEASES = {"xseg_1": "models-3.1.0", "xseg_2": "models-3.1.0", "xseg_3": "models-3.2.0"}
_MODEL_SHA256 = {
    "xseg_1.onnx": "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3",
    "xseg_2.onnx": "cd9a0879eaf43841d765472cf1f8c330dbf9dcb03da0eace93e95f3bcc399042",
    "xseg_3.onnx": "48ccd7e8541e159a5a754ec9e62df2f12065f7df8f9af842c1750342c6533559",
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
}
_YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "47534e27c9851bb1128ccc0102f1145e27f23f98/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Five point layout with room for forehead, jaw and a soft crop boundary.
_FACE_TEMPLATE = np.array(
    [[0.38, 0.46], [0.62, 0.46], [0.50, 0.62], [0.39, 0.74], [0.61, 0.74]],
    dtype=np.float32,
)


def _model_dir():
    import folder_paths

    directory = Path(folder_paths.models_dir) / "facefusion"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _valid_model(path, filename, minimum_bytes):
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == _MODEL_SHA256[filename]


def _ensure_model(filename, url, minimum_bytes):
    path = _model_dir() / filename
    with _MODEL_LOCK:
        if _valid_model(path, filename, minimum_bytes):
            return path
        if path.exists():
            raise RuntimeError(f"Model checksum mismatch at {path}; remove it and run again")
        temporary = path.with_name(path.name + ".download")
        try:
            LOG.info("Downloading face mask model %s", filename)
            with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not _valid_model(temporary, filename, minimum_bytes):
                raise RuntimeError(f"Download of {filename} failed checksum validation")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=1)
def _yunet_path():
    return _ensure_model("face_detection_yunet_2023mar.onnx", _YUNET_URL, 100_000)


@lru_cache(maxsize=3)
def _xseg_path(name):
    release = _XSEG_RELEASES[name]
    return _ensure_model(
        name + ".onnx",
        f"https://github.com/facefusion/facefusion-assets/releases/download/{release}/{name}.onnx",
        1_000_000,
    )


@lru_cache(maxsize=3)
def _xseg_session(name):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "Face Occlusion Mask needs onnxruntime or onnxruntime-gpu in ComfyUI's Python environment"
        ) from error

    available = ort.get_available_providers()
    providers = [provider for provider in
                 ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider",
                  "CoreMLExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider")
                 if provider in available]
    if not providers:
        raise RuntimeError("No ONNX Runtime execution provider is available")
    session = ort.InferenceSession(str(_xseg_path(name)), providers=providers)
    return session, session.get_inputs()[0].name


@lru_cache(maxsize=16)
def _yunet_detector(width, height, score_threshold):
    import cv2

    return cv2.FaceDetectorYN.create(str(_yunet_path()), "", (width, height),
                                     score_threshold, 0.3, 5000)


def _detect_faces(bgr, score_threshold, detection_size):
    import cv2

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("Face Occlusion Mask needs OpenCV with FaceDetectorYN support")
    height, width = bgr.shape[:2]
    scale = min(1.0, detection_size / max(height, width))
    detect_width = max(1, round(width * scale))
    detect_height = max(1, round(height * scale))
    reduced = cv2.resize(bgr, (detect_width, detect_height), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    # Reuse the detector for video frames with the same size and threshold.
    # OpenCV's detector is mutable; serialize access to the cached instance.
    with _DETECTOR_LOCK:
        detector = _yunet_detector(detect_width, detect_height, score_threshold)
        _, faces = detector.detect(reduced)
    if faces is None:
        return []
    faces = faces.copy()
    faces[:, :14] /= scale
    return sorted(faces, key=lambda face: face[2] * face[3], reverse=True)


def _crop_box(size, blur):
    import cv2

    box = np.ones((size, size), dtype=np.float32)
    inset = max(1, round(size * 0.075))
    box[:inset, :] = 0
    box[-inset:, :] = 0
    box[:, :inset] = 0
    box[:, -inset:] = 0
    if blur > 0:
        box = cv2.GaussianBlur(box, (0, 0), max(0.01, size * blur * 0.125))
    return box


def _mask_one_face(bgr, face, session, input_name, box_blur):
    import cv2

    height, width = bgr.shape[:2]
    size = 256
    # YuNet returns left eye, right eye, nose, left mouth, right mouth.
    landmarks = face[4:14].reshape(5, 2).astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, _FACE_TEMPLATE * size, method=cv2.LMEDS)
    if matrix is None:
        return np.zeros((height, width), dtype=np.float32)
    crop = cv2.warpAffine(bgr, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    # FaceFusion's XSeg exports take BGR NHWC float32 in [0, 1].
    input_image = np.ascontiguousarray(crop[None].astype(np.float32) / 255.0)
    raw = np.asarray(session.run(None, {input_name: input_image})[0]).squeeze()
    if raw.shape != (size, size):
        raise RuntimeError(f"Unexpected XSeg mask shape: {raw.shape}")
    visible = np.clip(raw.astype(np.float32), 0, 1)
    visible = np.clip((cv2.GaussianBlur(visible, (0, 0), 5) - 0.5) * 2, 0, 1)
    visible *= _crop_box(size, box_blur)
    inverse = cv2.invertAffineTransform(matrix)
    return cv2.warpAffine(visible, inverse, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class FaceOcclusionMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "xseg_model": (list(_XSEG_RELEASES), {"default": "xseg_1",
                    "tooltip": "XSeg model from FaceFusion. Downloaded on first use into models/facefusion."}),
                "faces": (["all", "largest"], {"default": "all"}),
                "detection_score": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05}),
                "detection_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32,
                    "tooltip": "Longest side used for face detection. Increase for small faces; lower for speed."}),
                "box_blur": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Softness of the outer face crop. Object boundaries come from XSeg."}),
            },
            "optional": {
                "base_mask": ("MASK", {"tooltip": "Optional existing face mask to intersect with the XSeg result."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("visible_face_mask",)
    FUNCTION = "make_mask"
    CATEGORY = "DrakenNodes/Face"
    DESCRIPTION = "Full-frame mask of visible face pixels. Foreground objects crossing the face are excluded."

    def make_mask(self, images, xseg_model="xseg_1", faces="all", detection_score=0.7,
                  detection_size=640, box_blur=0.3, base_mask=None):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be a ComfyUI IMAGE batch [B, H, W, C] with at least 3 channels")
        if xseg_model not in _XSEG_RELEASES:
            raise ValueError(f"Unknown XSeg model: {xseg_model}")
        if faces not in ("all", "largest"):
            raise ValueError(f"Unknown face selection mode: {faces}")
        batch, height, width = images.shape[:3]
        if base_mask is not None:
            if base_mask.ndim != 3 or tuple(base_mask.shape[-2:]) != (height, width) or base_mask.shape[0] not in (1, batch):
                raise ValueError("base_mask must match image height and width, with batch size 1 or image batch size")

        import cv2

        session = input_name = None
        masks = []
        for index, image in enumerate(images):
            rgb = (image[..., :3].detach().clamp(0, 1).mul(255).byte().cpu().numpy())
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detections = _detect_faces(bgr, detection_score, detection_size)
            if faces == "largest":
                detections = detections[:1]
            mask = np.zeros((height, width), dtype=np.float32)
            if detections and session is None:
                session, input_name = _xseg_session(xseg_model)
            for face in detections:
                mask = np.maximum(mask, _mask_one_face(bgr, face, session, input_name, box_blur))
            if base_mask is not None:
                source = base_mask[0 if base_mask.shape[0] == 1 else index]
                mask *= source.detach().clamp(0, 1).cpu().numpy()
            masks.append(torch.from_numpy(np.clip(mask, 0, 1)))
        return (torch.stack(masks),)
