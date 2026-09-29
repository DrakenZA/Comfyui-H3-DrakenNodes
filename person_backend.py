"""Lazy, isolated EdgeTAM / SAM 2.1 video inference for ComfyUI tensor batches."""

import hashlib
import importlib.util
import logging
import os
import re
import sys
import threading
import urllib.request
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F


LOG = logging.getLogger(__name__)
VENDOR = Path(__file__).resolve().parent / "vendor" / "person_sam2"
NAMESPACE = "_draken_person_sam2"
MODELS = {
    "EdgeTAM": {
        "filename": "edgetam.pt", "config": "configs/edgetam.yaml",
        "url": "https://raw.githubusercontent.com/facebookresearch/EdgeTAM/"
               "7711e012a30a2402c4eaab637bdb00a521302c91/checkpoints/edgetam.pt",
        "sha256": "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df",
    },
    "SAM 2.1 Tiny": {
        "filename": "sam2.1_hiera_tiny.pt", "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sha256": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    },
}
_LOCK = threading.RLock()
_CACHED_MODEL = None  # At most one model, stored on CPU between executions.


def check_cancel():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _check_dependencies(model_name):
    version = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 3, 1):
        raise RuntimeError("Person Video Mask requires PyTorch >= 2.3.1 in ComfyUI's environment")
    try:
        import hydra.utils  # noqa: F401
        import iopath  # noqa: F401
        import tqdm  # noqa: F401
        if model_name == "EdgeTAM":
            import timm  # noqa: F401
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "Person Video Mask needs its optional dependencies in ComfyUI's Python: "
            'python -m pip install -e ".[person-mask]". '
            "Use a torchvision build that matches ComfyUI's torch. " + str(error)
        ) from error


@lru_cache(maxsize=4)
def _checksum(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path, expected):
    stat = path.stat()
    return _checksum(str(path), stat.st_size, stat.st_mtime_ns) == expected


def ensure_checkpoint(model_name):
    import folder_paths

    spec = MODELS[model_name]
    directory = Path(folder_paths.models_dir) / "person_segmentation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec["filename"]
    with _LOCK:
        if path.is_file():
            if not _verify(path, spec["sha256"]):
                raise RuntimeError(f"Model checksum mismatch at {path}; move it aside and rerun")
            return path
        temporary = path.with_name(path.name + ".download")
        try:
            LOG.info("Downloading %s to %s", model_name, path)
            with urllib.request.urlopen(spec["url"], timeout=30) as response, temporary.open("wb") as output:
                while True:
                    check_cancel()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not _verify(temporary, spec["sha256"]):
                raise RuntimeError(f"Download of {spec['filename']} failed checksum validation")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _load_private_package():
    if NAMESPACE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            NAMESPACE, VENDOR / "__init__.py", submodule_search_locations=[str(VENDOR)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[NAMESPACE] = package
        spec.loader.exec_module(package)


def build_predictor(model_name, checkpoint):
    """Load pure config data without initializing/changing Hydra's global state."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    _load_private_package()
    config = OmegaConf.load(VENDOR / MODELS[model_name]["config"])
    config.model._target_ = NAMESPACE + ".sam2_video_predictor.SAM2VideoPredictor"
    config.model.sam_mask_decoder_extra_args = {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }
    config.model.binarize_mask_from_pts_for_mem_enc = True
    # No compiled CUDA extension or connected-component postprocessing required.
    config.model.fill_hole_area = 0
    predictor = instantiate(config.model, _recursive_=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state["model"], strict=True)
    return predictor.eval()


class TensorFrames:
    """Resize/normalize one frame on demand, rather than storing a 1024px copy of the video."""

    def __init__(self, images):
        self.images = images

    def sam2_frames(self, image_size, device):
        self.image_size = image_size
        height, width = self.images.shape[1:3]
        return self, height, width

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        check_cancel()
        frame = self.images[index, ..., :3].detach().to(device="cpu", dtype=torch.float32)
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        frame = F.interpolate(frame, (self.image_size, self.image_size), mode="bilinear",
                              align_corners=False, antialias=True)[0].clamp(0, 1)
        mean = frame.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = frame.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (frame - mean) / std


def compute_device(choice):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose auto or cpu")
        return torch.device("cuda", torch.cuda.current_device())
    if choice != "auto":
        raise ValueError("device must be auto, cuda, or cpu")
    try:
        from comfy.model_management import get_torch_device
        device = get_torch_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The upstream predictor uses CUDA/CPU attention. Other Comfy backends use CPU here.
    return device if device.type == "cuda" else torch.device("cpu")


def _progress(total):
    try:
        from comfy.utils import ProgressBar
    except ImportError:
        return None
    return ProgressBar(total)


def track_masks(predictor, images, prompts, offload_state=True, progress=None):
    """A fresh state for every batch; always return masks in original frame order."""
    count, height, width = images.shape[:3]
    result = torch.zeros((count, height, width), dtype=torch.float32, device="cpu")
    seen = set()
    state = predictor.init_state(TensorFrames(images), offload_video_to_cpu=True,
                                 offload_state_to_cpu=offload_state)
    for prompt in prompts:
        check_cancel()
        points = prompt["points"]
        box = prompt["box"]
        predictor.add_new_points_or_box(
            state, frame_idx=prompt["frame_index"], obj_id=1,
            points=[[x * width, y * height] for x, y, _ in points] if points else None,
            labels=[label for _, _, label in points] if points else None,
            box=[box[0] * width, box[1] * height, box[2] * width, box[3] * height] if box else None,
        )
    start = prompts[0]["frame_index"]
    for reverse in (False, True) if start > 0 else (False,):
        for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, reverse=reverse):
            check_cancel()
            result[frame_index] = (logits[object_ids.index(1), 0] > 0).to(device="cpu", dtype=torch.float32)
            if frame_index not in seen:
                seen.add(frame_index)
                if progress is not None:
                    progress.update(1)
    if len(seen) != count:
        raise RuntimeError(f"Tracker returned {len(seen)} of {count} frames")
    return result


def segment_video(images, prompts, model_name, device_choice, offload_state, cache_model):
    global _CACHED_MODEL

    if model_name not in MODELS:
        raise ValueError(f"Unknown person mask model: {model_name}")
    _check_dependencies(model_name)
    device = compute_device(device_choice)
    with _LOCK:
        checkpoint = ensure_checkpoint(model_name)
        key = (model_name, str(checkpoint), checkpoint.stat().st_mtime_ns)
        if _CACHED_MODEL is None or _CACHED_MODEL[0] != key:
            _CACHED_MODEL = None
            predictor = build_predictor(model_name, checkpoint)
        else:
            predictor = _CACHED_MODEL[1]
        try:
            check_cancel()
            if device.type == "cuda":
                try:
                    from comfy.model_management import free_memory
                except ImportError:
                    pass
                else:
                    size = sum(p.numel() * p.element_size() for p in predictor.parameters())
                    free_memory(size + 1536 * 1024**2, device)
            predictor.to(device)
            precision = nullcontext()
            if device.type == "cuda":
                precision = torch.autocast("cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported()
                                                          else torch.float16))
            with torch.inference_mode(), precision:
                return track_masks(predictor, images, prompts, offload_state, _progress(len(images)))
        finally:
            # Release VRAM before the rest of the Comfy graph runs; retain only one CPU model if requested.
            predictor.to("cpu")
            _CACHED_MODEL = (key, predictor) if cache_model else None
