"""Select, track, and color a person's visible silhouette through a frame batch."""

import importlib.util
import json
import math
import uuid
from pathlib import Path

import torch


def _validate_images(images):
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] not in (3, 4)
            or min(images.shape[:3]) < 1 or not images.is_floating_point()):
        raise ValueError("images must be a nonempty ComfyUI IMAGE batch [frames, height, width, 3 or 4]")
    return images.shape[:3]


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Selection coordinates must be finite numbers between 0 and 1")
    return float(value)


def _prompt(data, frame_index, allow_empty=False):
    if not isinstance(data, dict):
        raise ValueError("Selection must be a JSON object with box and/or points")
    box = data.get("box")
    if box is not None:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("box must be [left, top, right, bottom] in normalized coordinates")
        box = [_coordinate(value) for value in box]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Selection box must have positive width and height")
    points = data.get("points", [])
    if not isinstance(points, list):
        raise ValueError("points must be a list of [x, y, label]; label 1 includes, 0 excludes")
    parsed = []
    for point in points:
        if not isinstance(point, list) or len(point) != 3 or point[2] not in (0, 1):
            raise ValueError("Each point must be [x, y, label]; label 1 includes, 0 excludes")
        parsed.append([_coordinate(point[0]), _coordinate(point[1]), int(point[2])])
    if not allow_empty and box is None and not parsed:
        raise ValueError("Select a person: queue Person Selection to load its preview, draw a box or include point, then queue again")
    return {"frame_index": frame_index, "box": box, "points": parsed}


def _json(text, label):
    try:
        return json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be valid JSON") from error


def _frame_index(value, count):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise ValueError(f"frame_index must be between 0 and {count - 1} (zero-based)")
    return value


class PersonSelection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "frame_index": ("INT", {"default": 0, "min": 0, "max": 1000000,
                "tooltip": "Frame to select on (zero-based). Queue after changing this to refresh the preview."}),
            "selection": ("STRING", {"default": '{"box":null,"points":[]}', "multiline": True,
                "tooltip": "Draw on the preview, or enter normalized box/points JSON. 1 includes; 0 excludes."}),
        }}

    RETURN_TYPES = ("DRAKEN_PERSON_SELECTION",)
    RETURN_NAMES = ("person",)
    FUNCTION = "select"
    CATEGORY = "DrakenNodes/Person"
    OUTPUT_NODE = True
    DESCRIPTION = "Queue to view a frame, then drag a box or add include/exclude points on the preview and queue again."

    def select(self, images, frame_index=0, selection='{"box":null,"points":[]}'):
        import folder_paths
        import numpy as np
        from PIL import Image

        count, height, width = _validate_images(images)
        frame_index = _frame_index(frame_index, count)
        prompt = _prompt(_json(selection, "selection"), frame_index, allow_empty=True)
        directory = Path(folder_paths.get_temp_directory()) / "draken_person_selector"
        directory.mkdir(parents=True, exist_ok=True)
        filename = "selection_" + uuid.uuid4().hex + ".png"
        # Only the selected frame is saved; tracking reads tensors directly.
        frame = images[frame_index, ..., :3].detach().to(device="cpu", dtype=torch.float32)
        frame = torch.nn.functional.interpolate(frame.permute(2, 0, 1)[None],
            size=(max(1, round(height * min(1, 1024 / max(height, width)))),
                  max(1, round(width * min(1, 1024 / max(height, width))))),
            mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        Image.fromarray((frame.clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(directory / filename)
        data = {"schema_version": 1, "frame_count": count, "height": height, "width": width,
                "prompt": prompt}
        preview = {"filename": filename, "subfolder": "draken_person_selector", "type": "temp",
                   "frame_index": frame_index, "frame_count": count, "height": height, "width": width}
        return {"ui": {"person_selector": [preview]}, "result": (data,)}


def _backend():
    # Keep optional imports out of node registration; also allow standalone CPU tests.
    try:
        from . import person_backend
        return person_backend
    except ImportError:
        spec = importlib.util.spec_from_file_location("draken_person_backend", Path(__file__).with_name("person_backend.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class PersonVideoMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Same frame batch, in the same order, as Person Selection."}),
            "person": ("DRAKEN_PERSON_SELECTION",),
            "model": (["EdgeTAM", "SAM 2.1 Tiny"],),
            "device": (["auto", "cuda", "cpu"],),
            "offload_state": ("BOOLEAN", {"default": True, "tooltip": "Store tracking memory on CPU to save VRAM."}),
            "cache_model": ("BOOLEAN", {"default": True, "tooltip": "Keep one model on CPU between runs. VRAM is released after each run."}),
        }, "optional": {
            "corrections": ("STRING", {"default": "[]", "multiline": True,
                "tooltip": 'Optional extra prompts for the same person: [{"frame_index":12,"points":[[0.5,0.5,1]]}].'}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("person_mask",)
    FUNCTION = "track"
    CATEGORY = "DrakenNodes/Person"
    DESCRIPTION = "Track the selected visible person forwards and backwards through all frames using EdgeTAM or SAM 2.1 Tiny."

    def track(self, images, person, model="EdgeTAM", device="auto", offload_state=True,
              cache_model=True, corrections="[]"):
        count, height, width = _validate_images(images)
        if not isinstance(person, dict) or person.get("schema_version") != 1:
            raise ValueError("Connect person from Person Selection (Draken)")
        if (person.get("frame_count"), person.get("height"), person.get("width")) != (count, height, width):
            raise ValueError("Person Selection and Person Video Mask need the same frame count, dimensions, and order")
        initial = person.get("prompt", {})
        index = _frame_index(initial.get("frame_index"), count)
        if not initial.get("points") and initial.get("box") is None:
            # First queue displays the selector without downloading/running a model or failing the graph.
            try:
                from comfy_execution.graph_utils import ExecutionBlocker
            except ImportError:
                pass
            else:
                return (ExecutionBlocker(None),)
        prompts = [_prompt(initial, index)]
        if prompts[0]["box"] is None and not any(point[2] == 1 for point in prompts[0]["points"]):
            raise ValueError("Initial selection needs an include point or a box")
        additional = _json(corrections, "corrections")
        if not isinstance(additional, list):
            raise ValueError("corrections must be a JSON list of frame prompts")
        used_frames = {index}
        for data in additional:
            if not isinstance(data, dict):
                raise ValueError("Each correction must have a frame_index and box/points")
            frame = _frame_index(data.get("frame_index"), count)
            if frame in used_frames:
                raise ValueError("Use one prompt per frame; put all points for that frame together")
            used_frames.add(frame)
            prompts.append(_prompt(data, frame))
        return (_backend().segment_video(images, prompts, model, device, offload_state, cache_model),)


class PersonMaskOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "person_mask": ("MASK",),
            "mode": (["color", "transparent"],),
            "color": ("STRING", {"default": "#FF0000", "tooltip": "Replacement color as #RRGGBB."}),
            "opacity": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.05,
                "tooltip": "Color strength, or how transparent the person becomes. 1 fully replaces/removes; 0 keeps the original."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("preview", "rgba_frames", "transparency_mask")
    FUNCTION = "apply"
    CATEGORY = "DrakenNodes/Person"
    DESCRIPTION = "Fill the tracked silhouette with a color or make it transparent. RGBA output retains real alpha; preview shows a checkerboard."

    def apply(self, images, person_mask, mode="color", color="#FF0000", opacity=1.0):
        count, height, width = _validate_images(images)
        if person_mask.ndim != 3 or tuple(person_mask.shape) != (count, height, width):
            raise ValueError("person_mask must match the images' frame count, height, and width")
        if not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("opacity must be between 0 and 1")
        if mode not in ("color", "transparent"):
            raise ValueError("mode must be color or transparent")
        value = color.strip().removeprefix("#")
        if len(value) != 6 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("color must be a hex RGB color such as #FF0000")
        rgb = images[..., :3]
        mask = person_mask.to(device=images.device, dtype=images.dtype)
        if not torch.isfinite(mask).all():
            raise ValueError("person_mask contains non-finite values")
        strength = mask.clamp(0, 1)[..., None] * opacity
        alpha = images[..., 3:4] if images.shape[-1] == 4 else torch.ones_like(strength)
        if mode == "color":
            fill = images.new_tensor([int(value[i:i+2], 16) / 255 for i in (0, 2, 4)])
            rgb = rgb * (1 - strength) + fill * strength
        else:
            alpha = alpha * (1 - strength)
        rgba = torch.cat((rgb, alpha), dim=-1)
        # Preview is always RGB and clearly displays alpha, including for RGBA inputs.
        ys = torch.arange(height, device=images.device)[:, None] // 16
        xs = torch.arange(width, device=images.device)[None, :] // 16
        checker = (0.3 + ((xs + ys) % 2).to(images.dtype) * 0.15)[None, ..., None]
        preview = rgb * alpha + checker * (1 - alpha)
        return preview, rgba, 1 - alpha[..., 0]
