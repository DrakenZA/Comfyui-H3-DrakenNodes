"""Fill the visible face masks on matching frames with a color overlay."""

import math

import torch


class FaceMaskHighlighter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The same frame batch used by Face Occlusion Mask."}),
            "face_masks": ("DRAKEN_FACE_MASKS",),
            "color": ("STRING", {"default": "#FF0000", "tooltip": "Highlight color as #RRGGBB, e.g. #00FF00 for green."}),
            "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "0 leaves the frames unchanged; 0.5 is a 50% overlay; 1 fills the visible faces."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("highlighted_frames",)
    FUNCTION = "apply"
    CATEGORY = "DrakenNodes/Face"
    DESCRIPTION = "Fill every visible face mask with the chosen color and opacity, preserving occlusions and soft edges."

    def apply(self, images, face_masks, color="#FF0000", opacity=0.5):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be a ComfyUI IMAGE batch [B, H, W, C] with at least 3 channels")
        if not isinstance(face_masks, dict) or face_masks.get("schema_version") != 1:
            raise ValueError("Connect the face_masks output from Face Occlusion Mask (Draken)")
        batch, height, width = images.shape[:3]
        if (face_masks.get("height"), face_masks.get("width")) != (height, width):
            raise ValueError("Face masks and images must have the same height and width")
        frames = face_masks.get("frames", [])
        if len(frames) != batch:
            raise ValueError("Face masks and images must have the same frame count and order")
        if not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("opacity must be between 0 and 1")
        hex_color = color.strip().removeprefix("#")
        if len(hex_color) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hex_color):
            raise ValueError("color must be a hex RGB color such as #FF0000")
        fill = images.new_tensor([int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)])
        output = images.clone()
        for frame_index, records in enumerate(frames):
            if not records:
                continue
            # Union the compact crops so overlapping faces receive opacity only once.
            mask = images.new_zeros((height, width))
            for record in records:
                crop = record.get("mask")
                origin = record.get("mask_origin")
                if not isinstance(crop, torch.Tensor) or crop.ndim != 2 or origin is None or len(origin) != 2:
                    raise ValueError("Face data must contain mask crops and mask_origin from Face Occlusion Mask (Draken)")
                x, y = origin
                crop_height, crop_width = crop.shape
                left, top = max(0, x), max(0, y)
                right, bottom = min(width, x + crop_width), min(height, y + crop_height)
                if right <= left or bottom <= top:
                    continue
                visible = crop[top - y:bottom - y, left - x:right - x].to(
                    device=images.device, dtype=images.dtype).clamp(0, 1)
                region = mask[top:bottom, left:right]
                region.copy_(torch.maximum(region, visible))
            strength = mask.unsqueeze(-1) * opacity
            rgb = images[frame_index, ..., :3]
            output[frame_index, ..., :3] = rgb * (1 - strength) + fill * strength
        return (output,)
