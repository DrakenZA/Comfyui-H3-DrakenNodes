"""Draw one colored square per face from Face Occlusion Mask's separate data."""

import math

import torch


# RGB, in the requested fixed order. Repeat after the tenth face.
FACE_COLORS = (
    (1.0, 0.0, 0.0),  # red
    (0.0, 0.0, 1.0),  # blue
    (0.0, 1.0, 0.0),  # green
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.5, 0.0),  # orange
    (0.5, 0.0, 1.0),  # purple
    (1.0, 0.5, 0.75), # pink
    (0.0, 0.5, 0.5),  # teal
)


def _paint_rectangle(image, x0, y0, x1, y1, color):
    height, width = image.shape[:2]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 > x0 and y1 > y0:
        image[y0:y1, x0:x1, :3] = color


def _draw_square(image, bbox, size_scale, line_width, color):
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(1, round(max(x1 - x0, y1 - y0) * size_scale))
    left, top = round(center_x - side / 2), round(center_y - side / 2)
    right, bottom = left + side, top + side
    stroke = min(line_width, (side + 1) // 2)
    _paint_rectangle(image, left, top, right, top + stroke, color)
    _paint_rectangle(image, left, bottom - stroke, right, bottom, color)
    _paint_rectangle(image, left, top, left + stroke, bottom, color)
    _paint_rectangle(image, right - stroke, top, right, bottom, color)


class FaceMaskSquares:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The same frame batch used by Face Occlusion Mask."}),
                "face_masks": ("DRAKEN_FACE_MASKS",),
                "size_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05,
                    "tooltip": "1 fits the visible mask bounds; 0.8 is smaller, 1.2 is larger. Squares stay centered."}),
                "line_width": ("INT", {"default": 3, "min": 1, "max": 64,
                    "tooltip": "Outline thickness in image pixels."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("outlined_frames",)
    FUNCTION = "draw"
    CATEGORY = "DrakenNodes/Face"
    DESCRIPTION = "Draw square face outlines on matching frames, ordered left to right: red, blue, green, yellow, and more."

    def draw(self, images, face_masks, size_scale=1.0, line_width=3):
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
        if not math.isfinite(size_scale) or size_scale <= 0 or line_width < 1:
            raise ValueError("size_scale and line_width must be positive")

        output = images.clone()
        colors = [torch.tensor(rgb, dtype=images.dtype, device=images.device) for rgb in FACE_COLORS]
        for frame_index, records in enumerate(frames):
            ordered = sorted(records, key=lambda record: ((record["bbox"][0] + record["bbox"][2]) / 2,
                                                         (record["bbox"][1] + record["bbox"][3]) / 2))
            for face_index, record in enumerate(ordered):
                _draw_square(output[frame_index], record["bbox"], size_scale, int(line_width),
                             colors[face_index % len(colors)])
        return (output,)
