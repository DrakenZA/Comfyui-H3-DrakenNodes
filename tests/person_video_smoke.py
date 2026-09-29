"""Opt-in real-model smoke test, using an EdgeTAM checkout's official bedroom sample.

python tests/person_video_smoke.py --reference /path/to/EdgeTAM --models /path/to/checkpoints --output /path/to/results
Does not modify ComfyUI or download models. Tests a middle-frame selection in both directions.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True, type=pathlib.Path)
parser.add_argument("--models", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import person_backend as backend
import person_mask as nodes

torch.set_num_threads(4)
args.output.mkdir(parents=True, exist_ok=True)
paths = sorted((args.reference / "notebooks" / "videos" / "bedroom").glob("*.jpg"))[:8]
images = torch.stack([torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).float() / 255 for path in paths])
count, height, width = images.shape[:3]
prompt = {"frame_index": 3, "box": [0.3125, 0.074, 0.54, 0.9], "points": []}
metrics = []
for name, spec in backend.MODELS.items():
    start = time.perf_counter()
    predictor = backend.build_predictor(name, args.models / spec["filename"])
    # Check namespace isolation even when another SAM package is installed.
    assert "sam2" not in sys.modules
    with torch.inference_mode():
        mask = backend.track_masks(predictor, images, [prompt])
    elapsed = time.perf_counter() - start
    assert tuple(mask.shape) == (count, height, width)
    assert torch.isfinite(mask).all() and all(float(frame.sum()) > 1000 for frame in mask)
    overlay, rgba, _ = nodes.PersonMaskOverlay().apply(images, mask, color="#00FF66")
    transparent, cutout, transparency = nodes.PersonMaskOverlay().apply(images, mask, mode="transparent")
    assert torch.equal(cutout[..., 3], 1 - mask)
    assert torch.equal(transparency, mask)
    assert torch.equal(overlay[mask == 0], images[mask == 0])
    prefix = "edgetam" if name == "EdgeTAM" else "sam21-tiny"
    frames = [Image.fromarray((frame.numpy() * 255).round().astype(np.uint8)) for frame in overlay]
    frames[0].save(args.output / (prefix + "-tracked.gif"), save_all=True, append_images=frames[1:], duration=100, loop=0)
    Image.fromarray((cutout[3].numpy() * 255).round().astype(np.uint8)).save(args.output / (prefix + "-transparent.png"))
    sheet = Image.new("RGB", (960, 420), "#222222")
    draw = ImageDraw.Draw(sheet)
    panels = [images[0], overlay[0], overlay[3], overlay[7], mask[3, ..., None].repeat(1, 1, 3), transparent[3]]
    labels = ["Original frame 0", "Tracked frame 0 (backwards)", "Selection frame 3", "Tracked frame 7 (forwards)", "Person mask", "Transparency preview"]
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x, y = (index % 3) * 320, (index // 3) * 210
        image = Image.fromarray((panel.numpy() * 255).round().astype(np.uint8)).resize((320, 180))
        sheet.paste(image, (x, y + 30))
        draw.text((x + 8, y + 8), label, fill="white")
    sheet.save(args.output / (prefix + "-person-test.png"))
    metrics.append({"model": name, "frames": count, "width": width, "height": height,
                    "seconds_cpu_including_load": round(elapsed, 3),
                    "mask_pixels": [int(frame.sum()) for frame in mask]})
    print(metrics[-1], flush=True)
(args.output / "person-video-test.json").write_text(json.dumps({
    "source": "https://github.com/facebookresearch/EdgeTAM/tree/7711e012a30a2402c4eaab637bdb00a521302c91/notebooks/videos/bedroom",
    "selection": prompt, "torch": torch.__version__, "device": "cpu", "results": metrics}, indent=2), encoding="utf-8")
