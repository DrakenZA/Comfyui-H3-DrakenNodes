"""Descriptive detail/temporal proxies for clips or lossless frame folders.

No neural models or external services. Do not rank quality from sharpness alone.
Use --reference ONLY for aligned source/VAE-reconstruction pairs.
Requires numpy and opencv-python, independently of the node's dependencies.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def frames(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        if not files:
            raise ValueError(f"No frames in {path}")
        for file in files:
            image = cv2.imread(str(file))
            if image is None:
                raise ValueError(f"Cannot decode {file}")
            yield image
        return
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError(f"Cannot open {path}")
    try:
        while True:
            ok, image = video.read()
            if not ok:
                break
            yield image
    finally:
        video.release()


def prepare(frame, size, roi=None):
    h, w = frame.shape[:2]
    if roi is not None:
        x, y, rw, rh = roi
        if not (0 <= x < 1 and 0 <= y < 1 and rw > 0 and rh > 0 and x+rw <= 1 and y+rh <= 1):
            raise ValueError("ROI must be normalized x y width height within [0,1]")
        x0, y0, x1, y1 = round(x*w), round(y*h), round((x+rw)*w), round((y+rh)*h)
        frame = frame[y0:y1, x0:x1]
        if frame.size == 0:
            raise ValueError("ROI is empty")
        h, w = frame.shape[:2]
    # Explicit center crop then resize, never stretch mismatched aspect ratios.
    ratio = size[0] / size[1]
    cw, ch = (max(1, round(h*ratio)), h) if w/h > ratio else (w, max(1, round(w/ratio)))
    x0, y0 = (w-cw)//2, (h-ch)//2
    crop = frame[y0:y0+ch, x0:x0+cw]
    method = cv2.INTER_AREA if cw >= size[0] and ch >= size[1] else cv2.INTER_CUBIC
    out = cv2.resize(crop, tuple(size), interpolation=method).astype(np.float32) / 255
    return out, 1.0 - (cw*ch)/(w*h)


def spatial_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx, dy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return {
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "sobel_energy": float(np.mean(dx*dx + dy*dy)),
        "highpass_rms": float(np.sqrt(np.mean((gray-cv2.GaussianBlur(gray, (0, 0), 1.0))**2))),
        "near_black_fraction": float(np.mean(gray <= 1/255)),
        "near_white_fraction": float(np.mean(gray >= 254/255)),
    }


def temporal_metrics(previous, current):
    old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback((old*255).astype(np.uint8), (new*255).astype(np.uint8),
                                       None, 0.5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[:old.shape[0], :old.shape[1]].astype(np.float32)
    mx, my = xx+flow[..., 0], yy+flow[..., 1]
    valid = (mx >= 0) & (mx <= old.shape[1]-1) & (my >= 0) & (my <= old.shape[0]-1)
    warped = cv2.remap(new, mx, my, cv2.INTER_LINEAR)
    return {
        "frame_difference_mae": float(np.mean(np.abs(new-old))),
        "flow_magnitude_pixels": float(np.mean(np.linalg.norm(flow, axis=-1))),
        "flow_warp_residual_mae": float(np.mean(np.abs(warped-old)[valid])) if valid.any() else None,
        "flow_warp_valid_fraction": float(valid.mean()),
    }


def summarize(rows):
    return {key: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                  "p95": float(np.percentile(values, 95))}
            for key in rows[0] if (values := [r[key] for r in rows if r[key] is not None])}


def measure(path, size=(800, 480), max_frames=48, roi=None, reference=None):
    ref = iter(frames(reference)) if reference else None
    spatial, temporal, aligned = [], [], []
    previous, input_size, crop_fraction = None, None, None
    iterator = iter(frames(path))
    try:
        for index in range(max_frames):
            frame = next(iterator, None)
            if frame is None:
                break
            if input_size is None:
                input_size = [frame.shape[1], frame.shape[0]]
            if reference:
                other = next(ref, None)
                if other is None or other.shape != frame.shape:
                    raise ValueError("Aligned reference must have matching frame count and dimensions")
                # Reference error is measured BEFORE crop/resize: native VAE fidelity.
                error = np.mean(((frame.astype(np.float64)-other.astype(np.float64))/255)**2)
                aligned.append({"mse": float(error)})
            image, crop_fraction = prepare(frame, size, roi)
            spatial.append(spatial_metrics(image))
            if previous is not None:
                temporal.append(temporal_metrics(previous, image))
            previous = image
        if reference and len(spatial) < max_frames and next(ref, None) is not None:
            raise ValueError("Aligned reference has extra frames")
    finally:
        iterator.close()
        if ref is not None:
            ref.close()
    if not spatial:
        raise ValueError("No decodable frames")
    report = {
        "input": str(Path(path).resolve()), "input_size": input_size,
        "analysis_size": list(size), "roi_normalized": roi, "aspect_crop_fraction_after_roi": crop_fraction,
        "frames_measured": len(spatial), "spatial": summarize(spatial),
        "temporal": summarize(temporal) if temporal else {},
        "limitations": ["Sharpness can increase with noise, halos or larger subjects; inspect matched-size crops.",
                         "Temporal residual also responds to motion, occlusion and optical-flow errors.",
                         "Composition and artifacts require visual review; these proxies do not establish quality."],
    }
    if aligned:
        mse = float(np.mean([r["mse"] for r in aligned]))
        report["aligned_reference"] = {"input": str(Path(reference).resolve()), "mse": mse,
                                       "psnr_db": 10*math.log10(1/mse) if mse else None,
                                       "perfect_match": mse == 0}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Video file or folder of zero-padded frame images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(800, 480), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--reference", help="Aligned source frames for a VAE reconstruction; NEVER a separate generation")
    args = parser.parse_args()
    if min(args.size) < 8 or args.max_frames < 1:
        parser.error("Analysis dimensions must be >=8 and max-frames must be >=1")
    result = measure(args.input, args.size, args.max_frames, args.roi, args.reference)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Measured {result['frames_measured']} frames -> {args.output}")


if __name__ == "__main__":
    main()
