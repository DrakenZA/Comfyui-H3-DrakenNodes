"""Refresh the small, private inference subset from a pinned EdgeTAM checkout.

Usage: python scripts/vendor_person_sam2.py /path/to/EdgeTAM
No model weights, demos, CUDA builds, or training code are copied.
"""

import pathlib
import re
import subprocess
import sys


REVISION = "7711e012a30a2402c4eaab637bdb00a521302c91"
NAMESPACE = "_draken_person_sam2"
source = pathlib.Path(sys.argv[1]).resolve()
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REVISION:
    raise SystemExit(f"Expected EdgeTAM commit {REVISION}")
target = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "person_sam2"
files = list((source / "sam2" / "modeling").rglob("*.py"))
files += [source / "sam2" / "sam2_video_predictor.py", source / "sam2" / "utils" / "misc.py",
          source / "sam2" / "utils" / "__init__.py",
          source / "sam2" / "configs" / "edgetam.yaml",
          source / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"]
for path in sorted(files):
    relative = path.relative_to(source / "sam2")
    output = target / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    contents = re.sub(r"\bsam2(?=\.| import)", NAMESPACE, path.read_text(encoding="utf-8"))
    if relative.as_posix() == "modeling/backbones/timm.py":
        contents = contents.replace("pretrained=True,", "pretrained=False,  # Draken: full checkpoint supplies these weights.")
    elif relative.as_posix() == "modeling/sam/transformer.py":
        contents = contents.replace('warnings.simplefilter(action="ignore", category=FutureWarning)\n', "")
    elif relative.as_posix() == "utils/misc.py":
        marker = "    is_bytes = isinstance(video_path, bytes)"
        contents = contents.replace(marker, "    # Draken: a lazy ComfyUI IMAGE adapter; no JPEG/MP4 round trip.\n"
            "    if hasattr(video_path, 'sam2_frames'):\n"
            "        return video_path.sam2_frames(image_size, compute_device)\n" + marker, 1)
    output.write_text(contents, encoding="utf-8", newline="\n")
target.mkdir(parents=True, exist_ok=True)
(target / "__init__.py").write_text('"""Private EdgeTAM inference package; no global Hydra initialization."""\n', encoding="utf-8")
(target / "LICENSE").write_text((source / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
print(f"Copied {len(files)} inference files from {REVISION}")
