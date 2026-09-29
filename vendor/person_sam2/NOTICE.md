# EdgeTAM / SAM 2 inference code

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Source: https://github.com/facebookresearch/EdgeTAM
Pinned commit: `7711e012a30a2402c4eaab637bdb00a521302c91`
License: Apache 2.0 (the accompanying `LICENSE` applies to this directory).

This is only the shared video inference code and the EdgeTAM / SAM 2.1 Tiny
configs. Model weights, training, notebooks, demos, image predictor, automatic
mask generator, and compiled CUDA extensions are excluded.

Local changes, reproducible with `scripts/vendor_person_sam2.py`:

- Imports/config targets use the private `_draken_person_sam2` namespace.
- Package initialization does not initialize or change global Hydra state.
- The video loader accepts a lazy tensor frame adapter supplied by our node.
- The timm backbone uses `pretrained=False`: the full EdgeTAM checkpoint
  already includes its weights, so another checkpoint/download is unnecessary.
- The attention module does not change global warning filters.

The node integration outside this directory is independent code. The upstream
EdgeTAM and SAM 2.1 checkpoints are Apache 2.0; downloaded weights are not
distributed in this repository.
