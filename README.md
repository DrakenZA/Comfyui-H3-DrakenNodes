# Comfyui-H3-DrakenNodes

MiniMax H3 long-form nodes for ComfyUI: joint context-window sampling, footage extension, audio lock, seam probe.
Joint context-window sampling for **MiniMax H3** in ComfyUI: one long audio-video latent (30 s, 60 s, more),
denoised every step as a set of overlapping windows that each look like a normal H3 clip, fused in the overlaps.
Optionally seeded with the tail of existing footage so the result is a true extension of that footage.

This is the FreeNoise / AnimateDiff / Wan / LTX-2 "context windows" approach, which ComfyUI core already ships,
wired to H3's real temporal grid. It is **not** clip chaining: no tail is copied between generations, no VAE
round trip, no compounding drift. Every frame and every audio tick is denoised once, from noise, in a single
trajectory, and the audio is one continuous stream.

## Prior art, and what is different here

Two packs already run ComfyUI's context windows on H3 with a handler subclass:
[wordbrew/ComfyUI-H3-Toolkit](https://github.com/wordbrew/ComfyUI-H3-Toolkit) (`windowing.py`, with real
renders measured through August 2026) and [ckinpdx/ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools)
(`nodes_windows.py`). This pack is a third, narrower implementation of the same idea. What it does differently:

* **No monkey-patching.** The toolkit installs hooks on the `MiniMaxH3` class, replaces `PackedLayout` globally and
  passes the window offset through a consume-once module global. Here the per-window layout is built once, cached
  and handed to the model inside the per-window payload, so nothing outside the handler object changes and two
  graphs cannot leak settings into each other.
* **Exact audio windows.** Windows are planned so every window's audio has the standard tick count for its length,
  and strides on the 51-frame grid start on exact ticks. The toolkit maps per token with floor/ceil, which can hand
  a window one extra tick.
* **Guide clips are cropped, not dropped.** A multi-frame `MiniMaxH3AddGuide` clip that crosses a window seam is
  cropped to whole token cycles and shifted; guide audio is cropped to the window's ticks. The toolkit keeps or
  drops a keyframe by its start index only.
* **Footage extension in one node.** `H3 Long AV Latent` writes the encoded tail of existing footage into the head of
  the long latent under the model's native denoise mask. The toolkit reaches the same state through
  `H3EncodeAV` + `H3LatentPin`, or via its chunked pipeline.
* **VRAM budgeted per window** when `expected_total_frames` is set (core's own wrapper skips packed latents).

What the toolkit has that this pack does not: months of measured renders, the chunked pipeline (latent pin, motion
context, per-chunk prompts, seam checks) that it recommends for locked-off shots, masked V2V, subject crop,
prompt tooling, 34 example workflows. Its measurements shaped this pack's defaults:

* `absolute_positions` on. With every window placed at the clip origin, each window renders "the opening of the
  shot" and the overlaps crossfade between two openings: flicker at the window period. Shifting the target rows'
  RoPE time by `5/3 x window_start_frame` fixed it in their A/B.
* Static windows by default. Moving windows between steps was measured as harmful (before absolute positions);
  `phase_alternate` stays available but off.
* Causal anchor frame off, window and stride on the 17-frame grid.
* Their nulls: FreeNoise and overlap-linear fusing made no visible difference on their test; background content
  can still disagree between windows on a locked-off shot, which the overlap blend cannot fix. Moving-camera shots
  hide seams; static shots expose them.

Joint windowed denoising is still the approach with no tail copying, no VAE round trips and no compounding
drift, at the price of `windows x steps` model evaluations (about the same compute as chaining).

## How it works (from ComfyUI core and the H3 model code)

* H3 packs 24 fps video into latent tokens with the repeating pattern `1,4,4,4,4` frames per token (17 frames per 5
  tokens). A valid clip is `17k+5` frames. Audio runs at 40 Hz, so 3 frames = 5 audio ticks; lengths
  `39, 90, 141, 192, 243, 294, 345` frames have an integer tick count ("audio-exact").
* Core `comfy.context_windows` already slices video + audio latents per window, runs the model on each window every
  step, and fuses overlaps with pyramid weights (built for LTX-2). H3 does not plug in because it has no modality
  mapper, its audio latent keeps time on axis 3 (`[B,32,2,T]`), and its packed layout / keyframe indices are global.
* This pack subclasses the core handler and adds exactly that:
  * each window's target and keyframe rows are placed at their absolute position on the clip's RoPE time axis
    (`absolute_positions`, default on);
  * video-token -> audio-tick mapping on H3's grid, audio sliced on axis 3, per-modality fusion;
  * every window starts on a 5-token (17-frame) boundary and is a valid `5j+2`-token clip, so the model sees the
    token phase it was trained on; strides that are multiples of 51 frames keep the audio grid exact;
  * the H3 payload is rewritten per window: the global `PackedLayout` is dropped and rebuilt per window (cached),
    keyframes and guide clips (`MiniMaxH3AddGuide`) are shifted into window-local frame indices, cropped to whole
    token cycles, and their audio cropped to the window's ticks;
  * the core "causal anchor frame" is disabled (it would break the token phase); window phase alternates between
    steps so seams never sit on the same tokens; FreeNoise shuffles video and audio noise with matching windows.
* Existing footage enters through the model's native denoise mask: the encoded tail is written into the first
  tokens of the long latent and masked out, so the DiT treats those rows at its 0.999 "clean conditioning" timestep,
  the same way it treats keyframe rows, with zero extra tokens.

Nothing in ComfyUI core or the H3 model object is monkey-patched.

## Seam probe and halo context (new, experimental)

The toolkit's measurements say the remaining windowing problem is not blending: neighbouring windows *generate*
different content for the same frames (a locked-off background drifts), and no crossfade curve fixes a disagreement
about content. Two things in this pack act on that:

* **`probe_seams`** measures it. Every step, for every pair of neighbouring windows, the handler compares the two
  windows' predictions over their shared tokens and logs the relative disagreement (mean and max). Zero means the
  windows agree; a number that stays high late in the schedule means the seam is content, not blending. Cheap.
* **`halo_frames`** attacks it during generation. From `halo_start_percent` of the steps onward, each window is
  also given the fused prediction of the frames just before and after it (from the previous step, so it is the
  neighbours' current belief) as conditioning rows at their absolute time on the clip. The window then denoises
  with its neighbours' context in view instead of only being averaged with them afterwards. It costs extra rows
  per window (17 frames each side = 5 tokens x the frame's rows) and is untested on real weights. Start with
  `halo_frames` 17 and `halo_start_percent` 0.3; use the probe to see whether the disagreement drops.

**Result so far (2026-09-11, real renders, 4-step turbo):** the halo made the output look strange. The likely
reason is structural: H3 pins every conditioning row at its 0.999 "clean" timestep, so the coarse prediction from
the previous step was treated as clean context and the windows anchored to blur. It stays available for experiments
but is not recommended.

* **`margin_frames`** is the replacement. Each window is evaluated over its own span plus a margin on each side,
  taken from the shared latent at the current noise level (nothing coarse is injected), but only the inner span is
  fused. Edge tokens, which only ever see one-sided context and disagree the most, are never used. Same schedule
  and seams, a bit more compute per window. Try 17 or 34.

Both are per-branch handler state, so two model branches with different settings never interact.

## Nodes

All five nodes are under **Add Node → DrakenNodes → H3** (search "Draken" or "H3" finds them; every node name ends in "(Draken)" and the node IDs are prefixed `Draken` so they cannot collide with other H3 packs).

| Node | What it does |
| --- | --- |
| **H3 Context Windows (Draken)** (model patch) | Installs the handler. `window_frames` (default 141 = 5.9 s), `stride_frames` (default 51), fuse method, phase alternation, FreeNoise, per-region prompts. `expected_total_frames` lets it budget VRAM per window instead of per full latent. Outputs an info string. |
| **H3 Long AV Latent, footage prefix (Draken)** | Builds the long empty AV latent (snapped to `17k+5`, and with `snap_length_to_audio_grid` to an audio-exact length). Three ways to seed it: `prefix_latent` (a previous H3 AV latent, e.g. the last KSampler output; its tail is copied straight in, no VAEs), or `prefix_frames` + `vae` (footage encoded), optionally with `prefix_audio` + `audio_vae`. The last `prefix_context_frames` (default 39) become a hard masked prefix, with optional feathering. Audio is only taken when something is wired into `prefix_audio` / from the latent; otherwise H3 generates it. Outputs the latent, the number of prefix frames to trim later, and an info string. |
| **H3 Audio Lock, long latent (Draken)** | Pins a real soundtrack into the whole long latent (from an offset, with strength) so only video is generated: one continuous track, lip sync and beats across the entire duration. Ported from the toolkit. |
| **H3 Window Plan (Draken)** | Prints the window schedule for a length / window / stride, or (`windows -> length`) the audio-exact length that gives N clean windows with no clamped last window. |
| **H3 Trim Prefix, image+audio (Draken)** | Drops the footage prefix from the decoded frames and the matching seconds of audio. |

## Wiring

```
Load Diffusion Model (minimax_h3_fl2va_*.safetensors)
  -> ModelSamplingMiniMaxH3 (shift_video 12, shift_audio 3)
  -> H3 Context Windows (window 141, stride 51, expected_total_frames = your length)
  -> KSampler.model

Load CLIP (qwen3vl_32b_minimax_h3_*.safetensors, type "minimax")
Load VAE (minimax_h3_video_vae_fp16) ; Load VAE (minimax_h3_audio_vae_fp32)

MiniMax H3 Image to Video (clip, vae, prompt, width, height, first_frame = last frame of the footage)
  -> positive (also use it as negative; H3 is CFG-free, cfg = 1.0)
  (ignore its LATENT output)

Load Video -> Get Video Components (images, audio)
  -> H3 Long AV Latent (width, height, length_frames, vae, audio_vae, prefix_frames = images, prefix_audio = audio)
  -> KSampler.latent_image        (steps 20, cfg 1.0, res_multistep / simple, denoise 1.0)

KSampler -> VAE Decode (video vae)      -> images
         -> VAE Decode Audio (audio vae) -> audio
  -> H3 Trim Prefix (prefix_frames_used from the latent node)
  -> Create Video (fps 24, audio) -> Save Video
```

Notes:

* Use the same width/height in *Image to Video* and *Long AV Latent*. The Image to Video node's `length` only affects
  its own (unused) latent; its `first_frame` keyframe lands at frame 0 of the long timeline, which coincides with the
  footage prefix and also shows the frame to the Qwen text encoder.
* To pin a frame later in the long timeline, add `Add Guide for MiniMax H3` on the **long** latent with `frame_idx`
  (negative counts from the end). Guides are automatically remapped into the windows that contain them.
* For a prompt that evolves over time: build several positive conditionings, `Conditioning Combine` them in timeline
  order, and enable `split_conds_to_windows`. Each window picks the prompt for its position.
* `denoise` must stay 1.0 and `cfg` 1.0 (H3 constraints, unchanged by this pack).
* Text-to-video with no footage: leave `prefix_frames` empty. Reference-to-video: use `MiniMax H3 Reference to Video`
  for the conditioning; reference blocks ride along into every window.

`example_workflows/h3_extend_footage_api.json` is the graph above in ComfyUI API format (run it with the API or
adapt it in the editor). `h3_extend_footage_ref2va_api.json` is the same extension on the **Ref2VA** checkpoint:
`MiniMax H3 Reference to Video` supplies the conditioning (reference stills for identity, `<Picture N>` prompt,
`length` set to the long length because it also caps reference videos), and `H3 Long AV Latent (Draken)` supplies
the latent with the footage prefix. No keyframes are used; reference blocks ride into every window.

## Parameters that matter

| Setting | Guidance |
| --- | --- |
| `window_frames` | 141 (5.9 s) is the sweet spot: comfortably inside H3's trained 124-362 range, audio-exact. 192 or 243 give each window more context at higher VRAM/compute. |
| `stride_frames` | 51 = 90-frame overlap with a 141 window (heavy blending, best continuity, 13 windows per step for 30 s). 102 halves the cost with a 39-frame overlap. Keep it a multiple of 51 for exact audio. |
| `fuse_method` | `pyramid` (default). `relative` is the running-average variant; `overlap-linear` ramps only inside the overlap. |
| `absolute_positions` | On. Each window is told where it sits on the clip's time axis. Off reproduces origin placement (flickers). |
| `phase_alternate` | Off. Experimental: shifts interior window starts by half a stride on odd steps. |
| `freenoise` | On. Correlated initial noise across windows (the toolkit measured no visible difference either way). |
| `prefix_context_frames` | 39 (1.6 s, audio-exact). 90 for more motion context. |
| `feather_tokens` | 0-3. Leaves a soft handoff after the hard prefix if the join looks stiff. |
| `length_frames` | Anything; snapped up to `51n-12` (39, 90, 141, ..., 753 = 31.4 s, 1467 = 61 s). |
| `probe_seams` | Off. Logs per-step window disagreement in the overlaps. |
| `halo_frames` / `halo_start_percent` | 0 / 0.3. Experimental halo context; measured as harmful, see above. |
| `margin_frames` | 0. Evaluate each window with extra context on both sides, fuse only the inner span. |

Cost estimate: windows per step = about `(length - window) / stride + 1`. 30 s at 141/51 is 13 windows x 20 steps
= 260 window evaluations, each the cost of one 5.9 s clip step.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DrakenZA/Comfyui-H3-DrakenNodes.git
```

No extra Python dependencies (torch, torchaudio and ComfyUI core only). Requires a ComfyUI with native MiniMax H3
support (August 2026 or newer, `MiniMaxH3AddGuide` present).

### Colab

```bash
# 1. ComfyUI
!git clone https://github.com/Comfy-Org/ComfyUI.git
%cd ComfyUI
!pip install -q -r requirements.txt

# 2. this pack
!git clone https://github.com/DrakenZA/Comfyui-H3-DrakenNodes.git custom_nodes/Comfyui-H3-DrakenNodes

# 3. weights from Comfy-Org/MiniMax-H3 (top-level folders: diffusion_models/, text_encoders/, vae/)
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download
repo = "Comfy-Org/MiniMax-H3"
for f in ["diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",   # 21 GB (fp8_scaled variant is also 21 GB)
          "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",          # 15.7 GB
          "vae/minimax_h3_video_vae_fp16.safetensors",                            # 5.2 GB
          "vae/minimax_h3_audio_vae_fp32.safetensors"]:                           # 0.6 GB
    hf_hub_download(repo, f, local_dir="models")

# 4. run (add --use-sage-attention if sageattention is installed; --disable-pinned-memory on flaky hosts)
!python main.py --listen --port 8188
```

Available variants in the repo (sizes as listed on Hugging Face):

| Folder | Files |
| --- | --- |
| `diffusion_models/` | `minimax_h3_fl2va_bf16` 66.3 GB, `_int8_convrot` 34 GB, `_pruned_bf16` 40.2 GB, `_pruned_fp8_scaled` 21 GB, `_pruned_int8_convrot` 21 GB; same five for `ref2va` |
| `text_encoders/` | `qwen3vl_32b_minimax_h3_bf16` 51.5 GB, `_int8_convrot` 27.1 GB, `_nvfp4_awq` 15.7 GB |
| `vae/` | `minimax_h3_video_vae_fp16` 5.2 GB, `minimax_h3_audio_vae_fp32` 0.6 GB |

An A100 80 GB runs the full int8 diffusion model plus the int8 text encoder. On a 40 GB card use a pruned 21 GB
model and the nvfp4 text encoder. Colab's free T4 (15 GB) is too small for H3.

## Tests

`tests/run_tests.py --comfy /path/to/ComfyUI` runs on CPU against real ComfyUI core with a stand-in model that
returns `-x` for every window. The fused result must equal `-x` for the whole latent exactly, which validates window
slicing, audio-tick mapping, packing/unpacking, per-modality fusion and phase alternation. Further checks cover
keyframe/guide remapping and cropping, audio guide cropping, denoise-mask slicing, reference blocks, FreeNoise, the
long-latent prefix/mask construction, trimming, planning and the VRAM wrapper.

```
python tests/run_tests.py --comfy C:\path\to\ComfyUI
...
ALL TESTS OK
```

What is **not** covered: a real H3 render. The pack was written against the H3 code in ComfyUI core (September
2026) but has not yet been run with the weights; treat the first render as the integration test. If the DiT
complains about a layout signature or a keyframe index, that is the place to look (`handler.py`,
`_window_payload`).

## Known limits

* No closed-loop / looping schedules in v1 (a wrapped window breaks the token phase).
* Windows are contiguous only (no strided "uniform" schedules), for the same reason.
* Strides that are not multiples of 51 frames make interior audio windows start on a rounded tick (at most 17 ms).
* Batch size 1 (H3 limit).
* The VRAM estimate wrapper needs `expected_total_frames`; without it ComfyUI budgets memory for the full latent
  (safe but may offload more than necessary).
