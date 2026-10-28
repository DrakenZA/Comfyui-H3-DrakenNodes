# H3 Low-Resolution Detail Bias: protocol and implementation status

## Objective and status

Test whether changing spatial positional relationships improves native 480p detail
at a fixed spatial token budget. The standalone **H3 Spatial RoPE** MODEL patch
implements the intervention. This is an unproven hypothesis; no real H3 generation
or VAE reconstruction results have been collected here.

Local inspection on 2026-10-28 found an RTX 3060 with 12 GiB VRAM, but no H3 model
or VAE weights in the configured ComfyUI/shared model directories. Native-core CPU
checks use small random weights and cannot answer the visual research question.

## Source-grounded mechanism

Primary implementation: [ComfyUI MiniMax model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).
The locally tested checkout is `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.

* `_axis_from_sqrt_area(dim, patch, sqrt_area)` uses `dim/sqrt_area` to set each
  spatial axis's extent. The nominal continuous extent is centered on 16; the
  sampled grid excludes its upper endpoint, so its sample mean is slightly lower.
* `_frame_grid(h,w)` calls it with `sqrt(h*w)` and 2x2 spatial patches. At a fixed
  aspect ratio the nominal spatial extent is independent of resolution, while
  token spacing changes with resolution.
* `PackedLayout.position_ids` holds `(time,height,width)` for all packed rows,
  distinguishing target `video`, guides `cond`, visual references `ref_img`, text
  and audio. Target rows can also contain pinned prefix tokens under native masks.
* `MiniMaxH3Model.rope_freqs()` multiplies each coordinate axis by inverse rotary
  frequencies. `_forward()` converts these angles into rotation matrices once per
  call and uses the resulting table in transformer attention.

The patch copies the layout positions immediately before this native calculation.
For `reference_scope=target_only`, only `video` rows and columns 1:3 receive
`p_new = 16 + s*(p_native-16)`. Text, guides, reference rows, audio and time remain
bitwise unchanged. Weights, input tensors, masks, sampler and shift parameters are
not edited. Generated audio can still change through joint attention.

The center is the native nominal extent center, not a recomputed sample mean.
Changing that convention would add a coordinate offset relative to conditioning.

Architecture reference: [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3).
The official README describes 768 as the default short edge. That alone does not
establish the model's training-resolution distribution or prove conservative
low-resolution behavior.

Research context: [FiT, arXiv:2402.12376v4](https://arxiv.org/html/2402.12376v4).
FiT investigates 2D RoPE and resolution generalization in an image diffusion
transformer. Its results are motivation, not H3 evidence: simple position
interpolation worsened its tested extrapolation results, while other methods had
different outcomes. This node does not implement VisionNTK or VisionYaRN.

## Controls and experimental order

Use `reference_scope=target_only` explicitly. Older saved workflows may retain
`all_visual`; do not mix their results with this experiment. First use a clean
base checkpoint without detail LoRAs, upscalers, Motion Memory or context windows.
Keep negative/positive conditioning unchanged within each comparison.

1. **VAE control.** Choose sharp natural footage with hair, fabric and skin, plus a
   textured object/static shot. Prepare a 24 fps, 39-frame clip at the exact target
   size (both axes multiples of 32). Encode/decode with H3's video VAE, without any
   diffusion call. Use `example_workflows/h3_vae_roundtrip_api.json`; replace the
   VAE filename and input video. Save source and reconstruction as PNG sequences.
   Check frame count, dimensions and alignment before computing fidelity.
   Also compare `downsample(decode(encode(high_res_source)))` with the direct
   480p VAE round trip of the same source. This isolates a resolution-dependent
   VAE contribution without involving transformer generation.
2. **Native resolution comparison.** Generate at 800x480 and 1280x768 for an exact
   common 5:3 aspect ratio; spatial rows per latent frame are 375 and 960.
   Use the same prompt, conditions and seed, then downsample the high-resolution
   output in pixel space. Record the effective canvas and conditioning after any
   stock node adaptation. 832x480 and 1344x768 have different aspect ratios, so
   using that common pair introduces a framing/crop confound.
3. **Position intervention.** At native 480p, run static scales 0.85, 0.90, 1.00
   and 1.10, holding everything else fixed. Begin with three scenes (close portrait,
   textured object, medium shot with motion) and three seeds: 36 low-resolution
   runs plus nine native high-resolution controls. Keep every output, including
   failures; do not select only the best seed.
4. **Late intervention.** If static scaling shows an effect, test smooth ramps
   from 1.0 to the promising scale over 40-90% and 60-90% of actual denoiser-call
   progress. Count repeated CFG/window calls by sigma, not a mutable call counter.
   Check whether later activation preserves composition better. An all-native
   scale-1 ramp is the baseline; a one-call schedule never progresses past zero.
5. **Sampling control.** Separately compare the base checkpoint at video shift 12
   and 10, keeping audio shift 3, steps, sampler and the actual sigma arrays logged.
   Use the stock H3 sigma-shift mechanism so model and sampler schedules stay
   consistent. Test scale 1 and the promising scale at each shift, as a small
   factorial comparison. Do not use a turbo model trained for another schedule as
   a substitute for this control.

Matching seeds across different grid sizes do not mean matching initial noise or
pixel-aligned motion. Treat the resolution comparison as matched experimental
conditions across seeds, not a direct pixel-error comparison.

## Measurements and records

Record model/VAE hashes, ComfyUI and node commits, full workflow, seed, prompt,
conditioning inputs, actual dimensions, frame count, sampler, steps, sigma arrays,
CFG, shifts, scale/scope/ramp, wall-clock sampling time and peak VRAM. Measure
timing after an equal warm-up policy, separating model load, text encoding and VAE
decode from sampling. Compare overhead at the same resolution/token count.

`scripts/measure_h3_detail.py` accepts video files or lexically sorted PNG frame
folders. Run it using a Python environment containing NumPy and OpenCV:

```powershell
python scripts/measure_h3_detail.py native480.mp4 --output native480.json --size 800 480 --max-frames 39
python scripts/measure_h3_detail.py native768.mp4 --output native768_down480.json --size 800 480 --max-frames 39
python scripts/measure_h3_detail.py output/h3_vae/reconstruction --reference output/h3_vae/source --output vae480.json --size 800 480 --max-frames 39
```

For composition-sensitive comparisons, also choose a per-clip subject ROI using
`--roi x y width height` in normalized coordinates, and resize these crops to the
same analysis size (for example `--size 256 256`). Record the actual subject box
and size separately; a fixed ROI box is insufficient if faces change size inside
it. The script center-crops to the analysis aspect ratio and reports the removed
fraction. Prefer equal-aspect source/analysis sizes to avoid extra cropping.

* **Detail:** Laplacian variance, Sobel energy and high-pass RMS, both full-frame
  and matched-subject crops. These are descriptive proxies: noise, oversharpening
  and hallucinated texture can score well. Inspect eyelashes, hair, fabric and
  surface edges visually, with randomly ordered labels where possible.
* **Composition:** record subject bounding box/area, pose, framing and coarse
  scene layout, plus a human judgment of whether changes are acceptable. Do not
  infer preservation from a sharpness score.
* **Temporal consistency:** adjacent-frame difference, optical-flow magnitude and
  flow-warp residual. The latter still mixes flicker, occlusion, flow errors and
  real changes. Inspect motion and texture shimmer; check lip/audio sync.
* **Artifacts:** manually mark halos, ringing, false texture, anatomy errors,
  spatial distortion and unstable backgrounds. Near-black/white pixel fractions
  are only exposure/clipping proxies, not an artifact detector.
* **VAE fidelity:** native-size MSE/PSNR only for verified aligned source and
  reconstruction pairs. Never use PSNR across independently generated clips.
  The script does not assert alignment from matching dimensions alone.

Analyze comparable opening windows first. Retain full clips for later motion
review, rather than attributing an opening-window failure to long-range changes.

## Interpretation and stopping rules

| Observation | Supported interpretation | Still unresolved |
| --- | --- | --- |
| Sharp 480p source loses fine texture in VAE round trip | VAE contributes to softness on that material | Transformer detail capacity |
| Downsampled 768p generation looks better | Extra-resolution pipeline gives better results | Token count vs VAE vs learned transformer behavior |
| Scale helps at fixed 480p and matched subject size across seeds | Evidence that positional intervention can improve these settings | Whether it mimics native-resolution learning or generalizes |
| Scale only enlarges subjects or changes framing | Spatial geometry changed | Intended detail benefit is unestablished |
| Changing shift fixes native 480p | Schedule contributes | Any independent positional benefit |
| Detail increases with shimmer, false texture or high cost | Intervention has a tradeoff | Whether tradeoff is useful |

These controls cannot uniquely identify all four causal contributions: VAE,
token budget and transformer attention interact. Equal-budget position scaling
tests an accessible intervention; it does not prove a training-distribution story.
Keep scale 1.0 as the default until repeated visual results justify otherwise.
Publish negative results as readily as positive ones.
