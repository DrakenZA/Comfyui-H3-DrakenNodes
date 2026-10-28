# H3 Spatial RoPE (experimental)

Changes H3's spatial rotary coordinates while keeping output dimensions, latent
tokens, weights and sampling schedules unchanged. This is an experiment, with no
checkpoint renders or proven detail improvement yet. It can change framing,
subject size, coherence and generated audio through joint attention.

Search **H3 Spatial RoPE (Draken, experimental)**. Connect the H3 model after its
loader/LoRAs/sampling patches, then connect this node's MODEL output to the sampler
or guider. Existing conditioning, latents and sampler settings remain connected
as usual. No special sampler is required. Restart ComfyUI to load the new node.

## Controls

* `spatial_scale`: default **1.0**, an exact native bypass with no wrapper.
  Values below one compress spatial coordinates around 16; above one expand them:
  `p_new = 16 + spatial_scale * (p_native - 16)` for visual height/width only.
  Start with **0.90**, **0.85**, **1.10** against **1.0**.
* `mode`: **static** applies the scale on every call. **smooth_ramp** starts native,
  uses smoothstep between `ramp_start` (default 0.4) and `ramp_end` (0.9), then holds
  the chosen scale. Establish whether static scaling helps before trying ramps.
* Ramp progress is measured within the sampler's supplied `sample_sigmas`, from
  its first to last nonterminal denoiser sigma. Extra intermediate evaluations
  interpolate between entries. A partial denoise run uses its own supplied range.
  A single denoiser entry has progress zero. Missing schedules raise an error in
  ramp mode; static mode requires no schedule. Require start < end, both in [0,1].
* `reference_scope`: **all_visual** scales the target, keyframes/guide clips and
  image/video references together. **target_and_guides** keeps reference rows native
  for a separate comparison. Both preserve text, all temporal coordinates and all
  audio coordinates (including stereo spatial markers).

Lower scale imitates closer neighbour spacing, but also shrinks the positional
canvas. It does not give the model extra tokens or faithfully simulate 768p.

## Compatibility and isolation

Uses a native DIFFUSION_MODEL wrapper, passing a copied layout/position tensor to
core before its normal RoPE calculation. Cached layouts are never modified, there
is no global patch or persistent GPU cache, and a clean model branch stays native.
Context-window layouts retain their absolute temporal offsets. The patch handles
core's spatial padding and its missing/stale-layout fallback.

Native MiniMax H3 AV models with 1x2x2 patches are supported. Motion Memory is
explicitly rejected in either node order because its independently constructed
source RoPE is not scaled. Other external patches that create their own positions
need separate compatibility checks. Duplicate spatial patches are rejected.

## Repeatable test

1. Choose a short clip and keep checkpoint, LoRAs, prompt, references, dimensions,
   duration, seed, sampler, steps, CFG and sigma shifts fixed.
2. Run static scales 1.0, 0.90, 0.85 and 1.10. Repeat with several seeds.
3. Compare both full frames and crops resized to the same face/subject size.
   Check texture, framing, motion, flicker and audio sync. Enlarged faces alone
   are not evidence of improved detail at a fixed subject scale.
4. If a static scale helps, compare it with a smooth ramp to that scale.

CPU tests use native ComfyUI layouts, patchers and a tiny randomly initialized H3
model. They verify coordinate isolation, ramps, output shape/finite values and
that active scaling changes the computation. They do not establish visual quality
or GPU performance.
