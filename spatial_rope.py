"""Experimental, branch-local spatial RoPE scaling for native MiniMax H3."""

import copy
import math

import torch

PATCH_KEY = "DrakenH3SpatialRoPE"
MODES = ("static", "smooth_ramp")
SCOPES = ("all_visual", "target_and_guides")


def sampling_progress(timestep, schedule):
    """Position within the supplied denoiser-call schedule, interpolating extra calls.

    Terminal zero is not a denoiser call. Partial schedules deliberately describe
    progress within that sampling run, rather than an imaginary full trajectory.
    """
    if not isinstance(schedule, torch.Tensor) or schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("H3 Spatial RoPE smooth ramp needs the sampler's sample_sigmas schedule.")
    if not torch.isfinite(schedule).all() or (schedule < 0).any() or (schedule[1:] > schedule[:-1]).any():
        raise ValueError("H3 Spatial RoPE needs finite, nonnegative, non-increasing sample_sigmas.")
    calls = schedule[:-1] if float(schedule[-1]) == 0 else schedule
    if calls.numel() == 1:
        return 0.0
    sigma = timestep.flatten()[0].to(calls) / 1000.0
    if not torch.isfinite(sigma):
        raise ValueError("H3 Spatial RoPE received a non-finite timestep.")
    upper = int(torch.searchsorted(-calls.contiguous(), -sigma).clamp(1, calls.numel() - 1))
    lower = upper - 1
    span = calls[lower] - calls[upper]
    fraction = float(((calls[lower] - sigma) / span.clamp_min(1e-12)).clamp(0, 1))
    return (lower + fraction) / (calls.numel() - 1)


def scale_layout(layout, scale, scope):
    """Copy only positions: cached layouts, row membership and time stay native."""
    if scale == 1.0:
        return layout
    out = copy.copy(layout)
    out.position_ids = layout.position_ids.clone()
    kinds = {"video", "cond"}
    if scope == "all_visual":
        kinds.add("ref_img")  # H3 uses this kind for both image and video refs.
    for a, b, kind in layout.segments:
        if kind in kinds:
            out.position_ids[a:b, 1:] = 16.0 + scale * (layout.position_ids[a:b, 1:] - 16.0)
    return out


class SpatialRoPEPatch:
    def __init__(self, spatial_scale, mode, ramp_start, ramp_end, reference_scope):
        self.spatial_scale = spatial_scale
        self.mode = mode
        self.ramp_start, self.ramp_end = ramp_start, ramp_end
        self.reference_scope = reference_scope

    def scale_at(self, timestep, options):
        if self.mode == "static":
            return self.spatial_scale
        progress = sampling_progress(timestep, options.get("sample_sigmas"))
        u = min(1.0, max(0.0, (progress - self.ramp_start) / (self.ramp_end - self.ramp_start)))
        weight = u * u * (3.0 - 2.0 * u)
        return 1.0 + weight * (self.spatial_scale - 1.0)

    def wrap(self, executor, x, timestep, context, transformer_options, **kwargs):
        wrappers = transformer_options.get("wrappers", {}).get("diffusion_model", {})
        if wrappers.get("DrakenH3MotionMemory"):
            raise ValueError("H3 Spatial RoPE and Motion Memory cannot be combined yet: source RoPE would be unaligned.")
        scale = self.scale_at(timestep, transformer_options)
        if scale == 1.0:
            return executor(x, timestep, context, transformer_options, **kwargs)
        from comfy.ldm.minimax.model import PackedLayout
        model = executor.class_obj
        # Match core padding without allocating a padded copy of the video.
        dims = tuple(((d + p - 1) // p) * p for d, p in zip(x[0].shape[2:], model.patch_size))
        signature = (context.shape[1], *dims, x[1].shape[-1])
        payload = dict(kwargs.get("minimax_payload") or {})
        layout = payload.get("layout")
        if layout is None or layout.signature != signature:
            layout = PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        payload["layout"] = scale_layout(layout, scale, self.reference_scope)
        # Core writes minimax_h3_layout into options. Keep that call-local too.
        return executor(x, timestep, context, dict(transformer_options),
                        **{**kwargs, "minimax_payload": payload})


def patch_model(model, spatial_scale=1.0, mode="static", ramp_start=0.4,
                ramp_end=0.9, reference_scope="all_visual"):
    if not math.isfinite(spatial_scale) or not 0.5 <= spatial_scale <= 1.5:
        raise ValueError("H3 Spatial RoPE spatial_scale must be finite and between 0.5 and 1.5.")
    if mode not in MODES or reference_scope not in SCOPES:
        raise ValueError("Unknown H3 Spatial RoPE mode or reference scope.")
    if not (math.isfinite(ramp_start) and math.isfinite(ramp_end) and 0 <= ramp_start < ramp_end <= 1):
        raise ValueError("H3 Spatial RoPE requires 0 <= ramp_start < ramp_end <= 1.")
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import WrappersMP
    diffusion = model.get_model_object("diffusion_model")
    if not isinstance(diffusion, MiniMaxH3Model) or tuple(diffusion.patch_size) != (1, 2, 2):
        raise ValueError("H3 Spatial RoPE requires a native MiniMax H3 model with patch size 1x2x2.")
    wrappers = getattr(model, "wrappers", {}).get(WrappersMP.DIFFUSION_MODEL, {})
    if wrappers.get(PATCH_KEY):
        raise ValueError("H3 Spatial RoPE is already applied to this model branch.")
    patched = model.clone()
    if spatial_scale == 1.0:
        return patched  # Exact native baseline: no wrapper or table allocation.
    if wrappers.get("DrakenH3MotionMemory"):
        raise ValueError("H3 Spatial RoPE and Motion Memory cannot be combined yet: source RoPE would be unaligned.")
    patch = SpatialRoPEPatch(spatial_scale, mode, ramp_start, ramp_end, reference_scope)
    patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, PATCH_KEY, patch.wrap)
    return patched


class H3SpatialRoPE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "spatial_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                "tooltip": "1 = native bypass. Try 0.90, 0.85 and 1.10. Changes spatial relationships; detail improvement is unproven."}),
            "mode": (list(MODES), {"default": "static"}),
            "ramp_start": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Smooth ramp only: sampling progress where scale begins leaving 1.0."}),
            "ramp_end": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Smooth ramp only: sampling progress where the chosen scale is reached and held."}),
            "reference_scope": (list(SCOPES), {"default": "all_visual",
                "tooltip": "all_visual scales target, guides, image/video references. target_and_guides leaves reference coordinates native. Audio/text/time coordinates stay native in both modes."}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = "DrakenNodes/H3"
    DESCRIPTION = "Experimental spatial RoPE scaling at unchanged resolution/token count. May alter framing, motion and audio through joint attention. No proven detail gain."

    def apply(self, model, spatial_scale=1.0, mode="static", ramp_start=0.4,
              ramp_end=0.9, reference_scope="all_visual"):
        patched = patch_model(model, spatial_scale, mode, ramp_start, ramp_end, reference_scope)
        info = ("H3 Spatial RoPE: native bypass (1.0)" if spatial_scale == 1.0 else
                f"H3 Spatial RoPE: {spatial_scale:.3f}x; {mode}; {reference_scope}")
        if spatial_scale != 1.0 and mode == "smooth_ramp":
            info += f"; ramp {ramp_start:.0%} to {ramp_end:.0%}, then hold"
        return patched, info
