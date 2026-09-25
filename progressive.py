"""Multi-stage progressive-resolution sampling (SelfLift-zero, generalised).

Based on SelfLift (arXiv:2609.02036) and facok/comfyui-SelfLift. That node runs the
first steps at one low resolution, lifts the clean endpoint to full resolution
with the Artifact-Aware Consistency Lift, re-noises it and finishes there. It only
accepts plain Euler, because it rebuilds the last low-res Euler step by hand.

Two generalisations here:

* Any sampler. Each stage is an ordinary comfy.samplers.sample() call over its
  slice of the schedule. At a boundary the video stream is rebuilt from the last
  clean prediction x0 (the sampler's callback) lifted to the next grid and noised
  at the boundary sigma. With Euler this is exactly SelfLift's transition: an
  Euler step from (1-s_k) z0 + s_k n to s_next lands on (1-s_next) z0 + s_next n.
  Audio streams (no spatial dims) carry the sampler's own state across the
  boundary. Multistep samplers restart their history at each stage.
* Any number of stages, each with its own step count and spatial scale, the
  last one always at full resolution. Every boundary runs the same lift.
"""

import logging
import sys

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils

LOG = logging.getLogger("h3_progressive")
MAX_STAGES = 8


# ---------------------------------------------------------------- lift (Eqs. 4-9)

def _upsample_latent(z, out_hw, mode):
    H, W = out_hw
    if z.ndim == 4:
        return F.interpolate(z.float(), size=(H, W), mode="nearest" if mode == "nearest" else "bilinear")
    return F.interpolate(z.float(), size=(z.shape[2], H, W), mode="nearest" if mode == "nearest" else "trilinear")


def _pixel_anchor(z0, vae, out_hw):
    """Decode at the current grid, upscale in pixels, re-encode at the next grid (Eq. 5)."""
    H, W = out_hw
    if z0.ndim == 4:
        img = vae.decode(z0)  # [B, h*r, w*r, 3]
        ratio = img.shape[1] // z0.shape[-2]
        up = comfy.utils.common_upscale(img.movedim(-1, 1), W * ratio, H * ratio, "lanczos", "disabled").movedim(1, -1)
        del img
        return vae.encode(up).float()
    if z0.shape[0] > 1:  # one video per batch item
        return torch.cat([_pixel_anchor(s, vae, out_hw) for s in z0.split(1)], dim=0)
    frames = vae.decode(z0)
    if frames.ndim == 5:  # frames-as-batch
        frames = frames.reshape(-1, *frames.shape[-3:])
    ratio = frames.shape[1] // z0.shape[-2]
    Hp, Wp = H * ratio, W * ratio
    dev = getattr(vae, "device", torch.device("cpu"))
    dt = getattr(vae, "vae_dtype", torch.float32)
    if dt not in (torch.float16, torch.bfloat16, torch.float32) or torch.device(dev).type == "cpu":
        dt = torch.float32
    # chunked upscale into one buffer: a whole-clip fp32 lanczos pass can take tens of GB of RAM
    up = torch.empty((frames.shape[0], Hp, Wp, frames.shape[-1]), dtype=dt)
    for i in range(0, frames.shape[0], 32):
        chunk = frames[i:i + 32].movedim(-1, 1).to(device=dev, dtype=dt)
        up[i:i + 32] = F.interpolate(chunk, size=(Hp, Wp), mode="bicubic", antialias=True).movedim(1, -1).to(up.device)
        del chunk
    del frames
    return vae.encode(up).float()


def paired_lifts(z0, vae, out_hw, mode="nearest", lifter=None, need_lat=True, need_pix=True):
    """(direct lift, pixel-VAE anchor) of a VAE-space clean latent at out_hw; None for skipped branches."""
    z_lat = z_pix = None
    if need_lat:
        z_lat = lifter(z0, out_hw) if (lifter is not None and z0.ndim == 5) else _upsample_latent(z0, out_hw, mode)
    if need_pix:
        z_pix = _pixel_anchor(z0, vae, out_hw)
        if z_lat is not None:
            z_lat = z_lat.to(z_pix.device)
    return z_lat, z_pix


def consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=None):
    """Correct the top-rho most inconsistent locations of the direct lift toward the pixel anchor.

    mask: optional [B, 1, (T,) H, W] generate region; statistics and the correction stay inside it.
    """
    if rho <= 0.0 or w_max <= 0.0:
        return z_lat
    if mask is None and rho >= 1.0 and w_min >= 1.0:
        return z_pix
    delta = z_pix - z_lat
    s = delta.abs().mean(dim=1)  # per-location inconsistency [B, (T,) H, W]
    view = (-1,) + (1,) * (s.ndim - 1)
    if mask is None:
        thr = torch.quantile(s.flatten(1), 1.0 - rho, dim=1).view(view)
        selected = s >= thr
    else:
        m = mask.squeeze(1).to(s.device)
        s = s * m
        region = (m > 0).expand_as(s)
        thr = torch.stack([torch.quantile(s[b][region[b]], 1.0 - rho) if region[b].any()
                           else torch.full((), float("inf"), device=s.device, dtype=s.dtype)
                           for b in range(s.shape[0])]).view(view)
        selected = (s >= thr) & region
    if not selected.any():
        return z_lat
    s_min = s.masked_fill(~selected, float("inf")).flatten(1).amin(dim=1).view(view)
    s_max = s.masked_fill(~selected, float("-inf")).flatten(1).amax(dim=1).view(view)
    w = w_min + (w_max - w_min) * (s - s_min) / (s_max - s_min + 1e-8)
    w = torch.where(selected, w, torch.zeros_like(w)).unsqueeze(1)
    return z_lat + w * delta


# ---------------------------------------------------------------- helpers

def _streams(x):
    return (list(x.unbind()), True) if getattr(x, "is_nested", False) else ([x], False)


def _pack(streams, nested):
    return comfy.nested_tensor.NestedTensor(streams) if nested else streams[0]


def _resize(x, size, mode="bilinear"):
    """Spatial resize of a [B, C, H, W] or [B, C, T, H, W] tensor; never mixes frames."""
    h, w = size
    if x.shape[-2:] == (h, w):
        return x
    if x.ndim == 5:
        b, c, t = x.shape[:3]
        y = F.interpolate(x.float().permute(0, 2, 1, 3, 4).reshape(b * t, c, *x.shape[-2:]), size=(h, w),
                          mode=mode, align_corners=False if mode == "bilinear" else None)
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return F.interpolate(x.float(), size=(h, w), mode=mode, align_corners=False if mode == "bilinear" else None)


def _resize_keyframes(cond, h, w):
    """MiniMax H3 keyframe latents share the generation grid; follow the stage grid (mean-matched)."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and tuple(lat.shape[-2:]) != (h, w):
                r = _resize(lat, (h, w))
                kf["latent"] = (r + (lat.float().mean(dim=(-2, -1), keepdim=True) - r.mean(dim=(-2, -1), keepdim=True))).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _mask_hook(anchor, mask):
    """post-CFG: pin x0 to the original content outside the generate region."""
    def fn(args):
        d = args["denoised"]
        m = mask.to(device=d.device, dtype=d.dtype)
        return d * m + anchor.to(device=d.device, dtype=d.dtype) * (1.0 - m)
    return fn


def normalize_mask(latent_image, video_shape):
    """noise_mask -> [B, 1, H, W] (image) or [B, 1, T, H, W] (video), or None. Resized to the latent grid."""
    mask = latent_image.get("noise_mask")
    if mask is None:
        return None
    if getattr(mask, "is_nested", False):
        mask = mask.unbind()[0]  # video stream mask; audio is always generated
    video = len(video_shape) == 5
    b, H, W = video_shape[0], video_shape[-2], video_shape[-1]
    if mask.ndim == 3:
        mask = mask[:, None]
    if video and mask.ndim == 4:
        mask = mask[:, :, None]
    if mask.ndim != (5 if video else 4) or mask.shape[1] != 1:
        raise ValueError("progressive sampler: noise_mask must be [B, H, W], [B, 1, H, W] or [B, 1, T, H, W]")
    if mask.shape[0] != b:
        if mask.shape[0] != 1:
            raise ValueError(f"progressive sampler: noise_mask batch {mask.shape[0]} does not match latent batch {b}")
        mask = mask.expand(b, *mask.shape[1:])
    if tuple(mask.shape[-2:]) != (H, W):
        lead = mask.shape[:-2]
        mask = F.interpolate(mask.reshape(-1, 1, *mask.shape[-2:]).float(), size=(H, W), mode="bilinear",
                             align_corners=False).reshape(*lead, H, W)
    if video:
        if mask.shape[2] not in (1, video_shape[2]):
            raise ValueError(f"progressive sampler: noise_mask has {mask.shape[2]} frames, latent has {video_shape[2]}")
        mask = mask.expand(b, 1, video_shape[2], H, W)
    return mask.float().clamp(0.0, 1.0)


def plan_stages(total_steps, stages, H, W):
    """stages: [(steps, scale), ...] for all but the last stage. Returns [(start, end, h, w), ...] incl. the final one."""
    plan, start, prev = [], 0, 0.0
    for i, (steps, scale) in enumerate(stages, 1):
        if steps < 1:
            raise ValueError(f"progressive sampler: stage {i} needs at least 1 step")
        if not 0.1 <= scale < 1.0:
            raise ValueError(f"progressive sampler: stage {i} scale {scale} must be in [0.1, 1)")
        if scale <= prev:
            raise ValueError(f"progressive sampler: stage scales must increase (stage {i}: {scale} <= {prev})")
        h, w = max(2, round(H * scale / 2) * 2), max(2, round(W * scale / 2) * 2)
        if plan and (h, w) == plan[-1][2:]:
            raise ValueError(f"progressive sampler: stages {i - 1} and {i} round to the same {h}x{w} latent grid")
        if (h, w) == (H, W):
            raise ValueError(f"progressive sampler: stage {i} scale {scale} rounds to the full {H}x{W} grid")
        plan.append((start, start + steps, h, w))
        start, prev = start + steps, scale
    if start >= total_steps:
        raise ValueError(f"progressive sampler: the stages use {start} of {total_steps} steps; "
                         "the full-resolution stage needs at least 1")
    plan.append((start, total_steps, H, W))
    return plan


def _validate_schedule(sigmas, plan):
    if sigmas.ndim != 1 or not sigmas.is_floating_point() or not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("progressive sampler: sigmas must be a finite, nonnegative 1-D float tensor")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("progressive sampler: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("progressive sampler: only the final sigma may be zero")
    for _, end, _, _ in plan[:-1]:
        if sigmas[end] >= 1:
            raise ValueError(f"progressive sampler: stage boundary sigma at step {end} must be < 1")


# ---------------------------------------------------------------- engine

def progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg, stages,
                       rho=0.0, w_min=0.5, w_max=1.0, latent_upsample="nearest", lifter=None, model_hires=None):
    """stages: [(steps, scale), ...] for every stage before the full-resolution one."""
    if not 0.0 <= rho <= 1.0 or not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("progressive sampler: need 0 <= rho <= 1 and 0 <= w_min <= w_max <= 1")
    total = sigmas.shape[-1] - 1
    if total < 1:
        return latent_image
    model_sampling = model.get_model_object("model_sampling")
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("progressive sampler: requires a rectified-flow (CONST) model such as MiniMax H3")

    samples = comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None))
    streams, nested = _streams(samples)
    if streams[0].ndim not in (4, 5):
        raise ValueError("progressive sampler: expected a 4D image or 5D video latent")
    video = streams[0].ndim == 5
    H, W = streams[0].shape[-2:]
    b, c = streams[0].shape[:2]
    plan = plan_stages(total, stages, H, W)
    _validate_schedule(sigmas, plan)
    mask_full = normalize_mask(latent_image, tuple(streams[0].shape))

    device = comfy.model_management.intermediate_device()
    source = streams[0].to(device)
    audio = [s.to(device) for s in streams[1:]]
    latent_format = model.get_model_object("latent_format")
    batch_index = latent_image.get("batch_index", None)
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0)
    LOG.info("[progressive plan] stages=%s sigmas_at_boundaries=%s rho=%.3f weights=(%.3f, %.3f) direct_lift=%s "
             "pixel_anchor=%s mask=%s hires_model=%s",
             [(s, e, (h, w)) for s, e, h, w in plan], [round(sigmas[e].item(), 6) for _, e, _, _ in plan[:-1]],
             rho, w_min, w_max, "skipped" if not need_lat else ("learned" if lifter is not None and video else latent_upsample),
             need_pix, None if mask_full is None else tuple(mask_full.shape), model_hires is not None)

    def stage_model(base, anchor, m):
        if m is None:
            return base
        patched = base.clone()
        if nested:  # post-CFG sees the flat pack: video (channel-major) then audio streams
            flat_m = m.reshape(b, 1, -1).repeat(1, 1, c)
            flat_a = anchor.reshape(b, 1, -1)
            for s in audio:
                flat_m = torch.cat([flat_m, torch.ones_like(s.reshape(b, 1, -1))], dim=-1)
                flat_a = torch.cat([flat_a, s.reshape(b, 1, -1)], dim=-1)
            hook = _mask_hook(flat_a, flat_m)
        else:
            hook = _mask_hook(anchor, m)
        patched.model_options = comfy.model_patcher.set_model_options_post_cfg_function(patched.model_options, hook)
        return patched

    callback = None
    try:
        import latent_preview
        callback = latent_preview.prepare_callback(model, total)
    except Exception:  # no preview machinery (tests)
        pass
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # stage 1 input: the template (and its content) at the first grid; later stages resume from `current`
    h0, w0 = plan[0][2:]
    current = None  # (latent in the sampler's output space, noise) for the next stage
    first = _pack([_resize(source, (h0, w0)).to(source.dtype)] + audio, nested)
    current = (first, comfy.sample.prepare_noise(first, seed, batch_index))

    out = None
    for idx, (start, end, h, w) in enumerate(plan):
        last = idx == len(plan) - 1
        anchor = _resize(source, (h, w)).to(source.dtype)
        m = None if mask_full is None else _resize(mask_full, (h, w)).clamp(0.0, 1.0)
        base = model_hires if (last and model_hires is not None) else model
        smodel = stage_model(base, anchor, m)
        pos, neg = (_resize_keyframes(positive, h, w), _resize_keyframes(negative, h, w)) if video and (h, w) != (H, W) \
            else (positive, negative)
        seen = {"n": 0, "x0": None}

        def cb(step, x0, x, _total, start=start, h=h, w=w, seen=seen):
            seen["n"] += 1
            seen["x0"] = x0
            if callback is None:
                return None
            preview = x0
            if (h, w) != (H, W):  # previewers expect the target grid
                ps, pn = _streams(x0)
                preview = _pack([_resize(ps[0], (H, W)).to(ps[0].dtype)] + ps[1:], pn)
            return callback(min(start + seen["n"] - 1, total - 1), preview, preview, total)

        latent_in, noise_in = current
        LOG.info("[progressive] stage %d/%d steps %d-%d latent %dx%d", idx + 1, len(plan), start, end, h, w)
        out = comfy.samplers.sample(smodel, noise_in, pos, neg, cfg, smodel.load_device, sampler,
                                    sigmas[start:end + 1], smodel.model_options, latent_image=latent_in,
                                    callback=cb, disable_pbar=disable_pbar, seed=seed)
        if last:
            break
        if seen["x0"] is None:
            raise RuntimeError(f"progressive sampler: stage {idx + 1} reported no x0 through the sampler callback; "
                               "this sampler cannot be used for a lifted stage")

        # ---- transition to the next grid
        nh, nw = plan[idx + 1][2:]
        sigma = sigmas[end]
        x0_streams, _ = _streams(seen.pop("x0"))
        z0 = latent_format.process_out(x0_streams[0].float()).to(device)
        del x0_streams
        z_lat, z_pix = paired_lifts(z0, vae, (nh, nw), latent_upsample, lifter, need_lat=need_lat, need_pix=need_pix)
        z_lat = latent_format.process_in(z_lat) if z_lat is not None else None
        z_pix = latent_format.process_in(z_pix) if z_pix is not None else None
        m_next = None if mask_full is None else _resize(mask_full, (nh, nw)).clamp(0.0, 1.0)
        z0_next = consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=m_next).to(device)
        if m_next is not None:  # keep region: the true original, not a lifted estimate
            mm = m_next.to(z0_next)
            z0_next = z0_next * mm + _resize(source, (nh, nw)).to(z0_next) * (1.0 - mm)
        del z0, z_lat, z_pix

        # re-noise at the boundary sigma (== SelfLift's re-noise at s_k + reused Euler step, see module doc)
        noise = comfy.sample.prepare_noise(z0_next, (seed + idx + 1) % (1 << 64), batch_index).to(z0_next)
        video_state = model_sampling.noise_scaling(sigma, noise, z0_next)
        del noise, z0_next
        # The sampler returned process_latent_out(x / (1 - sigma)); fed back with zero noise, the next
        # stage starts exactly at x. Audio streams keep the sampler's own state; the video stream
        # is replaced by the lifted one in the same space (video is never audio-scaled on H3).
        out_streams, _ = _streams(out)
        out_streams = [s.to(device) for s in out_streams]
        out_streams[0] = latent_format.process_out(model_sampling.inverse_noise_scaling(sigma, video_state))
        current = (_pack(out_streams, nested), _pack([torch.zeros_like(s) for s in out_streams], nested))
        del out_streams, video_state, out

    result = latent_image.copy()
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                               dtype=comfy.model_management.intermediate_dtype())
    return result


# ---------------------------------------------------------------- learned H3 lifter (optional, from comfyui-SelfLift)

def list_upscalers():
    try:
        import folder_paths
        return folder_paths.get_filename_list("latent_upscale_models")
    except Exception:
        return []


def selflift_upscaler_module():
    """h3_upscaler from an installed comfyui-SelfLift pack, or None."""
    for name, mod in list(sys.modules.items()):
        if name.endswith(".h3_upscaler") and hasattr(mod, "learned_latent_lift"):
            return mod
    return None


def learned_lifter(model_name, unload=True):
    mod = selflift_upscaler_module()
    if mod is None:
        raise RuntimeError("progressive sampler: the learned H3 upscaler needs comfyui-SelfLift installed "
                           "(https://github.com/facok/comfyui-SelfLift); set upscaler_model to none to lift without it")
    return lambda z, hw: mod.learned_latent_lift(z, hw, model_name, force_unload=unload)
