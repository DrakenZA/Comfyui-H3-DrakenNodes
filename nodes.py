"""ComfyUI nodes: H3 Context Windows, H3 Long AV Latent, H3 Audio Lock, H3 Window Plan, H3 Trim Prefix AV."""

import logging

import torch

import comfy.context_windows
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils

from . import h3_grid as G
from .handler import H3ContextHandler, make_prepare_sampling_wrapper

LOG = logging.getLogger("h3_context_windows")
FUSE_METHODS = ["pyramid", "relative", "flat", "overlap-linear"]
MAX_RES = 16384


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _audio_to_vae_rate(audio_vae, audio):
    import torchaudio
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return waveform, vae_sr


def _encode_audio_tail(audio_vae, audio, seconds):
    """Encode the last `seconds` of an AUDIO dict with the H3 audio VAE -> [1, 32, 2, T]."""
    waveform, vae_sr = _audio_to_vae_rate(audio_vae, audio)
    n = int(round(seconds * vae_sr))
    if waveform.shape[-1] < n:
        raise ValueError(f"prefix audio is {waveform.shape[-1] / vae_sr:.2f}s, shorter than the {seconds:.2f}s prefix")
    waveform = waveform[..., -n:]
    return audio_vae.encode(waveform[:1].movedim(1, -1))


class H3ContextWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "window_frames": ("INT", {"default": 141, "min": 39, "max": 362, "step": 17,
                                          "tooltip": "Frames per window, snapped to 17k+5. Audio-exact lengths: 39, 90, 141, 192, 243, 294, 345."}),
                "stride_frames": ("INT", {"default": 51, "min": 17, "max": 345, "step": 17,
                                          "tooltip": "Frames between window starts (window - overlap), snapped to 17n. Multiples of 51 keep the audio grid exact."}),
                "fuse_method": (FUSE_METHODS, {"default": "pyramid"}),
                "absolute_positions": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Place each window at its real position on the clip's time axis (RoPE) instead of the clip origin. Off = every window renders 'the opening of the shot' and seams flicker."}),
                "phase_alternate": ("BOOLEAN", {"default": False,
                                                "tooltip": "EXPERIMENTAL: shift interior window starts by half a stride on odd steps so seams never sit on the same tokens. Moving windows per step was measured as harmful without absolute positions; untested with them."}),
                "freenoise": ("BOOLEAN", {"default": True, "tooltip": "FreeNoise shuffling of the initial noise for smoother window blending."}),
                "split_conds_to_windows": ("BOOLEAN", {"default": False,
                                                       "tooltip": "With several positive conditionings (Conditioning Combine), each window uses the one for its position on the timeline."}),
            },
            "optional": {
                "expected_total_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 17,
                                                  "tooltip": "Length of the long latent, only used to budget VRAM per window instead of per full latent. 0 = off."}),
                "probe_seams": ("BOOLEAN", {"default": False,
                                            "tooltip": "Log, every step, how much neighbouring windows disagree in their overlaps (relative). Cheap; tells you whether a seam problem is blending or content."}),
                "halo_frames": ("INT", {"default": 0, "min": 0, "max": 85, "step": 17,
                                        "tooltip": "EXPERIMENTAL halo context: from halo_start onward, each window also sees the fused prediction of this many frames just before and after it (the neighbours' belief) as conditioning rows at their absolute time. 0 = off. Costs extra tokens per window."}),
                "halo_start_percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                                 "tooltip": "Fraction of the steps after which the halo is injected. NOTE: measured on real renders 2026-09-11, the halo made output look strange; kept for experiments only."}),
                "margin_frames": ("INT", {"default": 0, "min": 0, "max": 85, "step": 17,
                                          "tooltip": "Margins: each window is evaluated over this many extra frames on each side (real tokens of the shared latent at the current noise level) but only its inner span is fused, so edge tokens that saw one-sided context are never used. 0 = off. Costs extra tokens per window."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = "DrakenNodes/H3"
    DESCRIPTION = "Joint context-window sampling for MiniMax H3: one long AV latent, overlapping phase-aligned windows denoised together every step."

    def apply(self, model, window_frames, stride_frames, fuse_method, absolute_positions, phase_alternate, freenoise,
              split_conds_to_windows, expected_total_frames=0, probe_seams=False, halo_frames=0, halo_start_percent=0.3,
              margin_frames=0):
        L = G.snap_window_tokens(window_frames)
        S = G.snap_stride_tokens(stride_frames)
        if S >= L:
            raise ValueError(f"stride ({S} tokens) must be smaller than the window ({L} tokens); lower stride_frames or raise window_frames")
        halo_tokens = (max(0, int(halo_frames)) // G.FRAMES_PER_CYCLE) * G.TOKENS_PER_CYCLE
        margin_tokens = (max(0, int(margin_frames)) // G.FRAMES_PER_CYCLE) * G.TOKENS_PER_CYCLE
        handler = H3ContextHandler(L, S, fuse_method=fuse_method, phase_alternate=phase_alternate,
                                   freenoise=freenoise, split_conds_to_windows=split_conds_to_windows,
                                   absolute_positions=absolute_positions, probe_seams=probe_seams,
                                   halo_tokens=halo_tokens, halo_start_percent=halo_start_percent,
                                   margin_tokens=margin_tokens)
        m = model.clone()
        m.model_options["context_handler"] = handler
        if expected_total_frames and expected_total_frames > 0:
            total_tokens = G.frames_to_tokens(G.align_frames_up(expected_total_frames))
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                   "H3ContextWindows_prepare_sampling",
                                   make_prepare_sampling_wrapper(handler, total_tokens))
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(m)
        wf = G.tokens_to_frames(L)
        sf = (S // G.TOKENS_PER_CYCLE) * G.FRAMES_PER_CYCLE
        info = (f"window {wf} frames ({wf / G.FPS:.2f}s, {L} tokens), stride {sf} frames ({S} tokens), "
                f"overlap {wf - sf} frames; absolute_positions={'on' if absolute_positions else 'off'}; "
                f"phase_alternate={'on' if handler.phase_alternate else 'off'}")
        if halo_tokens:
            info += f"; halo {halo_tokens // 5 * 17} frames each side from {halo_start_percent:.0%} of the steps"
        if margin_tokens:
            info += f"; margins {margin_tokens // 5 * 17} frames each side (evaluated, not fused)"
        if probe_seams:
            info += "; seam probe on (see console log)"
        if not G.is_av_exact(wf):
            info += "; window audio length rounds to the nearest tick (pick 39/90/141/192/243/294/345 for exact)"
        if S % G.AV_EXACT_TOKEN_STRIDE:
            info += "; stride is not a multiple of 51 frames, interior audio windows round by <=17ms"
        LOG.info("H3 context windows: %s", info)
        return (m, info)


class H3LongAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1344, "min": 32, "max": MAX_RES, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": MAX_RES, "step": 32}),
                "length_frames": ("INT", {"default": 719, "min": 39, "max": 100000, "step": 17,
                                          "tooltip": "Total frames at 24 fps (719 = ~30s). Snapped up to 17k+5; with audio_exact also to an integer audio tick count (51n-12)."}),
                "audio_exact": ("BOOLEAN", {"default": True}),
                "prefix_context_frames": ("INT", {"default": 39, "min": 5, "max": 345, "step": 17,
                                                  "tooltip": "How many trailing frames of the footage to keep as the hard prefix (snapped to 17k+5; 39 keeps audio exact)."}),
                "feather_tokens": ("INT", {"default": 0, "min": 0, "max": 50,
                                           "tooltip": "Latent tokens after the hard prefix that are only partially preserved (soft handoff)."}),
            },
            "optional": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prefix_frames": ("IMAGE", {"tooltip": "Existing footage to extend; its last prefix_context_frames are written into the start of the long latent."}),
                "prefix_audio": ("AUDIO", {"tooltip": "Soundtrack of the footage (same tail is used)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "prefix_frames_used", "info")
    FUNCTION = "build"
    CATEGORY = "DrakenNodes/H3"
    DESCRIPTION = "Long MiniMax H3 AV latent for context-window sampling, optionally seeded with the tail of existing footage as a hard (masked) prefix."

    def build(self, width, height, length_frames, audio_exact, prefix_context_frames, feather_tokens,
              vae=None, audio_vae=None, prefix_frames=None, prefix_audio=None):
        total = G.align_frames_av_exact_up(length_frames) if audio_exact else G.align_frames_up(length_frames)
        T = G.frames_to_tokens(total)
        Ta = G.audio_ticks_for_frames(total)
        lat_h, lat_w = height // 16, width // 16
        dev = comfy.model_management.intermediate_device()
        video = torch.zeros([1, 24, T, lat_h, lat_w], device=dev)
        audio = torch.zeros([1, 32, 2, Ta], device=dev)
        vmask = torch.ones([1, 1, T, lat_h, lat_w], device=dev)
        amask = torch.ones([1, 1, 2, Ta], device=dev)
        info = [f"{total} frames ({total / G.FPS:.2f}s) = {T} video tokens, {Ta} audio ticks, {width}x{height}"]
        if width % 32 or height % 32:
            info.append(f"WARNING: {width}x{height} is not on the 32 px grid (latent {lat_w}x{lat_h} has an odd side). "
                        "H3 pads the target internally, but stock keyframe/guide nodes encode at this size and fail "
                        "in the DiT; use a multiple of 32 (e.g. {}x{}).".format((width + 31) // 32 * 32, (height + 31) // 32 * 32))
            LOG.warning(info[-1])
        used = 0

        if prefix_frames is not None:
            if vae is None:
                raise ValueError("prefix_frames needs the video VAE")
            avail = int(prefix_frames.shape[0])
            ctx = G.align_frames_nearest(prefix_context_frames)
            while ctx > avail and ctx > 5:
                ctx -= G.FRAMES_PER_CYCLE
            if ctx > avail:
                raise ValueError(f"footage has {avail} frames, need at least 5")
            tok = G.frames_to_tokens(ctx)
            if tok >= T:
                raise ValueError("prefix covers the whole latent; raise length_frames")
            frames = _resize(prefix_frames[-ctx:], width, height, "center")
            enc = vae.encode(frames).to(device=dev, dtype=video.dtype)
            if tuple(enc.shape[2:]) != (tok, lat_h, lat_w):
                raise ValueError(f"video VAE gave {tuple(enc.shape)}, expected [1,24,{tok},{lat_h},{lat_w}]")
            video[:, :, :tok] = enc[:1]
            vmask[:, :, :tok] = 0.0
            for k in range(min(feather_tokens, T - tok)):
                vmask[:, :, tok + k] = (k + 1) / (feather_tokens + 1)
            used = ctx
            info.append(f"video prefix: last {ctx} frames of footage -> tokens [0,{tok}) hard"
                        + (f", {feather_tokens} feathered tokens" if feather_tokens else ""))
            if prefix_audio is not None:
                if audio_vae is None:
                    raise ValueError("prefix_audio needs the audio VAE")
                ticks = G.audio_ticks_for_frames(ctx)
                aenc = _encode_audio_tail(audio_vae, prefix_audio, ctx / G.FPS).to(device=dev, dtype=audio.dtype)
                n = min(ticks, int(aenc.shape[-1]))
                audio[..., :n] = aenc[:1, ..., -n:]
                amask[..., :n] = 0.0
                af = int(round(feather_tokens * G.FRAME_PER_TOKEN[1] * G.AUDIO_HZ / G.FPS))
                for k in range(min(af, Ta - n)):
                    amask[..., n + k] = (k + 1) / (af + 1)
                info.append(f"audio prefix: {n} ticks hard" + (f", {af} feathered" if af else ""))
                if not G.is_av_exact(ctx):
                    info.append("note: prefix length is not audio-exact; audio prefix rounded to the nearest tick")
        elif prefix_audio is not None:
            info.append("prefix_audio ignored: no prefix_frames given")

        latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
        if used:
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor((vmask, amask))
        return (latent, used, "\n".join(info))


class H3AudioLock:
    """Pin a real soundtrack into the long AV latent so only video is generated.

    Ported from wordbrew/ComfyUI-H3-Toolkit's H3AudioLock (MIT): with one long latent the whole track is locked
    in one go, no per-link slicing. Video and audio share one packed sequence, so clamping audio every step makes
    the picture answer to it (lip sync, beats). Keeps any existing video mask (e.g. a footage prefix).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "audio_vae": ("VAE",),
            "audio": ("AUDIO",),
            "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                         "tooltip": "Where in the track the latent starts."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "1.0 pins the audio exactly; lower lets the model reinterpret it."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "lock"
    CATEGORY = "DrakenNodes/H3"

    def lock(self, latent, audio_vae, audio, offset_seconds, strength):
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("H3 Audio Lock needs an H3 AV latent")
        video, aud = samples.unbind()
        target_t = int(aud.shape[-1])
        waveform, vae_sr = _audio_to_vae_rate(audio_vae, audio)
        start_s = int(round(float(offset_seconds) * vae_sr))
        need_s = int(round(target_t / G.AUDIO_HZ * vae_sr))
        chunk = waveform[..., start_s:start_s + need_s]
        if chunk.shape[-1] < need_s:
            chunk = torch.cat([chunk, chunk.new_zeros(chunk.shape[:-1] + (need_s - chunk.shape[-1],))], dim=-1)
        z = audio_vae.encode(chunk[:1].movedim(1, -1))
        z = z[..., :target_t]
        if z.shape[-1] < target_t:
            pad = target_t - z.shape[-1]
            z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], pad)], dim=-1)
        z = z.to(aud.device, aud.dtype)
        if "noise_mask" in latent and getattr(latent["noise_mask"], "is_nested", False):
            mask_v = latent["noise_mask"].unbind()[0]
        else:
            mask_v = torch.ones([1, 1, video.shape[2], video.shape[3], video.shape[4]], device=video.device)
        mask_a = torch.full([1, 1, 2, target_t], 1.0 - float(strength), device=aud.device)
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        covered = min(need_s, max(0, waveform.shape[-1] - start_s)) / vae_sr
        info = (f"audio locked: {target_t} ticks ({target_t / G.AUDIO_HZ:.2f}s) from {offset_seconds:.2f}s, "
                f"strength {strength:.2f}; source covers {covered:.2f}s"
                + ("" if covered >= target_t / G.AUDIO_HZ - 0.05 else " (padded past the end of the track)"))
        return (out, info)


class H3WindowPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["length -> windows", "windows -> length"], {"default": "length -> windows"}),
            "length_frames": ("INT", {"default": 719, "min": 39, "max": 100000, "step": 17,
                                      "tooltip": "For 'length -> windows'."}),
            "windows": ("INT", {"default": 6, "min": 1, "max": 200, "tooltip": "For 'windows -> length': how many windows per step you want."}),
            "window_frames": ("INT", {"default": 141, "min": 39, "max": 362, "step": 17}),
            "stride_frames": ("INT", {"default": 51, "min": 17, "max": 345, "step": 17}),
        }}

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("plan", "windows_per_step", "length_frames")
    FUNCTION = "plan"
    CATEGORY = "DrakenNodes/H3"
    OUTPUT_NODE = True

    def plan(self, mode, length_frames, windows, window_frames, stride_frames):
        L = G.snap_window_tokens(window_frames)
        S = G.snap_stride_tokens(stride_frames)
        if mode == "windows -> length":
            # n windows with no clamped last window: total = L + (n-1)*S tokens, then up to audio-exact
            total_tokens = L + (max(1, int(windows)) - 1) * S
            length = G.align_frames_av_exact_up(G.tokens_to_frames(total_tokens))
        else:
            length = G.align_frames_up(length_frames)
        text = G.describe_plan(length, window_frames, stride_frames)
        n = len(G.plan_window_starts(G.frames_to_tokens(length), L, S))
        if mode == "windows -> length":
            text = f"length for {int(windows)} clean windows: {length} frames ({length / G.FPS:.2f}s)\n" + text
        return {"ui": {"text": [text]}, "result": (text, n, length)}


class H3TrimPrefixAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prefix_frames": ("INT", {"default": 0, "min": 0, "max": 100000})},
            "optional": {"images": ("IMAGE",), "audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "trim"
    CATEGORY = "DrakenNodes/H3"
    DESCRIPTION = "Drop the footage prefix (frames and the matching seconds of audio) from a decoded H3 result."

    def trim(self, prefix_frames, images=None, audio=None):
        if images is not None and prefix_frames > 0:
            images = images[prefix_frames:]
        if audio is not None and prefix_frames > 0:
            sr = int(audio["sample_rate"])
            n = int(round(prefix_frames / G.FPS * sr))
            audio = {"waveform": audio["waveform"][..., n:], "sample_rate": sr}
        return (images, audio)


NODE_CLASS_MAPPINGS = {
    "DrakenH3ContextWindows": H3ContextWindows,
    "DrakenH3LongAVLatent": H3LongAVLatent,
    "DrakenH3AudioLock": H3AudioLock,
    "DrakenH3WindowPlan": H3WindowPlan,
    "DrakenH3TrimPrefixAV": H3TrimPrefixAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DrakenH3ContextWindows": "H3 Context Windows (Draken)",
    "DrakenH3LongAVLatent": "H3 Long AV Latent, footage prefix (Draken)",
    "DrakenH3AudioLock": "H3 Audio Lock, long latent (Draken)",
    "DrakenH3WindowPlan": "H3 Window Plan (Draken)",
    "DrakenH3TrimPrefixAV": "H3 Trim Prefix, image+audio (Draken)",
}
