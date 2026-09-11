"""Context-window handler for MiniMax H3 (joint multi-window denoising of one long AV latent).

Built on ComfyUI core's comfy.context_windows (the LTX-2 multimodal path) with what H3
needs and core does not provide:

  1. video-token -> audio-tick window mapping on H3's real grid
     (17-frame token cycle, 5/3 ticks per frame), with the audio latent's time
     axis on dim 3 ([B, 32, 2, T]) instead of dim 2,
  2. a per-window rewrite of the H3 payload: the global PackedLayout is dropped
     and rebuilt per window (cached), keyframes / guide clips are shifted into
     window-local frame indices and cropped to the window on token-cycle
     boundaries, guide audio is cropped to the window's ticks,
  3. each window's target and keyframe rows placed at their absolute position on
     the clip's RoPE time axis (origin placement makes every window render "the
     opening of the shot" and the seams flicker; measured by
     wordbrew/ComfyUI-H3-Toolkit, 2026-08-28),
  4. a phase-aligned window schedule (every window starts on a 5-token boundary
     and is a valid 5j+2 clip), optional per-step phase alternation, and no
     causal anchor frame (it would break the token phase),
  5. a seam probe that measures how much neighbouring windows disagree in their
     overlaps every step, and an experimental "halo context": after a chosen
     fraction of the steps, each window also sees the fused prediction of the
     frames just outside it (the neighbours' belief) as conditioning rows at
     their absolute time, so windows are pulled toward agreement during
     denoising instead of only being blended afterwards.

All state lives on this handler object, which lives in one cloned model's
model_options. Nothing in ComfyUI core or the shared H3 model object is patched,
so two model branches with different settings cannot affect each other.
"""

import logging
import math

import torch

import comfy.conds
import comfy.context_windows as cw
import comfy.ldm.common_dit
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.minimax.model import FRAME_RESCALE, PackedLayout

from . import h3_grid as G

VIDEO_DIM = 2
AUDIO_DIM = 3
LOG = logging.getLogger("h3_context_windows")
_LAYOUT_CACHE_MAX = 128


class H3WindowingState(cw.WindowingState):
    """Per-step state; derives the audio window for each video window on H3's grid."""

    def prepare_window(self, window: cw.IndexListContextWindow, model) -> cw.IndexListContextWindow:
        idx = list(window.index_list)
        s, L = idx[0], len(idx)
        if idx != list(range(s, s + L)):
            raise ValueError("H3 context windows must be contiguous (no strided or wrapped windows)")
        if not G.is_phase_aligned_window(s, L):
            raise ValueError(f"H3 window [{s}, {s + L}) is not phase aligned (start % 5 == 0, len % 5 == 2)")
        T = self.latents[0].shape[VIDEO_DIM]
        Ta = self.latents[1].shape[AUDIO_DIM]
        a0, a1 = G.window_audio_range(s, L, T, Ta)
        a_overlap = max(0, round(window.context_overlap * (a1 - a0) / L))
        audio_window = cw.IndexListContextWindow(list(range(a0, a1)), dim=AUDIO_DIM, total_frames=Ta,
                                                 context_overlap=a_overlap)
        out = cw.IndexListContextWindow(idx, dim=VIDEO_DIM, total_frames=T,
                                        modality_windows={1: audio_window}, context_overlap=window.context_overlap)
        out.h3_frame_start = G.token_frame_offset(s)
        out.h3_frame_stop = out.h3_frame_start + G.tokens_to_frames(L)
        inner = getattr(window, "h3_inner", None)
        if inner is not None:
            # margins: only the inner span [s+off, s+off+n) is fused; the rest is context for the model
            off, n = inner
            out.h3_inner = (off, n)
            ai0, ai1 = G.window_audio_range(s + off, n, T, Ta)
            ai0 = min(max(ai0, a0), a1)
            ai1 = min(max(ai1, ai0), a1)
            audio_window.h3_inner = (ai0 - a0, ai1 - ai0)
        return out


def _aligned_schedule(num_frames: int, handler: "H3ContextHandler", model_options: dict):
    shift = 0
    if handler.phase_alternate and (handler._step % 2 == 1):
        shift = handler.phase_shift_tokens
    return [list(range(s, s + handler.window_tokens))
            for s in G.plan_window_starts(num_frames, handler.window_tokens, handler.stride_tokens, shift)]


def _extend_with_margins(inner_windows, num_frames: int, margin: int):
    """Each inner window [s, s+L) becomes [s-M, s+L+M) clipped to the clip; the inner span is kept
    as (offset, length) so only it is fused. Phase stays valid: M is a multiple of 5 and the ends
    clip to 0 / to the 5k+2 total."""
    out = []
    for idx in inner_windows:
        s, L = idx[0], len(idx)
        e0 = max(0, s - margin)
        e1 = min(num_frames, s + L + margin)
        out.append((list(range(e0, e1)), s - e0, L))
    return out


class H3ContextHandler(cw.IndexListContextHandler):
    def __init__(self, window_tokens: int, stride_tokens: int, fuse_method: str = "pyramid",
                 phase_alternate: bool = False, freenoise: bool = True, split_conds_to_windows: bool = False,
                 absolute_positions: bool = True, probe_seams: bool = False,
                 halo_tokens: int = 0, halo_start_percent: float = 0.3, margin_tokens: int = 0):
        if window_tokens % G.TOKENS_PER_CYCLE != 2:
            raise ValueError("window_tokens must be 5j+2")
        if stride_tokens % G.TOKENS_PER_CYCLE or stride_tokens <= 0 or stride_tokens >= window_tokens:
            raise ValueError("stride_tokens must be a positive multiple of 5 smaller than window_tokens")
        if halo_tokens % G.TOKENS_PER_CYCLE or halo_tokens < 0:
            raise ValueError("halo_tokens must be a multiple of 5 (whole token cycles)")
        if margin_tokens % G.TOKENS_PER_CYCLE or margin_tokens < 0:
            raise ValueError("margin_tokens must be a multiple of 5 (whole token cycles)")
        super().__init__(context_schedule=cw.ContextSchedule("h3_aligned", _aligned_schedule),
                         fuse_method=cw.get_matching_fuse_method(fuse_method),
                         context_length=window_tokens, context_overlap=window_tokens - stride_tokens,
                         context_stride=1, closed_loop=False, dim=VIDEO_DIM, freenoise=freenoise,
                         cond_retain_index_list="", split_conds_to_windows=split_conds_to_windows,
                         latent_retain_index_list="", causal_window_fix=False)
        self.window_tokens = int(window_tokens)
        self.stride_tokens = int(stride_tokens)
        self.phase_alternate = bool(phase_alternate)
        self.absolute_positions = bool(absolute_positions)
        self.probe_seams = bool(probe_seams)
        self.halo_tokens = int(halo_tokens)
        self.margin_tokens = int(margin_tokens)
        self.halo_start_percent = float(halo_start_percent)
        # half-stride phase shift, preferring the audio-exact 15-token grid
        half = self.stride_tokens // 2
        if self.stride_tokens >= 2 * G.AV_EXACT_TOKEN_STRIDE:
            self.phase_shift_tokens = (half // G.AV_EXACT_TOKEN_STRIDE) * G.AV_EXACT_TOKEN_STRIDE
        else:
            self.phase_shift_tokens = (half // G.TOKENS_PER_CYCLE) * G.TOKENS_PER_CYCLE
        if self.phase_shift_tokens == 0:
            self.phase_alternate = False
        self._payload_cache = {}
        self._layout_cache = {}
        self._warned = set()
        self.last_total_tokens = 0
        # per-sampling-run state
        self._last_x0_video = None      # fused denoised video estimate from the previous step
        self._total_steps = 0
        self.seam_stats = []            # [(step, mean_rel_disagreement, max_rel_disagreement, n_seams)]
        self.halo_active = False

    # ---- state -------------------------------------------------------------------------------

    def _build_window_state(self, x_in, conds, model):
        latent_shapes = self._get_latent_shapes(conds)
        if latent_shapes is None or len(latent_shapes) != 2 or len(latent_shapes[0]) != 5:
            raise ValueError("H3 Context Windows needs a MiniMax H3 audio-video latent "
                             "(Empty MiniMax H3 AV Latent / H3 Long AV Latent)")
        latents = list(comfy.utils.unpack_latents(x_in, latent_shapes))
        T = latents[0].shape[VIDEO_DIM]
        if T % G.TOKENS_PER_CYCLE != 2:
            raise ValueError(f"H3 latent has {T} video tokens; expected 5k+2 (a 17k+5 frame clip)")
        self.last_total_tokens = T
        return H3WindowingState(latents=latents, guide_latents=[None, None], guide_entries=[None, None],
                                keyframe_idxs=[None, None], latent_shapes=latent_shapes, dim=VIDEO_DIM,
                                is_multimodal=True, temporal_downscale_ratio=4)

    def get_context_windows(self, model, x_in, model_options):
        full_length = x_in.size(self.dim)
        inner = self.context_schedule.func(full_length, self, model_options)
        if self.margin_tokens <= 0:
            return [cw.IndexListContextWindow(w, dim=self.dim, total_frames=full_length, context_overlap=self.context_overlap)
                    for w in inner]
        windows = []
        for idx, off, n in _extend_with_margins(inner, full_length, self.margin_tokens):
            w = cw.IndexListContextWindow(idx, dim=self.dim, total_frames=full_length, context_overlap=self.context_overlap)
            w.h3_inner = (off, n)
            windows.append(w)
        return windows

    def set_step(self, timestep, model_options):
        super().set_step(timestep, model_options)
        sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
        self._total_steps = int(sigmas.shape[0]) - 1 if sigmas is not None else 0
        if self._step == 0:
            # a new sampling run: forget the previous run's prediction and stats
            self._last_x0_video = None
            self.seam_stats = []
        frac = self._step / max(1, self._total_steps)
        self.halo_active = (self.halo_tokens > 0 and self._last_x0_video is not None
                            and frac >= self.halo_start_percent)

    # ---- conds -------------------------------------------------------------------------------

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        resized = super().get_resized_cond(cond_in, x_in, window, device)
        if resized is None:
            return None
        audio_window = window.get_window_for_modality(1)
        for c in resized:
            mc = c.get("model_conds")
            if not isinstance(mc, dict):
                continue
            am = mc.get("audio_denoise_mask")
            if am is not None and hasattr(am, "cond") and isinstance(am.cond, torch.Tensor):
                mc["audio_denoise_mask"] = am._copy_with(audio_window.get_tensor(am.cond, device, dim=AUDIO_DIM))
            pl = mc.get("minimax_payload")
            if pl is not None and isinstance(getattr(pl, "cond", None), dict):
                text_len = None
                ca = mc.get("c_crossattn")
                if ca is not None and isinstance(getattr(ca, "cond", None), torch.Tensor):
                    text_len = int(ca.cond.shape[1])
                mc["minimax_payload"] = comfy.conds.CONDConstant(
                    self._window_payload(pl.cond, window, audio_window, x_in, text_len))
        return resized

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            LOG.warning(msg)

    def _crop_guide_video(self, latent, g, f0, f1):
        """Crop a guide clip anchored at global frame g to window frames [f0, f1).

        Returns (local_frame_index, cropped_latent) or None. Handles clips that start before the
        window and extend into it, and clips that run past its end. Keeps whole token cycles only so
        the cropped clip still has the (1,4,4,4,4) phase the model expects.
        """
        n_tok = int(latent.shape[VIDEO_DIM])
        n_frames = G.tokens_to_frames(n_tok)
        if g >= f0 and g + n_frames <= f1:
            return g - f0, latent
        if g + n_frames <= f0 or g >= f1:
            return None
        j0 = None
        for j in range(0, n_tok, G.TOKENS_PER_CYCLE):
            if g + G.token_frame_offset(j) >= f0:
                j0 = j
                break
        if j0 is None:
            return None
        best = None
        n = 2
        while j0 + n <= n_tok:
            if g + G.token_frame_offset(j0) + G.tokens_to_frames(n) <= f1:
                best = n
            n += G.TOKENS_PER_CYCLE
        if best is None:
            return None
        local = g + G.token_frame_offset(j0) - f0
        return local, latent[:, :, j0:j0 + best]

    def _base_keyframes(self, payload, window, audio_window):
        """Source keyframes remapped into this window (local indices, cropped)."""
        f0, f1 = window.h3_frame_start, window.h3_frame_stop
        a_len = len(audio_window.index_list)
        keyframes = []
        for kf in list(payload.get("keyframes") or []):
            g = int(kf["resolved_frame_index"])
            lat = kf.get("latent")
            alat = kf.get("audio_latent")
            new_kf = None
            local = None
            if lat is not None:
                res = self._crop_guide_video(lat, g, f0, f1)
                if res is not None:
                    local, cropped = res
                    if cropped is not lat:
                        self._warn_once(("crop", g, id(lat)),
                                        f"H3 context windows: guide clip at frame {g} crosses window "
                                        f"[{f0},{f1}); cropped to whole token cycles.")
                    new_kf = {"resolved_frame_index": local, "latent": cropped}
            if alat is not None:
                rt = int(alat.shape[-1])
                if local is None and lat is None:
                    local = max(0, g - f0)
                    if g >= f1:
                        local = None
                if local is not None:
                    k0 = max(0, int(round((local + f0 - g) * FRAME_RESCALE)))
                    ticks_avail = int(math.floor(a_len - FRAME_RESCALE * local))
                    n = min(rt - k0, ticks_avail)
                    if n >= 1:
                        if new_kf is None:
                            new_kf = {"resolved_frame_index": local}
                        new_kf["audio_latent"] = alat if (k0 == 0 and n == rt) else alat[..., k0:k0 + n]
            if new_kf is not None:
                keyframes.append(new_kf)
        return keyframes

    def _halo_geometry(self, window):
        """[(local_frame_index, token_start, token_stop), ...] for the halo blocks around this window."""
        if not self.halo_active:
            return []
        s = window.index_list[0]
        L = len(window.index_list)
        T = self.last_total_tokens
        H = self.halo_tokens
        f0 = window.h3_frame_start
        blocks = []
        if s > 0:
            t0 = max(0, s - H)
            blocks.append((G.token_frame_offset(t0) - f0, t0, s))
        if s + L < T:
            t1 = min(T, s + L + H)
            # keep whole cycles: the tail of the clip (T) is a 5k+2 boundary, so clamp to a cycle
            t1 = s + L + ((t1 - (s + L)) // G.TOKENS_PER_CYCLE) * G.TOKENS_PER_CYCLE
            if t1 > s + L:
                blocks.append((G.token_frame_offset(s + L) - f0, s + L, t1))
        return blocks

    def _layout_for(self, sig, text_len, L, lat_h, lat_w, a_len, keyframes, refs, f0):
        layout = self._layout_cache.get(sig)
        if layout is not None:
            return layout
        layout = PackedLayout(text_len, L, lat_h, lat_w, a_len, keyframes=keyframes or None, refs=refs)
        if self.absolute_positions and f0 > 0:
            # both grids are affine in the cursor, so shifting the finished table's time column is
            # identical to building the window from a later cursor; refs and text stay where they are
            off = FRAME_RESCALE * float(f0)
            for a, b, kind in layout.segments:
                if kind in ("video", "audio", "cond", "cond_audio"):
                    layout.position_ids[a:b, 0] += off
        if len(self._layout_cache) >= _LAYOUT_CACHE_MAX:
            self._layout_cache.clear()
        self._layout_cache[sig] = layout
        return layout

    def _window_payload(self, payload, window, audio_window, x_in, text_len):
        s = window.index_list[0]
        L = len(window.index_list)
        f0 = window.h3_frame_start
        a0 = audio_window.index_list[0]
        a_len = len(audio_window.index_list)
        src_keyframes = list(payload.get("keyframes") or [])
        refs = payload.get("refs")
        sig = (id(payload), s, L, text_len, a0, a_len,
               tuple((int(kf["resolved_frame_index"]), id(kf.get("latent")), id(kf.get("audio_latent")))
                     for kf in src_keyframes),
               id(refs))
        base = self._payload_cache.get(sig)
        if base is None:
            base = self._base_keyframes(payload, window, audio_window)
            if len(self._payload_cache) >= _LAYOUT_CACHE_MAX:
                self._payload_cache.clear()
            self._payload_cache[sig] = base

        keyframes = list(base)
        halo = self._halo_geometry(window)
        if halo:
            x0 = self._last_x0_video
            for local, t0, t1 in halo:
                lat = x0[:, :, t0:t1]
                if lat.shape[3] % 2 or lat.shape[4] % 2:
                    # the DiT pads its target latent to the 2x2 patch grid but patchifies cond rows as-is;
                    # pad the halo the same way the target is padded so odd latent sizes work
                    lat = comfy.ldm.common_dit.pad_to_patch_size(lat, (1, 2, 2))
                keyframes.append({"resolved_frame_index": local, "latent": lat, "h3_halo": True})

        new = dict(payload)
        new.pop("layout", None)
        new["keyframes"] = keyframes
        new["cond_video_latents"] = [kf["latent"] for kf in keyframes if kf.get("latent") is not None] + \
                                    [r["latent"] for r in (refs or []) if "latent" in r]
        new["cond_audio_latents"] = [kf["audio_latent"] for kf in keyframes if kf.get("audio_latent") is not None] + \
                                    [r["audio_latent"] for r in (refs or []) if r.get("audio_latent") is not None]
        if text_len is not None:
            lat_h = (int(x_in.shape[3]) + 1) // 2 * 2
            lat_w = (int(x_in.shape[4]) + 1) // 2 * 2
            halo_key = tuple((local, t1 - t0) for local, t0, t1 in halo)
            new["layout"] = self._layout_for(sig + (halo_key,), text_len, L, lat_h, lat_w, a_len, keyframes, refs, f0)
        new["h3_window_start_frame"] = f0
        return new

    # ---- noise -------------------------------------------------------------------------------

    def _apply_freenoise(self, noise, conds, seed):
        latent_shapes = self._get_latent_shapes(conds)
        if latent_shapes is None or len(latent_shapes) != 2:
            return noise
        mods = list(comfy.utils.unpack_latents(noise, latent_shapes))
        mods[0] = cw.apply_freenoise(mods[0], VIDEO_DIM, self.context_length, self.context_overlap, seed)
        T = mods[0].shape[VIDEO_DIM]
        Ta = mods[1].shape[AUDIO_DIM]
        a_len = G.audio_ticks_for_frames(G.tokens_to_frames(self.window_tokens))
        a_stride = max(1, round(self.stride_tokens * Ta / T))
        mods[1] = cw.apply_freenoise(mods[1], AUDIO_DIM, a_len, max(0, a_len - a_stride), seed + 1)
        noise, _ = comfy.utils.pack_latents(mods)
        return noise

    # ---- execution (per-modality temporal dims) ----------------------------------------------

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        self._model = model
        self.set_step(timestep, model_options)
        ws = self._build_window_state(x_in, conds, model)
        dims = (VIDEO_DIM, AUDIO_DIM)
        windows = self.get_context_windows(model, ws.latents[0], model_options)
        total_windows = len(windows)
        relative = self.fuse_method.name == cw.ContextFuseMethods.RELATIVE
        accum = [[torch.zeros_like(m) for _ in conds] for m in ws.latents]
        init = torch.ones if relative else torch.zeros
        counts = [[init(cw.get_shape_for_dim(m, d), device=m.device) for _ in conds] for m, d in zip(ws.latents, dims)]
        biases = [[[0.0] * m.shape[d] for _ in conds] for m, d in zip(ws.latents, dims)]
        probe = []  # (start, video x0 of cond 0) per window when probing

        for callback in comfy.patcher_extension.get_all_callbacks(cw.IndexListCallbacks.EXECUTE_START, self.callbacks):
            callback(self, model, x_in, conds, timestep, model_options)

        for enum_window in enumerate(windows):
            results = self.evaluate_context_windows(calc_cond_batch, model, x_in, conds, timestep, [enum_window],
                                                    model_options, window_state=ws, total_windows=total_windows)
            for result in results:
                if self.probe_seams:
                    inner = getattr(result.window, "h3_inner", None)
                    out0 = result.sub_conds_out[0][0]
                    if inner is not None:
                        off, n = inner
                        probe.append((result.window.index_list[0] + off, out0[:, :, off:off + n]))
                    else:
                        probe.append((result.window.index_list[0], out0))
                for mi in range(2):
                    mod_out = [result.sub_conds_out[ci][mi] for ci in range(len(conds))]
                    mw = result.window.get_window_for_modality(mi)
                    self.combine_context_window_results(ws.latents[mi], mod_out, result.sub_conds, mw,
                                                        result.window_idx, total_windows, timestep,
                                                        accum[mi], counts[mi], biases[mi])
        try:
            out = []
            for ci in range(len(conds)):
                finalized = []
                for mi in range(2):
                    if not relative:
                        accum[mi][ci] /= counts[mi][ci]
                    finalized.append(accum[mi][ci])
                packed, _ = comfy.utils.pack_latents(finalized)
                out.append(packed)
            if self.halo_tokens > 0:
                self._last_x0_video = accum[0][0].detach().clone()
            if self.probe_seams:
                self._record_seams(probe)
            return out
        finally:
            for callback in comfy.patcher_extension.get_all_callbacks(cw.IndexListCallbacks.EXECUTE_CLEANUP, self.callbacks):
                callback(self, model, x_in, conds, timestep, model_options)

    def _record_seams(self, probe):
        """Relative disagreement between neighbouring windows over their shared tokens."""
        probe.sort(key=lambda p: p[0])
        vals = []
        for (s_a, out_a), (s_b, out_b) in zip(probe, probe[1:]):
            L = out_a.shape[VIDEO_DIM]
            ov = s_a + L - s_b
            if ov <= 0:
                continue
            a = out_a[:, :, L - ov:]
            b = out_b[:, :, :ov]
            scale = 0.5 * (a.abs().mean() + b.abs().mean()) + 1e-6
            vals.append(float((a - b).abs().mean() / scale))
        if vals:
            mean_v, max_v = sum(vals) / len(vals), max(vals)
            self.seam_stats.append((self._step, mean_v, max_v, len(vals)))
            LOG.info("H3 seam probe step %d: windows disagree in overlaps by %.3f mean / %.3f max (relative, %d seams)%s",
                     self._step, mean_v, max_v, len(vals), " [halo on]" if self.halo_active else "")

    def combine_context_window_results(self, x_in, sub_conds_out, sub_conds, window, window_idx, total_windows,
                                       timestep, conds_final, counts_final, biases_final):
        dim = window.dim
        inner = getattr(window, "h3_inner", None)
        if self.fuse_method.name == cw.ContextFuseMethods.RELATIVE:
            lo, hi = (0, len(window.index_list)) if inner is None else (inner[0], inner[0] + inner[1])
            first, last = window.index_list[lo], window.index_list[hi - 1]
            for pos, idx in enumerate(window.index_list):
                if pos < lo or pos >= hi:
                    continue
                bias = 1 - abs(idx - (first + last) / 2) / ((last - first + 1e-2) / 2)
                bias = max(1e-2, bias)
                for i in range(len(sub_conds_out)):
                    bias_total = biases_final[i][idx]
                    prev_weight = bias_total / (bias_total + bias)
                    new_weight = bias / (bias_total + bias)
                    idx_window = tuple([slice(None)] * dim + [idx])
                    pos_window = tuple([slice(None)] * dim + [pos])
                    conds_final[i][idx_window] = conds_final[i][idx_window] * prev_weight + sub_conds_out[i][pos_window] * new_weight
                    biases_final[i][idx] = bias_total + bias
        else:
            if inner is None:
                weights = list(cw.get_context_weights(window.context_length, x_in.shape[dim], window.index_list, self,
                                                      sigma=timestep, context_overlap=window.context_overlap))
            else:
                off, n = inner
                inner_idx = window.index_list[off:off + n]
                w_inner = list(cw.get_context_weights(n, x_in.shape[dim], inner_idx, self,
                                                      sigma=timestep, context_overlap=window.context_overlap))
                # margins get a negligible weight: every token is inside some window's inner span
                weights = [1e-6] * len(window.index_list)
                weights[off:off + n] = [float(v) for v in w_inner]
            wt = cw.match_weights_to_dim(list(weights), x_in, dim, device=x_in.device)
            for i in range(len(sub_conds_out)):
                window.add_window(conds_final[i], sub_conds_out[i] * wt)
                window.add_window(counts_final[i], wt)
        for callback in comfy.patcher_extension.get_all_callbacks(cw.IndexListCallbacks.COMBINE_CONTEXT_WINDOW_RESULTS, self.callbacks):
            callback(self, x_in, sub_conds_out, sub_conds, window, window_idx, total_windows, timestep,
                     conds_final, counts_final, biases_final)


def make_prepare_sampling_wrapper(handler: H3ContextHandler, expected_total_tokens: int):
    """Budget VRAM for one window instead of the whole packed latent (core skips packed latents)."""
    def wrapper(executor, model, noise_shape, conds, *args, **kwargs):
        shape = list(noise_shape)
        if expected_total_tokens > handler.window_tokens and len(shape) == 3:
            shape[-1] = max(1, int(shape[-1] * handler.window_tokens / expected_total_tokens))
        return executor(model, shape, conds, *args, **kwargs)
    return wrapper
