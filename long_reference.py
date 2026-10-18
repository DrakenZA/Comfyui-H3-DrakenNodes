"""Native H3 reference conditioning prepared for the context handler's actual windows."""

import logging
import math
from dataclasses import dataclass, replace

import torch

import comfy.conds
import comfy.model_management
from comfy.ldm.minimax.model import _ref_t_span
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as native

from . import h3_grid as G

LOG = logging.getLogger("h3_context_windows")
REFERENCE_KEY = "draken_h3_window_reference"


class WindowReferenceCond(comfy.conds.CONDConstant):
    def process_cond(self, batch_size, **kwargs):
        raise ValueError("H3 Long Reference to Video needs H3 Context Windows (Draken) on the sampler's model.")


@dataclass
class ReferenceWindow:
    embeds: torch.Tensor
    tags: torch.Tensor
    refs: list


@dataclass
class PreparedReferences:
    windows: dict
    target_origin: float
    total_tokens: int


def _audio_slice(audio, start_frame, frame_count):
    sr = audio["sample_rate"]
    first = round(start_frame * sr / G.FPS)
    last = round((start_frame + frame_count) * sr / G.FPS)
    if last > audio["waveform"].shape[-1]:
        raise ValueError("A reference video soundtrack is shorter than its requested window. Supply matching 24 fps footage/audio or reduce length.")
    return {"waveform": audio["waveform"][..., first:last], "sample_rate": sr}


class WindowReferenceSource:
    def __init__(self, clip, vae, audio_vae, prompt, width, height, length,
                 ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios):
        self.clip, self.vae, self.audio_vae = clip, vae, audio_vae
        self.prompt = prompt
        self.width, self.height, self.length = width, height, length
        self.videos = {k: v.cpu() for k, v in (ref_videos or {}).items() if v is not None}
        self.soundtracks = {k: {**v, "waveform": v["waveform"].cpu()}
                            for k, v in (ref_video_audios or {}).items() if v is not None}
        self.images = []
        self.image_refs = []
        self.audio_refs = []
        self.audio_items = []
        for name, frames in self.videos.items():
            if frames.shape[0] < length:
                raise ValueError(f"{name} has {frames.shape[0]} frames; synchronized long references need at least {length} frames at 24 fps. Reduce length or supply a longer reference.")
            soundtrack = self.soundtracks.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            if soundtrack is not None:
                _audio_slice(soundtrack, 0, length)
        for image in (ref_images or {}).values():
            if image is None:
                continue
            h, w = image.shape[1:3]
            scale = min(1.0, math.sqrt(width * height / (w * h))) if ref_image_size == "match" else min(1.0, native.REF_IMAGE_SHORT_EDGE / min(w, h))
            tw, th = (max(32, round(d * scale / 32) * 32) for d in (w, h))
            resized = native._resize(image[:1], tw, th, "disabled").cpu()
            self.images.append({"type": "image", "data": resized})
            if vae is not None:
                self.image_refs.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                                        "latent": vae.encode(resized).cpu()})
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            self.audio_items.append({"type": "audio"})
            if audio_vae is not None:
                z, ticks = native._encode_ref_audio(audio_vae, audio)
                self.audio_refs.append({"kind": "audio", "ref_audio_t": ticks, "audio_latent": z.cpu()})

    def encode_text(self, items):
        cond = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(self.prompt, minimax_ref_items=items))
        if len(cond) != 1:
            raise ValueError("H3 long references require one Qwen conditioning entry per prompt.")
        embeds, meta = cond[0]
        tags = meta.get("minimax_token_tags")
        if tags is None:
            raise ValueError("Use the native MiniMax H3 Qwen encoder; its conditioning must include minimax_token_tags.")
        return embeds.cpu(), tags.cpu()

    def _window_range(self, start, tokens):
        first = G.token_frame_offset(start)
        count = G.tokens_to_frames(tokens)
        if first + count > self.length:
            raise ValueError("Target windows exceed the reference node's length. Use its latent output or match the reference length to the target.")
        return first, count

    def _segment(self, frames):
        h, w = frames.shape[1:3]
        cw, ch = native.adapt_canvas(w, h)
        if w * h < cw * ch:
            cw, ch = (max(32, round(d / 32) * 32) for d in (w, h))
        return native._resize(frames, cw, ch, "disabled")

    def encode_window_refs(self, start, tokens):
        first, count = self._window_range(start, tokens)
        refs = list(self.image_refs)
        for name, frames in self.videos.items():
            if self.vae is None:
                continue
            segment = self._segment(frames[first:first + count])
            ch, cw = segment.shape[1:3]
            soundtrack = self.soundtracks.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            z = self.vae.encode(segment).cpu()
            az, ticks = None, 0
            if soundtrack is not None and self.audio_vae is not None:
                az, ticks = native._encode_ref_audio(self.audio_vae, _audio_slice(soundtrack, first, count))
                az = az.cpu()
            refs.append({"kind": "video_audio" if ticks else "video", "latent_t": z.shape[2],
                         "latent_h": ch // 16, "latent_w": cw // 16, "ref_audio_t": ticks,
                         "latent": z, "audio_latent": az})
        refs.extend(self.audio_refs)
        return refs

    def encode_window_text(self, start, tokens):
        first, count = self._window_range(start, tokens)
        items = list(self.images)
        sample_idx = list(range(0, count, G.FPS // 2))
        for name, frames in self.videos.items():
            if "ref_video_audio_" + name.rsplit("_", 1)[-1] in self.soundtracks:
                items.append({"type": "audio"})
            sampled = self._segment(frames[first:first + count:G.FPS // 2])
            items.append({"type": "video", "data": sampled,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        return self.encode_text(items + self.audio_items)

    def encode_window(self, start, tokens):
        refs = self.encode_window_refs(start, tokens)
        embeds, tags = self.encode_window_text(start, tokens)
        return ReferenceWindow(embeds, tags, refs)


class H3LongReferenceToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return replace(native.MiniMaxH3ReferenceToVideo.define_schema(),
                       node_id="DrakenH3LongReferenceToVideo",
                       display_name="H3 Long Reference to Video (Draken)", category="DrakenNodes/H3",
                       description="Native H3 reference inputs/outputs with video and soundtrack references matched to H3 Context Windows. Source frames must already be 24 fps and cover length. Connect positive and latent to the sampler and use H3 Context Windows (Draken) on its model.")

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None, audio_vae=None,
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        if not any(v is not None for v in (ref_videos or {}).values()):
            return native.MiniMaxH3ReferenceToVideo.execute(
                clip=clip, prompt=prompt, width=width, height=height, length=length,
                ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
                ref_images=ref_images, ref_videos=ref_videos,
                ref_video_audios=ref_video_audios, ref_audios=ref_audios)
        latent, frames = native._empty_av_latent(width, height, length)
        source = WindowReferenceSource(clip, vae, audio_vae, prompt, width, height, frames,
                                       ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios)
        embeds, tags = source.encode_text(source.images + source.audio_items)
        source.base_embeds = embeds
        positive = [[embeds, {"minimax_token_tags": tags,
                              "minimax_refs": source.image_refs + source.audio_refs,
                              "model_conds": {REFERENCE_KEY: WindowReferenceCond(source)}}]]
        return io.NodeOutput(positive, latent)


def planned_reference_windows(handler, total_tokens):
    if total_tokens <= handler.window_tokens:
        return [(0, total_tokens)]
    shape = torch.empty((1, 1, total_tokens), device="meta")
    previous_step = handler._step
    plan = set()
    try:
        for step in range(2 if handler.phase_alternate else 1):
            handler._step = step
            for window in handler.get_context_windows(None, shape, {}):
                plan.add((window.index_list[0], len(window.index_list)))
    finally:
        handler._step = previous_step
    return sorted(plan)


def make_reference_sampling_wrapper(handler):
    def wrapper(executor, noise, latent_image, sampler, sigmas, *args, latent_shapes=None, **kwargs):
        guider = executor.class_obj
        original = guider.conds
        sources = [c["model_conds"][REFERENCE_KEY].cond for cs in original.values() for c in cs
                   if REFERENCE_KEY in c.get("model_conds", {})]
        if not sources:
            return executor(noise, latent_image, sampler, sigmas, *args, latent_shapes=latent_shapes, **kwargs)
        if latent_shapes is None or len(latent_shapes) != 2 or len(latent_shapes[0]) != 5:
            raise ValueError("H3 long references require a native H3 audio/video latent.")
        total = latent_shapes[0][2]
        plan = planned_reference_windows(handler, total)
        tables = {}
        try:
            for source in sources:
                if source in tables:
                    continue
                if G.tokens_to_frames(total) != source.length:
                    raise ValueError("Reference length and target latent length differ. Match length or use the long reference node's latent output.")
                refs = {}
                for i, key in enumerate(plan):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    LOG.info("H3 long reference: VAE window %d/%d, frames %d-%d", i + 1, len(plan),
                             G.token_frame_offset(key[0]), G.token_frame_offset(key[0]) + G.tokens_to_frames(key[1]))
                    refs[key] = source.encode_window_refs(*key)
                entries = {}
                for i, key in enumerate(plan):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    LOG.info("H3 long reference: Qwen window %d/%d", i + 1, len(plan))
                    embeds, tags = source.encode_window_text(*key)
                    entries[key] = ReferenceWindow(embeds, tags, refs[key])
                origin = max(e.embeds.shape[1] + sum(_ref_t_span(r) for r in e.refs) for e in entries.values())
                tables[source] = PreparedReferences(entries, origin, total)
            guider.conds = {}
            for branch, conds in original.items():
                guider.conds[branch] = []
                for cond in conds:
                    cond = cond.copy()
                    mc = cond.get("model_conds", {}).copy()
                    marker = mc.get(REFERENCE_KEY)
                    if marker is not None:
                        source = marker.cond
                        zeroed = not bool(torch.count_nonzero(cond["cross_attn"]))
                        if not zeroed and not torch.equal(cond["cross_attn"], source.base_embeds):
                            raise ValueError("H3 long reference embeddings were modified. Use prompts in the reference node; Conditioning Combine and Conditioning Zero Out are supported.")
                        table = tables[source]
                        if zeroed:
                            table = replace(table, windows={k: replace(e, embeds=torch.zeros_like(e.embeds))
                                                            for k, e in table.windows.items()})
                        # Use the largest encoded presentation for core's memory estimate.
                        entry = max(table.windows.values(), key=lambda e: e.embeds.shape[1])
                        cond.update(cross_attn=entry.embeds,
                                    minimax_refs=entry.refs, minimax_token_tags=entry.tags)
                        if total <= handler.window_tokens:
                            mc.pop(REFERENCE_KEY)
                        else:
                            mc[REFERENCE_KEY] = WindowReferenceCond(table)
                    cond["model_conds"] = mc
                    guider.conds[branch].append(cond)
            return executor(noise, latent_image, sampler, sigmas, *args, latent_shapes=latent_shapes, **kwargs)
        finally:
            guider.conds = original
            handler._payload_cache.clear()
            handler._layout_cache.clear()
    return wrapper


def make_reference_prepare_wrapper():
    def wrapper(executor, model, noise_shape, conds, *args, **kwargs):
        real_model, prepared_conds, models = executor(model, noise_shape, conds, *args, **kwargs)
        seen = set()
        for cs in prepared_conds.values():
            for cond in cs:
                marker = cond.get("model_conds", {}).get(REFERENCE_KEY)
                if marker is None:
                    continue
                for entry in marker.cond.windows.values():
                    if id(entry) in seen:
                        continue
                    seen.add(id(entry))
                    entry.embeds = real_model.diffusion_model.preprocess_text_embeds(
                        entry.embeds.to(device=model.load_device, dtype=real_model.get_dtype_inference())).cpu()
        return real_model, prepared_conds, models
    return wrapper
