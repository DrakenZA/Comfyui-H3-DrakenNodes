"""Window reference integration against native node schemas, layouts and cond processing."""

from types import SimpleNamespace

import torch

import comfy.conds
import comfy.model_base
import comfy.model_patcher
import comfy.nested_tensor
import comfy.ops
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers
import comfy.supported_models
import comfy.utils
from comfy.ldm.minimax.model import FRAME_RESCALE, PackedLayout
from comfy_extras import nodes_minimax_h3 as native

from h3_drakennodes_pkg import h3_grid as G
from h3_drakennodes_pkg import long_reference as R
from h3_drakennodes_pkg.handler import H3ContextHandler, make_prepare_sampling_wrapper
from h3_drakennodes_pkg.nodes import H3ContextWindows


def _frame_numbers(frames):
    return (frames[:, 0, 0, 0] * 255).round() + (frames[:, 0, 0, 1] * 255).round() * 256


def _frames(count, offset=0):
    numbers = torch.arange(count) + offset
    colors = torch.stack([numbers % 256 / 255, numbers // 256 / 255, torch.zeros(count)], dim=1)
    return colors[:, None, None, :].expand(-1, 32, 32, -1).clone()


class Clip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, minimax_ref_items):
        return minimax_ref_items

    def encode_from_tokens_scheduled(self, items):
        self.calls.append(items)
        videos = [i["data"] for i in items if i["type"] == "video"]
        value = float(_frame_numbers(videos[0])[0]) if videos else -1.0
        length = 4 + sum(v.shape[0] for v in videos)
        return [[torch.full((1, length, 8), value), {"minimax_token_tags": torch.arange(length) % 2}]]


class VideoVAE:
    def __init__(self):
        self.calls = []

    def encode(self, frames):
        n, h, w, _ = frames.shape
        value = -2.0 if n == 1 else float(_frame_numbers(frames)[0])
        self.calls.append((value, n))
        return torch.full((1, 24, 1 if n == 1 else G.frames_to_tokens(n), h // 16, w // 16), value)


class AudioVAE:
    audio_sample_rate = 240

    def __init__(self):
        self.calls = []

    def encode(self, waveform):
        value = float(waveform[0, 0, 0])
        self.calls.append((value, waveform.shape[1]))
        return torch.full((1, 32, 2, round(waveform.shape[1] / 240 * 40)), value)


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.dtype = torch.float32

    def preprocess_text_embeds(self, states):
        if states.shape[-1] == 12:
            return states
        self.calls += 1
        return torch.nn.functional.pad(states + 0.25, (0, 4))


def _model(shapes):
    # Exercise native MiniMaxH3.extra_conds without allocating model weights.
    model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.diffusion_model = Diffusion()
    model.concat_keys = ()
    model.manual_cast_dtype = None
    model.latent_shapes = shapes
    model.model_sampling = SimpleNamespace(audio_scale=4.0)
    return model


def _inputs(frames=345, vae=True, audio=True, extra_video=False):
    clip, video_vae, audio_vae = Clip(), VideoVAE(), AudioVAE()
    video = _frames(frames)
    soundtrack = {"waveform": torch.arange(frames * 10).float().view(1, 1, -1).repeat(1, 2, 1) / 10,
                  "sample_rate": 240}
    kwargs = dict(clip=clip, prompt="<Picture 1> performs <Video 1>", width=32, height=32, length=frames,
                  vae=video_vae if vae else None, audio_vae=audio_vae if audio else None,
                  ref_images={"ref_image_1": torch.full((1, 32, 32, 3), 0.5)},
                  ref_videos={"ref_video_1": video}, ref_video_audios={"ref_video_audio_1": soundtrack},
                  ref_audios={"ref_audio_1": {"waveform": torch.full((1, 2, 480), -3.0), "sample_rate": 240}})
    if extra_video:
        kwargs["ref_videos"]["ref_video_2"] = _frames(frames, 1000)
    return kwargs, clip, video_vae, audio_vae


def _raises(fn, message):
    try:
        fn()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {message}")


def test_reference_schema_and_segments():
    original = native.MiniMaxH3ReferenceToVideo.define_schema().get_v1_info(native.MiniMaxH3ReferenceToVideo)
    custom = R.H3LongReferenceToVideo.define_schema().get_v1_info(R.H3LongReferenceToVideo)
    assert original.input == custom.input
    assert original.output == custom.output
    assert original.output_name == custom.output_name
    kwargs, clip, vv, av = _inputs(extra_video=True)
    positive, latent = R.H3LongReferenceToVideo.execute(**kwargs)
    source = positive[0][1]["model_conds"][R.REFERENCE_KEY].cond
    assert vv.calls == [(-2.0, 1)]  # Shared image encoded once, long video still deferred.
    assert av.calls == [(-3.0, 480)]
    entry = source.encode_window(15, G.frames_to_tokens(141))
    assert vv.calls[1:] == [(51.0, 141), (1051.0, 141)]
    assert av.calls[1:] == [(51.0, 1410)]
    assert [r["kind"] for r in entry.refs] == ["image", "video_audio", "video", "audio"]
    assert [i["type"] for i in clip.calls[-1]] == ["image", "audio", "video", "video", "audio"]
    assert torch.equal(_frame_numbers(clip.calls[-1][2]["data"]), torch.arange(51, 51 + 141, 12).float())
    assert clip.calls[-1][2]["timestamps"] == [i / 2 for i in range(12)]
    assert entry.refs[0] is source.image_refs[0] and entry.refs[-1] is source.audio_refs[0]
    assert latent["samples"].unbind()[0].shape[2] == G.frames_to_tokens(345)
    _raises(lambda: positive[0][1]["model_conds"][R.REFERENCE_KEY].process_cond(1), "needs H3 Context Windows")
    # The window matches the stock node on the same source segment, in both paths.
    local = dict(kwargs, length=141,
                 ref_videos={k: v[51:192] for k, v in kwargs["ref_videos"].items()},
                 ref_video_audios={k: {**a, "waveform": a["waveform"][..., 510:1920]}
                                   for k, a in kwargs["ref_video_audios"].items()})
    stock, _ = native.MiniMaxH3ReferenceToVideo.execute(**local)
    torch.testing.assert_close(stock[0][0], entry.embeds)
    assert comfy.conds.is_equal(stock[0][1]["minimax_refs"], entry.refs)
    assert torch.equal(stock[0][1]["minimax_token_tags"], entry.tags)


def _sampling_case(frames=345, margins=0, alternate=False, zero_negative=False):
    kwargs, clip, vv, av = _inputs(frames)
    positive, latent = R.H3LongReferenceToVideo.execute(**kwargs)
    marker = positive[0][1]["model_conds"][R.REFERENCE_KEY]
    original = {"positive": comfy.sampler_helpers.convert_cond(positive)}
    if zero_negative:
        negative = [[torch.zeros_like(positive[0][0]), positive[0][1].copy()]]
        original["negative"] = comfy.sampler_helpers.convert_cond(negative)
    streams = latent["samples"].unbind()
    noise, shapes = comfy.utils.pack_latents(list(streams))
    handler = H3ContextHandler(G.frames_to_tokens(141), 15, phase_alternate=alternate,
                               margin_tokens=margins)
    real = _model(shapes)
    guider = SimpleNamespace(conds=original)
    patcher = SimpleNamespace(load_device=torch.device("cpu"))
    expected_plan = R.planned_reference_windows(handler, shapes[0][2])
    seen = []

    def sampling(noise, latent_image, sampler, sigmas, **kw):
        def load(model, shape, conds):
            assert len(clip.calls) == 1 + len(expected_plan)
            return real, conds, []
        _, raw, _ = R.make_reference_prepare_wrapper()(load, patcher, noise.shape, guider.conds)
        conds = [comfy.samplers.encode_model_conds(real.extra_conds, cs, noise, "cpu", branch,
                                                  latent_shapes=shapes)
                 for branch, cs in raw.items()]
        encoded_counts = len(clip.calls), len(vv.calls), len(av.calls), real.diffusion_model.calls
        if shapes[0][2] <= handler.window_tokens:
            assert R.REFERENCE_KEY not in conds[0][0]["model_conds"]
            assert conds[0][0]["model_conds"]["minimax_payload"].cond["refs"][1]["latent_t"] == shapes[0][2]
            return noise

        def calc(model, sub_conds, x, timestep, opts):
            window = opts["transformer_options"]["context_window"]
            key = window.index_list[0], len(window.index_list)
            assert key in expected_plan
            first = G.token_frame_offset(key[0])
            for ci, branch in enumerate(sub_conds):
                mc = branch[0]["model_conds"]
                assert R.REFERENCE_KEY not in mc
                payload = mc["minimax_payload"].cond
                video_ref = payload["refs"][1]
                assert video_ref["latent_t"] == key[1]
                assert float(video_ref["latent"][0, 0, 0, 0, 0]) == first
                assert float(video_ref["audio_latent"][0, 0, 0, 0]) == first
                # Qwen content and VAE content agree; projection is done once before denoising.
                assert float(mc["c_crossattn"].cond[0, 0, 0]) == (0.25 if ci else first + 0.25)
                table = raw[list(raw)[ci]][0]["model_conds"][R.REFERENCE_KEY].cond
                layout = payload["layout"]
                va = next(a for a, _, kind in layout.segments if kind == "video")
                assert abs(float(layout.position_ids[va, 0]) - table.target_origin - FRAME_RESCALE * first) < 1e-6
                assert payload["text_token_tags"].shape[0] == mc["c_crossattn"].cond.shape[1]
                assert layout.signature[0] == mc["c_crossattn"].cond.shape[1]
                assert payload["audio_scale"] == 4.0
            seen.append(key)
            assert encoded_counts == (len(clip.calls), len(vv.calls), len(av.calls), real.diffusion_model.calls)
            return [-x for _ in sub_conds]

        options = {"transformer_options": {"sample_sigmas": sigmas}}
        for step in range(2):
            outputs = handler.execute(calc, real, conds, noise, sigmas[step:step + 1], options)
            assert all(torch.allclose(out, -noise, atol=1e-5) for out in outputs)
        assert set(seen) == set(expected_plan)
        return noise

    executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
        sampling, guider, [R.make_reference_sampling_wrapper(handler)])
    executor.execute(noise, noise, None, torch.tensor([1.0, 0.5, 0.0]), latent_shapes=shapes)
    assert guider.conds is original
    assert marker.cond is positive[0][1]["model_conds"][R.REFERENCE_KEY].cond
    assert not handler._payload_cache and not handler._layout_cache
    assert len(clip.calls) == 1 + len(expected_plan)
    assert len(vv.calls) == 1 + len(expected_plan)
    assert len(av.calls) == 1 + len(expected_plan)


def test_reference_sampling_and_cache():
    _sampling_case(alternate=True, margins=5, zero_negative=True)
    _sampling_case(margins=5)
    _sampling_case(frames=90)


def test_reference_validation_and_optional_vaes():
    kwargs, clip, vv, av = _inputs(vae=False, audio=False)
    positive, _ = R.H3LongReferenceToVideo.execute(**kwargs)
    entry = positive[0][1]["model_conds"][R.REFERENCE_KEY].cond.encode_window(15, 42)
    assert not entry.refs and not vv.calls and not av.calls
    assert [i["type"] for i in clip.calls[-1]] == ["image", "audio", "video", "audio"]
    assert entry.embeds[0, 0, 0] == 51
    kwargs["length"] = 362
    _raises(lambda: R.H3LongReferenceToVideo.execute(**kwargs), "at least 362 frames")
    kwargs["length"] = 345
    kwargs["ref_video_audios"]["ref_video_audio_1"]["waveform"] = torch.zeros(1, 2, 10)
    _raises(lambda: R.H3LongReferenceToVideo.execute(**kwargs), "soundtrack is shorter")
    kwargs["ref_videos"] = {}
    plain, _ = R.H3LongReferenceToVideo.execute(**kwargs)
    assert R.REFERENCE_KEY not in plain[0][1].get("model_conds", {})


def test_reference_cleanup_and_memory_budget():
    kwargs, _, _, _ = _inputs()
    positive, latent = R.H3LongReferenceToVideo.execute(**kwargs)
    original = {"positive": comfy.sampler_helpers.convert_cond(positive)}
    guider = SimpleNamespace(conds=original)
    handler = H3ContextHandler(42, 15)
    noise, shapes = comfy.utils.pack_latents(list(latent["samples"].unbind()))

    def fail(*args, **kwargs):
        captured = {}
        make_prepare_sampling_wrapper(handler, 0)(
            lambda m, shape, cs: captured.setdefault("shape", shape), None, noise.shape, guider.conds)
        assert captured["shape"][-1] == int(noise.shape[-1] * 42 / shapes[0][2])
        raise RuntimeError("cancel sampling")

    executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
        fail, guider, [R.make_reference_sampling_wrapper(handler)])
    try:
        executor.execute(noise, noise, None, torch.tensor([1.0, 0.0]), latent_shapes=shapes)
    except RuntimeError as error:
        assert str(error) == "cancel sampling"
    else:
        raise AssertionError("Sampling did not fail")
    assert guider.conds is original and not handler._payload_cache and not handler._layout_cache
    short_shapes = [torch.Size([1, 24, 42, 2, 2]), shapes[1]]
    _raises(lambda: executor.execute(noise, noise, None, torch.tensor([1.0, 0.0]), latent_shapes=short_shapes),
            "Reference length and target latent length differ")


def test_reference_native_sampler():
    # Real CFGGuider + ModelPatcher + H3 model loading/extra_conds/forward, with
    # random tiny weights and no attention blocks (no optional compiled kernels).
    config = comfy.supported_models.MiniMaxH3(dict(
        hidden_size=12, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=2, attention_head_dim=6, ffn_hidden_size=16,
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=12,
        time_embed_dim=6, rope_inv_freq_len=1, dtype=torch.float32))
    config.custom_operations = comfy.ops.disable_weight_init
    base = comfy.model_base.MiniMaxH3(config, device=torch.device("cpu"))
    for parameter in base.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    base.diffusion_model.rope.inv_freq.fill_(0.1)
    patcher = comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    patched, _ = H3ContextWindows().apply(patcher, 141, 51, "pyramid", True, False, False, False)
    kwargs, clip, vv, av = _inputs(192)
    positive, latent = R.H3LongReferenceToVideo.execute(**kwargs)
    noise = comfy.nested_tensor.NestedTensor([torch.randn_like(z) for z in latent["samples"].unbind()])
    negative = [[torch.zeros_like(positive[0][0]), positive[0][1].copy()]]
    result = comfy.samplers.sample(patched, noise, positive, negative, 1.0, torch.device("cpu"),
                                   comfy.samplers.sampler_object("euler"), torch.tensor([1.0, 0.5, 0.0]),
                                   model_options=patched.model_options, latent_image=latent["samples"], seed=721)
    assert result.is_nested
    assert [z.shape for z in result.unbind()] == [z.shape for z in noise.unbind()]
    assert all(torch.isfinite(z).all() for z in result.unbind())
    assert len(clip.calls) == len(vv.calls) == len(av.calls) == 3
    assert not patched.model_options["context_handler"]._payload_cache
