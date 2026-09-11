"""CPU integration test of the handler against real ComfyUI core code with a stand-in model.

Run via tests/run_tests.py (it sets up sys.path for a ComfyUI checkout and stubs comfy_aimdo).
The stand-in "model" returns -x for every window, so the fused result must equal -x_in exactly:
that verifies slicing, audio mapping, packing/unpacking and fusion for both modalities.
It also checks per-window shapes, audio tick ranges, payload/keyframe remapping and layouts.
"""
import math

import torch

import comfy.conds
import comfy.utils
from comfy.ldm.minimax.model import PackedLayout

from h3_drakennodes_pkg import h3_grid as G
from h3_drakennodes_pkg.handler import H3ContextHandler, VIDEO_DIM, AUDIO_DIM


class FakeModel:
    latent_format = type("LF", (), {"temporal_downscale_ratio": 4})()

    def resize_cond_for_context_window(self, cond_key, cond_value, window, x_in, device, retain_index_list=[]):
        return None  # same as comfy.model_base.BaseModel: MiniMaxH3 does not override it


def build_conds(total_frames, lat_h, lat_w, text_len, keyframes, refs=None, with_masks=False):
    T = G.frames_to_tokens(total_frames)
    Ta = G.audio_ticks_for_frames(total_frames)
    latent_shapes = [torch.Size([1, 24, T, lat_h, lat_w]), torch.Size([1, 32, 2, Ta])]
    payload = {"keyframes": keyframes, "seed": 0, "audio_scale": 4.0, "text_token_tags": torch.zeros(text_len, dtype=torch.long)}
    if refs:
        payload["refs"] = refs
    payload["cond_video_latents"] = [kf["latent"] for kf in keyframes if kf.get("latent") is not None]
    payload["cond_audio_latents"] = [kf["audio_latent"] for kf in keyframes if kf.get("audio_latent") is not None]
    payload["layout"] = PackedLayout(text_len, T, lat_h, lat_w, Ta, keyframes=keyframes or None, refs=refs)
    mc = {
        "latent_shapes": comfy.conds.CONDConstant(latent_shapes),
        "c_crossattn": comfy.conds.CONDRegular(torch.randn(1, text_len, 8)),
        "minimax_payload": comfy.conds.CONDConstant(payload),
    }
    if with_masks:
        vm = torch.ones(1, 1, T, lat_h, lat_w)
        vm[:, :, :12] = 0.0
        am = torch.ones(1, 1, 2, Ta)
        am[..., :65] = 0.0
        mc["denoise_mask"] = comfy.conds.CONDRegular(vm)
        mc["audio_denoise_mask"] = comfy.conds.CONDRegular(am)
    cond = [{"model_conds": mc}]
    return [cond], latent_shapes, payload


def run_case(total_frames, window_frames, stride_frames, keyframes, lat_h=6, lat_w=8, text_len=16,
             fuse="pyramid", steps=(0, 1), with_masks=False, refs=None, split=False, absolute=True,
             phase_alternate=True):
    conds, latent_shapes, payload = build_conds(total_frames, lat_h, lat_w, text_len, keyframes, refs, with_masks)
    T, Ta = latent_shapes[0][2], latent_shapes[1][3]
    video = torch.randn(latent_shapes[0])
    audio = torch.randn(latent_shapes[1])
    x_in, shapes = comfy.utils.pack_latents([video, audio])
    L = G.snap_window_tokens(window_frames)
    S = G.snap_stride_tokens(stride_frames)
    handler = H3ContextHandler(L, S, fuse_method=fuse, phase_alternate=phase_alternate, freenoise=True,
                               split_conds_to_windows=split, absolute_positions=absolute)
    seen = []

    def calc_cond_batch(model, sub_conds, sub_x, sub_t, model_options):
        assert model is fake
        outs = []
        for cond in sub_conds:
            mc = cond[0]["model_conds"]
            shp = mc["latent_shapes"].cond
            v, a = comfy.utils.unpack_latents(sub_x, shp)
            win = model_options["transformer_options"]["context_window"]
            s = win.index_list[0]
            assert v.shape[VIDEO_DIM] == L and v.shape[VIDEO_DIM] % 5 == 2, v.shape
            assert s % 5 == 0
            a0, a1 = G.window_audio_range(s, L, T, Ta)
            assert a.shape[AUDIO_DIM] == a1 - a0 == G.audio_ticks_for_frames(G.tokens_to_frames(L)), (a.shape, a0, a1)
            # window content must be the exact slice of the global latent
            assert torch.equal(v, video[:, :, s:s + L]) and torch.equal(a, audio[..., a0:a1])
            pl = mc["minimax_payload"].cond
            lay = pl["layout"]
            assert lay.signature == (text_len, L, lat_h, lat_w, a1 - a0), lay.signature
            f0 = G.token_frame_offset(s)
            f1 = f0 + G.tokens_to_frames(L)
            # RoPE time axis: text/refs untouched; target + keyframe rows offset by 5/3 * f0 when absolute
            from comfy.ldm.minimax.model import FRAME_RESCALE
            ref_span = sum(1.0 for r in (refs or []) if r.get("kind") == "image")
            cursor = text_len + ref_span
            expect = cursor + (FRAME_RESCALE * f0 if absolute else 0.0)
            va, vb, _ = next(seg for seg in lay.segments if seg[2] == "video")
            aa, ab, _ = next(seg for seg in lay.segments if seg[2] == "audio")
            assert abs(float(lay.position_ids[va, 0]) - expect) < 1e-6, (float(lay.position_ids[va, 0]), expect)
            assert abs(float(lay.position_ids[aa, 0]) - expect) < 1e-6
            assert abs(float(lay.position_ids[0, 0])) < 1e-9  # text starts at 0
            for ra, rb, kind in lay.segments:
                if kind == "ref_img":
                    assert abs(float(lay.position_ids[ra, 0]) - text_len) < 1e-6
                if kind == "cond":
                    t = float(lay.position_ids[ra, 0])
                    assert expect - 1e-6 <= t < expect + FRAME_RESCALE * (f1 - f0) + 1e-6, (t, expect)
            for kf in pl["keyframes"]:
                li = kf["resolved_frame_index"]
                assert 0 <= li < f1 - f0, (li, f0, f1)
                if kf.get("latent") is not None:
                    n = kf["latent"].shape[2]
                    assert n == 1 or n % 5 == 2, n
                    assert li + G.tokens_to_frames(n) <= f1 - f0
                if kf.get("audio_latent") is not None:
                    assert kf["audio_latent"].shape[-1] <= math.floor((a1 - a0) - (5 / 3) * li)
            assert len(pl["cond_video_latents"]) == sum(1 for kf in pl["keyframes"] if kf.get("latent") is not None) + len([r for r in (refs or []) if "latent" in r])
            if with_masks:
                assert mc["denoise_mask"].cond.shape[2] == L
                assert mc["audio_denoise_mask"].cond.shape[3] == a1 - a0
                assert torch.equal(mc["audio_denoise_mask"].cond, conds[0][0]["model_conds"]["audio_denoise_mask"].cond[..., a0:a1])
            seen.append((s, a0, [dict(k, latent=None if k.get("latent") is None else tuple(k["latent"].shape),
                                      audio_latent=None if k.get("audio_latent") is None else tuple(k["audio_latent"].shape))
                                 for k in pl["keyframes"]]))
            outs.append(-sub_x)
        return outs

    fake = FakeModel()
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.0])
    model_options = {"transformer_options": {"sample_sigmas": sigmas}}
    assert handler.should_use_context(fake, conds, x_in, sigmas[:1], model_options)
    results = {}
    for step in steps:
        t = sigmas[step:step + 1]
        out = handler.execute(calc_cond_batch, fake, conds, x_in, t, model_options)
        assert len(out) == 1
        assert torch.allclose(out[0], -x_in, atol=1e-5), (out[0] - (-x_in)).abs().max()
        results[step] = [w for w in seen]
        seen.clear()
    return handler, results


def test_identity_and_phase():
    handler, res = run_case(753, 141, 51, keyframes=[], phase_alternate=True)
    starts0 = [s for s, _, _ in res[0]]
    starts1 = [s for s, _, _ in res[1]]
    assert starts0[0] == 0 and starts0[-1] == handler.last_total_tokens - handler.window_tokens
    assert starts0 != starts1, "phase alternation should move interior windows on odd steps"
    assert handler.phase_shift_tokens in starts1  # 5-token shift for a 15-token stride, 15 for >=30
    # all tokens covered on both phases
    for starts in (starts0, starts1):
        covered = set()
        for s in starts:
            covered.update(range(s, s + handler.window_tokens))
        assert covered == set(range(handler.last_total_tokens))


def test_keyframes_remap():
    vae_like = torch.randn(1, 24, 12, 6, 8)   # 39-frame guide clip (12 tokens)
    still = torch.randn(1, 24, 1, 6, 8)
    aud = torch.randn(1, 32, 2, 65)           # 65 ticks == 39 frames
    keyframes = [
        {"resolved_frame_index": 0, "latent": vae_like, "audio_latent": aud},   # footage tail at the head
        {"resolved_frame_index": 300, "latent": still},                          # still deep in the timeline
        {"resolved_frame_index": 752, "latent": still},                          # last frame
        {"resolved_frame_index": 119, "latent": vae_like, "audio_latent": aud},  # 39-frame clip crossing window seams
    ]
    handler, res = run_case(753, 141, 51, keyframes=keyframes, with_masks=True, fuse="relative")
    per_window = {s: kfs for s, _, kfs in res[0]}
    head = per_window[0]
    assert any(k["resolved_frame_index"] == 0 and k["latent"] == (1, 24, 12, 6, 8) and k["audio_latent"] == (1, 32, 2, 65) for k in head)
    assert not any(k["resolved_frame_index"] > 140 for k in head)
    T = handler.last_total_tokens
    tail = per_window[T - handler.window_tokens]
    f0 = G.token_frame_offset(T - handler.window_tokens)
    assert any(k["resolved_frame_index"] == 752 - f0 for k in tail)
    # the still at 300 must appear in every window whose frame range contains it, with a local index
    for s, _, kfs in res[0]:
        f0 = G.token_frame_offset(s)
        f1 = f0 + 141
        has = [k for k in kfs if k["latent"] == (1, 24, 1, 6, 8) and k["resolved_frame_index"] == 300 - f0]
        assert bool(has) == (f0 <= 300 < f1), (s, f0, f1, kfs)
    # the crossing clip: whole-in windows keep 12 tokens, partial windows get 2 or 7 tokens or drop it
    for s, _, kfs in res[0]:
        f0 = G.token_frame_offset(s)
        for k in kfs:
            if k["latent"] and k["latent"][2] in (2, 7):
                assert 0 <= k["resolved_frame_index"]
        if f0 <= 119 and 119 + 39 <= f0 + 141:
            assert any(k["latent"] == (1, 24, 12, 6, 8) and k["resolved_frame_index"] == 119 - f0 for k in kfs)


def test_refs_and_split_conds():
    refs = [{"kind": "image", "latent_h": 6, "latent_w": 8, "latent": torch.randn(1, 24, 1, 6, 8)}]
    handler, res = run_case(345, 141, 51, keyframes=[], refs=refs, fuse="overlap-linear", steps=(0,))
    assert len(res[0]) >= 3
    # origin positioning and static windows still available
    handler, res = run_case(345, 141, 51, keyframes=[], refs=refs, steps=(0, 1), absolute=False, phase_alternate=False)
    assert [s for s, _, _ in res[0]] == [0, 15, 30, 45, 60] == [s for s, _, _ in res[1]]


def test_freenoise_shapes():
    conds, latent_shapes, _ = build_conds(345, 6, 8, 16, [])
    x, _ = comfy.utils.pack_latents([torch.randn(latent_shapes[0]), torch.randn(latent_shapes[1])])
    handler = H3ContextHandler(G.frames_to_tokens(141), 15)
    out = handler._apply_freenoise(x.clone(), conds, seed=1)
    assert out.shape == x.shape and not torch.equal(out, x)


if __name__ == "__main__":
    test_identity_and_phase()
    test_keyframes_remap()
    test_refs_and_split_conds()
    test_freenoise_shapes()
    print("handler tests ok")
