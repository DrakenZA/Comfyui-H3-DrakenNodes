"""Halo context, seam probe, guide blocks starting before a window, branch isolation, split conds, audio lock."""
import torch

import comfy.utils

from h3_drakennodes_pkg import h3_grid as G
from h3_drakennodes_pkg.handler import H3ContextHandler, VIDEO_DIM
from h3_drakennodes_pkg.nodes import H3AudioLock, H3ContextWindows, H3LongAVLatent, H3WindowPlan
from test_handler_cpu import FakeModel, build_conds
from test_nodes_cpu import FakeAudioVAE, FakeModelPatcher


def _run(handler, conds, x_in, steps, hook):
    fake = FakeModel()
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    model_options = {"transformer_options": {"sample_sigmas": sigmas}}

    def calc(model, sub_conds, sub_x, sub_t, opts):
        outs = []
        for cond in sub_conds:
            mc = cond[0]["model_conds"]
            hook(opts["transformer_options"]["context_window"], mc["minimax_payload"].cond, mc)
            outs.append(-sub_x)
        return outs

    outs = []
    for step in range(steps):
        out = handler.execute(calc, fake, conds, x_in, sigmas[step:step + 1], model_options)
        assert torch.allclose(out[0], -x_in, atol=1e-5)
        outs.append(out[0])
    return outs


def test_halo_and_probe():
    _halo_case(6, 8)
    _halo_case(6, 7)   # odd latent width (e.g. a 464 px canvas): halo rows must be padded like the target


def _halo_case(lat_h, lat_w):
    conds, shapes, _ = build_conds(345, lat_h, lat_w, 16, [])
    x_in, _ = comfy.utils.pack_latents([torch.randn(shapes[0]), torch.randn(shapes[1])])
    L, S = G.frames_to_tokens(141), 15
    h = H3ContextHandler(L, S, probe_seams=True, halo_tokens=5, halo_start_percent=0.5)
    seen = []
    ph, pw = (lat_h + 1) // 2 * 2, (lat_w + 1) // 2 * 2

    def hook(win, pl, mc):
        s = win.index_list[0]
        halos = [k for k in pl["keyframes"] if k.get("h3_halo")]
        seen.append((h._step, s, [(k["resolved_frame_index"], tuple(k["latent"].shape)) for k in halos]))
        lay = pl["layout"]
        n_cond = sum(b - a for a, b, kind in lay.segments if kind == "cond")
        assert n_cond == sum(k["latent"].shape[2] for k in pl["keyframes"]) * (ph // 2) * (pw // 2)
        assert len(pl["cond_video_latents"]) == len(pl["keyframes"])
        for k in halos:  # what the DiT will patchify: must reshape to (t, ph/2, 2, pw/2, 2)
            k["latent"].reshape(1, 24, k["latent"].shape[2], 1, ph // 2, 2, pw // 2, 2)

    _run(h, conds, x_in, steps=4, hook=hook)
    T = h.last_total_tokens
    for step, s, halos in seen:
        if step < 2:  # 0/4, 1/4 < 0.5 -> no halo; step 2 is the first with 50%
            assert halos == [], (step, halos)
        else:
            expect = []
            if s > 0:
                expect.append((G.token_frame_offset(s - 5) - G.token_frame_offset(s), (1, 24, 5, ph, pw)))
            if s + L < T:
                expect.append((G.tokens_to_frames(L), (1, 24, 5, ph, pw)))
            assert halos == expect, (step, s, halos, expect)
            # halo latents come from the previous step's fused x0 (= -x_in video slice)
    assert h._last_x0_video is not None and h._last_x0_video.shape == shapes[0]
    assert torch.allclose(h._last_x0_video, -comfy.utils.unpack_latents(x_in, shapes)[0], atol=1e-5)
    # probe: identical windows (all -x) -> zero disagreement, one record per step
    assert len(h.seam_stats) == 4 and all(abs(m) < 1e-6 for _, m, _, _ in h.seam_stats)
    assert h.seam_stats[0][3] == len(G.plan_window_starts(T, L, S)) - 1
    # a new run (step 0) resets state
    _run(h, conds, x_in, steps=1, hook=lambda *a: None)
    assert len(h.seam_stats) == 1


def test_guide_starting_before_window():
    # a 90-frame block anchored at frame 34 spans windows [0,141) and [51,192) and [102,243)
    block = torch.randn(1, 24, G.frames_to_tokens(90), 6, 8)
    kfs = [{"resolved_frame_index": 34, "latent": block}]
    conds, shapes, _ = build_conds(345, 6, 8, 16, kfs)
    x_in, _ = comfy.utils.pack_latents([torch.randn(shapes[0]), torch.randn(shapes[1])])
    h = H3ContextHandler(G.frames_to_tokens(141), 15)
    got = {}

    def hook(win, pl, mc):
        got[win.index_list[0]] = [(k["resolved_frame_index"], k["latent"].shape[2]) for k in pl["keyframes"]]

    _run(h, conds, x_in, steps=1, hook=hook)
    assert got[0] == [(34, 27)]                       # fits entirely: 34+90 <= 141
    f0 = G.token_frame_offset(15)                     # window at frame 51: block [34,124) starts before it
    assert got[15] == [(51 - f0 + 0, 22)] or got[15] == [(0, 22)]
    local, n = got[15][0]
    assert local == 0 and n == 22                     # first cycle inside starts at frame 51 (34+17), 22 tokens = 73 frames
    f0 = G.token_frame_offset(30)                     # window at frame 102: block covers [102,124) -> 1 cycle + tail = 7 tokens
    assert got[30] == [(0, 7)], got[30]
    assert got[45] == []                              # window at 153: no overlap


def test_branch_isolation_and_split_conds():
    m = FakeModelPatcher()
    a, _ = H3ContextWindows().apply(m, 141, 51, "pyramid", True, False, False, False)
    b, _ = H3ContextWindows().apply(m, 141, 51, "pyramid", False, True, False, True, halo_frames=34, probe_seams=True)
    ha, hb = a.model_options["context_handler"], b.model_options["context_handler"]
    assert ha is not hb and ha.absolute_positions and not hb.absolute_positions
    assert hb.halo_tokens == 10 and ha.halo_tokens == 0 and hb.probe_seams and not ha.probe_seams
    assert "context_handler" not in m.model_options
    # two conditionings with different text lengths, one window each region
    conds1, shapes, _ = build_conds(345, 6, 8, 16, [])
    conds2, _, _ = build_conds(345, 6, 8, 24, [])
    conds = [conds1[0] + conds2[0]]
    x_in, _ = comfy.utils.pack_latents([torch.randn(shapes[0]), torch.randn(shapes[1])])
    h = H3ContextHandler(G.frames_to_tokens(141), 15, split_conds_to_windows=True)
    picks = []

    def hook(win, pl, mc):
        picks.append((win.index_list[0], pl["layout"].signature[0], mc["c_crossattn"].cond.shape[1]))

    _run(h, conds, x_in, steps=1, hook=hook)
    assert [p[0] for p in picks] == [0, 15, 30, 45, 60]
    assert all(p[1] == p[2] for p in picks)           # each window's layout matches the cond it received
    assert {p[1] for p in picks} == {16, 24}          # both regions used


def test_margins():
    conds, shapes, _ = build_conds(345, 6, 8, 16, [])
    x_in, _ = comfy.utils.pack_latents([torch.randn(shapes[0]), torch.randn(shapes[1])])
    L, S, M = G.frames_to_tokens(141), 15, 10
    for fuse in ("pyramid", "relative", "overlap-linear"):
        h = H3ContextHandler(L, S, fuse_method=fuse, probe_seams=True, margin_tokens=M)
        seen = []

        def hook(win, pl, mc):
            s, n = win.index_list[0], len(win.index_list)
            off, inner_n = win.h3_inner
            aw = win.get_window_for_modality(1)
            seen.append((s, n, off, inner_n, aw.h3_inner, len(aw.index_list)))
            assert n % 5 == 2 and s % 5 == 0 and inner_n == L
            assert pl["layout"].signature[1] == n

        _run(h, conds, x_in, steps=2, hook=hook)   # identity through margins for every fuse method
        T = h.last_total_tokens
        starts = G.plan_window_starts(T, L, S)
        per_step = len(starts)
        assert len(seen) == 2 * per_step
        for (s, n, off, inner_n, a_inner, a_len), s_inner in zip(seen[:per_step], starts):
            assert s == max(0, s_inner - M) and off == s_inner - s
            assert n == min(T, s_inner + L + M) - s
            assert 0 <= a_inner[0] and a_inner[0] + a_inner[1] <= a_len
            assert a_inner[1] == G.audio_ticks_for_frames(141)
        assert h.seam_stats and h.seam_stats[0][3] == per_step - 1
    # margins are no-ops on the schedule: same inner starts as without
    h0 = H3ContextHandler(L, S)
    assert [w.index_list[0] for w in h0.get_context_windows(None, torch.zeros(1, 24, 102, 6, 8), {})] == starts


def test_audio_lock_and_plan():
    latent, used, _ = H3LongAVLatent().build(160, 96, 141, True, 39, 0)
    audio = {"waveform": torch.randn(1, 2, 48000 * 20), "sample_rate": 48000}
    out, info = H3AudioLock().lock(latent, FakeAudioVAE(), audio, 3.0, 1.0)
    v, a = out["samples"].unbind()
    mv, ma = out["noise_mask"].unbind()
    assert a.shape == (1, 32, 2, 235) and torch.all(a == 0.25) and torch.all(ma == 0.0) and torch.all(mv == 1.0)
    out, info = H3AudioLock().lock(latent, FakeAudioVAE(), audio, 19.0, 0.5)
    assert "padded" in info and torch.all(out["noise_mask"].unbind()[1] == 0.5)
    text, n, length = H3WindowPlan().plan("windows -> length", 719, 6, 141, 51)["result"]
    assert n == 6 and length == G.align_frames_av_exact_up(G.tokens_to_frames(42 + 5 * 15))
    assert G.is_av_exact(length)
