"""Pure-python checks of the H3 grid math against ComfyUI's own constants and known values."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from h3_drakennodes_pkg import h3_grid as G  # noqa: E402


def test_basic_grid():
    # values cross-checked with ComfyUI comfy_extras/nodes_minimax_h3.py temporal_shape()
    assert G.frames_to_tokens(124) == 37
    assert G.tokens_to_frames(37) == 124
    assert G.frames_to_tokens(5) == 2 and G.tokens_to_frames(2) == 5
    assert G.audio_ticks_for_frames(124) == 207
    # Continuum's self-test values
    assert G.frames_to_tokens(22) == 7 and G.frames_to_tokens(39) == 12 and G.frames_to_tokens(141) == 42
    assert G.audio_ticks_for_frames(141) == 235
    assert [G.token_frame_offset(i) for i in range(7)] == [0, 1, 5, 9, 13, 17, 18]
    for n in (39, 90, 141, 192, 243, 294, 345):
        assert G.is_valid_frames(n) and G.is_av_exact(n)
    for n in (22, 56, 73, 124):
        assert G.is_valid_frames(n) and not G.is_av_exact(n)
    assert G.align_frames_av_exact_up(720) == 753
    assert G.align_frames_up(720) == 736
    assert G.align_frames_nearest(130) == 124 and G.align_frames_nearest(135) == 141


def test_plan():
    T = G.frames_to_tokens(753)  # 222 tokens
    L = G.frames_to_tokens(141)  # 42
    S = 15
    starts = G.plan_window_starts(T, L, S)
    assert starts[0] == 0 and starts[-1] == T - L
    assert all(s % 5 == 0 for s in starts)
    assert all(starts[i + 1] - starts[i] <= S for i in range(len(starts) - 1))
    shifted = G.plan_window_starts(T, L, 30, 15)
    assert 15 in shifted and 0 in shifted and shifted[-1] == T - L
    # audio ranges: exact on the 15-token grid, last window pinned to the end
    Ta = G.audio_ticks_for_frames(753)
    for s in starts:
        a0, a1 = G.window_audio_range(s, L, T, Ta)
        assert a1 - a0 == 235
        if s % 15 == 0:
            assert a0 * 3 == G.token_frame_offset(s) * 5 or s + L == T
    assert G.window_audio_range(T - L, L, T, Ta)[1] == Ta
    print(G.describe_plan(753, 141, 51))


if __name__ == "__main__":
    test_basic_grid()
    test_plan()
    print("grid tests ok")
