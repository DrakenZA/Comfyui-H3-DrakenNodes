"""MiniMax H3 temporal grid math.

H3 facts (from ComfyUI's comfy/ldm/minimax/model.py and comfy_extras/nodes_minimax_h3.py):
  * 24 fps pixel video; the video VAE packs frames into latent tokens with the
    repeating pattern FRAME_PER_TOKEN = (1, 4, 4, 4, 4): 5 tokens per 17 frames.
  * a valid clip is 17k + 5 frames -> 5k + 2 latent tokens.
  * audio latent runs at 40 Hz, i.e. 5/3 ticks per pixel frame (FRAME_RESCALE).
  * 3 frames == 5 ticks, so clip lengths with (17k+5) % 3 == 0 (k = 2, 5, 8, ...)
    have an integer tick count: 39, 90, 141, 192, 243, 294, 345 frames ...
    Those are the "AV-exact" lengths.

A context window over the long latent must:
  * start on a 5-token boundary (a 17-frame boundary) so the (1,4,4,4,4) phase
    of the window matches what the model expects for a standalone clip,
  * have 5j + 2 tokens (a valid clip length),
  * ideally start on a 15-token boundary (51 frames = 85 ticks) so its audio
    window starts on an integer tick.
"""

FPS = 24
AUDIO_HZ = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
TOKENS_PER_CYCLE = 5
FRAMES_PER_CYCLE = 17
AV_EXACT_TOKEN_STRIDE = 15   # 51 frames == 85 audio ticks
AV_EXACT_FRAME_STRIDE = 51
_CYCLE_OFFSETS = (0, 1, 5, 9, 13)


def is_valid_frames(n: int) -> bool:
    return n >= 5 and (n - 5) % FRAMES_PER_CYCLE == 0


def align_frames_up(n: int) -> int:
    n = max(5, int(n))
    while (n - 5) % FRAMES_PER_CYCLE:
        n += 1
    return n


def align_frames_nearest(n: int) -> int:
    n = max(5, int(n))
    up = align_frames_up(n)
    down = up - FRAMES_PER_CYCLE
    if down < 5:
        return up
    return down if (n - down) < (up - n) else up


def is_av_exact(frames: int) -> bool:
    return (frames * 5) % 3 == 0


def align_frames_av_exact_up(n: int) -> int:
    """Smallest 17k+5 >= n whose audio tick count is an integer (k % 3 == 2)."""
    n = align_frames_up(n)
    while not is_av_exact(n):
        n += FRAMES_PER_CYCLE
    return n


def frames_to_tokens(frames: int) -> int:
    if not is_valid_frames(frames):
        raise ValueError(f"H3 frame count must be 17k+5, got {frames}")
    return 2 if frames <= 5 else ((frames - 5) // FRAMES_PER_CYCLE) * TOKENS_PER_CYCLE + 2


def tokens_to_frames(tokens: int) -> int:
    return sum(FRAME_PER_TOKEN[k % TOKENS_PER_CYCLE] for k in range(int(tokens)))


def token_frame_offset(token_index: int) -> int:
    """Pixel frame index at which latent token `token_index` starts."""
    cycles, rem = divmod(int(token_index), TOKENS_PER_CYCLE)
    return cycles * FRAMES_PER_CYCLE + _CYCLE_OFFSETS[rem]


def audio_ticks_for_frames(frames: int) -> int:
    return round(frames * AUDIO_HZ / FPS)


def audio_tick_at_frame(frame: int) -> int:
    return round(frame * AUDIO_HZ / FPS)


def is_phase_aligned_window(start_tok: int, len_tok: int) -> bool:
    return start_tok % TOKENS_PER_CYCLE == 0 and len_tok % TOKENS_PER_CYCLE == 2


def snap_window_tokens(frames: int) -> int:
    return frames_to_tokens(align_frames_nearest(frames))


def snap_stride_tokens(frames: int) -> int:
    cycles = max(1, round(int(frames) / FRAMES_PER_CYCLE))
    return cycles * TOKENS_PER_CYCLE


def plan_window_starts(total_tokens: int, window_tokens: int, stride_tokens: int, shift_tokens: int = 0) -> list[int]:
    """Phase-aligned window start tokens covering [0, total_tokens).

    Always includes the first window (start 0) and the last window (start
    total - window). Interior windows start at shift + k*stride. All starts are
    multiples of 5 tokens.
    """
    T, L, S = int(total_tokens), int(window_tokens), int(stride_tokens)
    if L % TOKENS_PER_CYCLE != 2:
        raise ValueError(f"window_tokens must be 5j+2, got {L}")
    if T % TOKENS_PER_CYCLE != 2:
        raise ValueError(f"total_tokens must be 5k+2, got {T}")
    if S % TOKENS_PER_CYCLE or S <= 0:
        raise ValueError(f"stride_tokens must be a positive multiple of 5, got {S}")
    if L >= T:
        return [0]
    last = T - L
    shift = (int(shift_tokens) // TOKENS_PER_CYCLE) * TOKENS_PER_CYCLE
    shift %= S
    starts = {0, last}
    s = shift
    while s <= last:
        starts.add(s)
        s += S
    return sorted(starts)


def window_audio_range(start_tok: int, len_tok: int, total_tokens: int, total_audio_ticks: int) -> tuple[int, int]:
    """[tick_start, tick_stop) for a video token window. Exact when start is a 15-token multiple."""
    frames = tokens_to_frames(len_tok)
    ticks = audio_ticks_for_frames(frames)
    if start_tok + len_tok >= total_tokens:
        stop = int(total_audio_ticks)
        return max(0, stop - ticks), stop
    start = audio_tick_at_frame(token_frame_offset(start_tok))
    start = min(start, int(total_audio_ticks) - ticks)
    return start, start + ticks


def describe_plan(total_frames: int, window_frames: int, stride_frames: int) -> str:
    total_frames = align_frames_up(total_frames)
    T = frames_to_tokens(total_frames)
    L = snap_window_tokens(window_frames)
    S = snap_stride_tokens(stride_frames)
    Ta = audio_ticks_for_frames(total_frames)
    lines = [
        f"total: {total_frames} frames ({total_frames / FPS:.2f}s) = {T} tokens, {Ta} audio ticks"
        + ("" if is_av_exact(total_frames) else "  [audio grid inexact: pick 51n-12 frames]"),
        f"window: {tokens_to_frames(L)} frames = {L} tokens; stride: {S} tokens = {S // 5 * 17} frames; overlap: {L - S} tokens",
    ]
    if S % AV_EXACT_TOKEN_STRIDE:
        lines.append("stride is not a multiple of 51 frames: interior audio windows round to the nearest tick (<=17ms).")
    for phase in (0, 1):
        shift = (S // 2 // TOKENS_PER_CYCLE) * TOKENS_PER_CYCLE if phase else 0
        starts = plan_window_starts(T, L, S, shift)
        lines.append(f"phase {phase}: {len(starts)} windows/step -> " + ", ".join(
            f"[{s}-{s + L})" for s in starts))
    return "\n".join(lines)
