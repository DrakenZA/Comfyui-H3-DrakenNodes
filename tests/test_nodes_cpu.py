"""Node-level CPU tests with fake VAEs and a fake ModelPatcher."""
import torch

from h3_drakennodes_pkg import h3_grid as G
from h3_drakennodes_pkg.nodes import H3ContextWindows, H3LongAVLatent, H3TrimPrefixAV, H3WindowPlan


class FakeVideoVAE:
    def encode(self, images):  # [N,H,W,3] -> [1,24,tok,H/16,W/16]
        n, h, w, _ = images.shape
        tok = G.frames_to_tokens(n)
        return torch.full((1, 24, tok, h // 16, w // 16), 0.5)


class FakeAudioVAE:
    audio_sample_rate = 32000

    def encode(self, wav):  # [1, L, 2] -> [1,32,2,T] at 40 Hz
        t = int(round(wav.shape[1] / 32000 * 40))
        return torch.full((1, 32, 2, t), 0.25)


class FakeModelPatcher:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.wrappers = []

    def clone(self):
        c = FakeModelPatcher()
        c.model_options = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.model_options.items()}
        return c

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.append((wrapper_type, key, wrapper))


def test_long_latent_and_trim():
    node = H3LongAVLatent()
    frames = torch.rand(100, 96, 160, 3)
    audio = {"waveform": torch.randn(1, 2, 48000 * 5), "sample_rate": 48000}
    latent, used, info = node.build(160, 96, 719, True, 39, 2, vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                                    prefix_frames=frames, prefix_audio=audio)
    v, a = latent["samples"].unbind()
    assert v.shape == (1, 24, G.frames_to_tokens(753), 6, 10)
    assert a.shape == (1, 32, 2, G.audio_ticks_for_frames(753))
    assert used == 39
    vm, am = latent["noise_mask"].unbind()
    assert torch.all(vm[:, :, :12] == 0) and torch.all(vm[:, :, 14:] == 1)
    assert abs(float(vm[0, 0, 12, 0, 0]) - 1 / 3) < 1e-6 and abs(float(vm[0, 0, 13, 0, 0]) - 2 / 3) < 1e-6
    assert torch.all(am[..., :65] == 0) and torch.all(v[:, :, :12] == 0.5) and torch.all(a[..., :65] == 0.25)
    print(info)
    # no footage -> no mask
    latent2, used2, _ = node.build(160, 96, 141, False, 39, 0)
    assert used2 == 0 and "noise_mask" not in latent2
    assert latent2["samples"].unbind()[0].shape[2] == 42
    imgs, aud = H3TrimPrefixAV().trim(39, images=torch.zeros(753, 8, 8, 3), audio={"waveform": torch.zeros(1, 2, 32000 * 10), "sample_rate": 32000})
    assert imgs.shape[0] == 753 - 39 and aud["waveform"].shape[-1] == 32000 * 10 - 52000
    text, n, _ = H3WindowPlan().plan("length -> windows", 753, 6, 141, 51)["result"]
    assert n == len(G.plan_window_starts(222, 42, 15))
    print(text)


def test_context_windows_node():
    m = FakeModelPatcher()
    out, info = H3ContextWindows().apply(m, 141, 51, "pyramid", True, False, True, False, expected_total_frames=753)
    h = out.model_options["context_handler"]
    assert h.window_tokens == 42 and h.stride_tokens == 15 and h.context_overlap == 27
    assert any(k == "H3ContextWindows_prepare_sampling" for _, k, _ in out.wrappers)
    assert "context_handler" not in m.model_options
    # memory wrapper scales packed size by window/total
    wrapper = [w for _, k, w in out.wrappers if k == "H3ContextWindows_prepare_sampling"][0]
    captured = {}
    wrapper(lambda model, shape, conds: captured.setdefault("shape", shape), None, [1, 1, 222_000], None)
    assert captured["shape"][-1] == int(222_000 * 42 / 222)
    print(info)
    try:
        H3ContextWindows().apply(m, 39, 51, "pyramid", True, False, False, False)
        raise AssertionError("stride >= window must fail")
    except ValueError:
        pass
