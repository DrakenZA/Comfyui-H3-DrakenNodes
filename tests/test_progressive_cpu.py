"""Progressive (multi-stage SelfLift) sampler on CPU with a toy flow model through the real comfy sampling path."""
import torch

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.nested_tensor
import comfy.samplers
import comfy.supported_models_base

from h3_drakennodes_pkg import progressive as P
from h3_drakennodes_pkg.nodes import H3ProgressiveSampler


class ToyNet(torch.nn.Module):
    """Elementwise, resolution-agnostic 'velocity' predictor; works on the flat AV pack too."""

    def __init__(self, **kwargs):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.7))
        self.dtype = torch.float32

    def forward(self, x, timesteps, context=None, **kwargs):
        t = timesteps.reshape(-1, *([1] * (x.ndim - 1))) / 1000.0
        return torch.tanh(x) * self.w + 0.3 * t + 0.05 * context.mean()


class ToyConfig(comfy.supported_models_base.BASE):
    unet_config = {}
    latent_format = comfy.latent_formats.LatentFormat
    sampling_settings = {"shift": 1.0}

    def __init__(self):
        super().__init__({})
        self.latent_format = comfy.latent_formats.LatentFormat()
        self.latent_format.latent_channels = 4
        self.custom_operations = comfy.ops.disable_weight_init


def toy_model():
    import comfy.ops  # noqa: F401
    cfg = ToyConfig()
    model = comfy.model_base.BaseModel(cfg, model_type=comfy.model_base.ModelType.FLOW, unet_model=ToyNet)
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


class ToyVAE:
    """8x pixel VAE: decode = nearest upsample of the first 3 channels, encode = average pool back."""
    device = torch.device("cpu")
    vae_dtype = torch.float32

    def decode(self, z):
        if z.ndim == 5:
            b, c, t, h, w = z.shape
            z = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        img = torch.nn.functional.interpolate(z[:, :3].float(), scale_factor=8, mode="nearest")
        self.frames = z.shape[0]
        return img.movedim(1, -1).clamp(-3, 3)

    def encode(self, img):
        x = torch.nn.functional.avg_pool2d(img.movedim(-1, 1).float(), 8)
        x = torch.cat([x, x[:, :1]], dim=1)  # 4 channels
        if getattr(self, "video_t", None):
            n = x.shape[0] // self.video_t
            x = x.reshape(n, self.video_t, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        return x


def _cond():
    return [[torch.ones(1, 4, 8), {}]]


def _sampler(name):
    return comfy.samplers.sampler_object(name)



def test_plan_stages():
    plan = P.plan_stages(10, [(3, 0.5), (3, 0.75)], 32, 48)
    assert plan == [(0, 3, 16, 24), (3, 6, 24, 36), (6, 10, 32, 48)]
    for bad in ([(3, 0.75), (3, 0.5)], [(5, 0.5), (5, 0.75)], [(0, 0.5)], [(3, 1.0)]):
        try:
            P.plan_stages(10, bad, 32, 48)
        except ValueError:
            continue
        raise AssertionError(f"plan {bad} should be rejected")


def test_any_sampler_multi_stage_image():
    model, vae = toy_model(), ToyVAE()
    latent = {"samples": torch.zeros(1, 4, 32, 48)}
    sig = torch.linspace(0.95, 0.0, 11)
    sig[-1] = 0.0
    for name in ("euler", "euler_ancestral", "dpmpp_2m", "heun", "res_multistep", "dpmpp_2m_sde"):
        for stages in ([(4, 0.5)], [(3, 0.5), (3, 0.75)], [(2, 0.25), (2, 0.5), (2, 0.75)]):
            out = P.progressive_sample(model, _cond(), _cond(), vae, latent, _sampler(name), sig, 7, 1.0, stages,
                                       rho=0.6, w_min=1.0, w_max=1.0)["samples"]
            assert out.shape == (1, 4, 32, 48) and torch.isfinite(out).all(), (name, stages)


def test_noise_mask_keeps_content():
    model, vae = toy_model(), ToyVAE()
    src = torch.randn(1, 4, 32, 48)
    mask = torch.zeros(1, 32, 48)
    mask[:, :, 24:] = 1.0  # generate the right half, keep the left
    sig = torch.linspace(0.95, 0.0, 9)
    out = P.progressive_sample(model, _cond(), _cond(), vae, {"samples": src, "noise_mask": mask}, _sampler("dpmpp_2m"),
                               sig, 3, 1.0, [(2, 0.5), (2, 0.75)], rho=0.3)["samples"]
    assert torch.allclose(out[..., :20], src[..., :20], atol=1e-4)
    assert not torch.allclose(out[..., 28:], src[..., 28:], atol=1e-2)


def test_nested_av_stages_and_node():
    """Video [B,C,T,H,W] + audio [B,C,2,L] nested latent: audio carries the sampler state across stages."""
    model, vae = toy_model(), ToyVAE()
    vae.video_t = 3
    video = torch.zeros(1, 4, 3, 16, 24)
    audio = torch.zeros(1, 4, 2, 50)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    sig = torch.linspace(0.95, 0.0, 9)
    for name in ("euler", "dpmpp_2m", "res_multistep"):
        out = P.progressive_sample(model, _cond(), _cond(), vae, latent, _sampler(name), sig, 1, 1.0,
                                   [(2, 0.5), (2, 0.75)], rho=0.6, w_min=1.0, w_max=1.0)["samples"]
        v, a = out.unbind()
        assert v.shape == video.shape and a.shape == audio.shape and torch.isfinite(v).all() and torch.isfinite(a).all()
    # node front end: DynamicCombo values arrive as one dict
    stages = {"stages": "3", "stage_1_steps": 2, "stage_1_scale": 0.5, "stage_2_steps": 2, "stage_2_scale": 0.75}
    res = H3ProgressiveSampler.execute(model, _cond(), _cond(), vae, latent, _sampler("euler"), sig, 1, 1.0, stages,
                                       0.6, 1.0, 1.0, "nearest", "none", True)
    assert res.result[0]["samples"].unbind()[0].shape == video.shape
    schema = H3ProgressiveSampler.define_schema()
    combo = [i for i in schema.inputs if i.id == "stages"][0]
    assert [o.key for o in combo.options] == [str(n) for n in range(2, P.MAX_STAGES + 1)]
    assert len(combo.options[-1].inputs) == 2 * (P.MAX_STAGES - 1)


def test_euler_two_stage_equals_manual_selflift_transition():
    """With Euler, the generic boundary (noise z0 at s_next) equals SelfLift's re-noise at s_k + reused Euler step."""
    model = toy_model()
    ms = model.get_model_object("model_sampling")
    z0 = torch.randn(1, 4, 8, 8)
    n = torch.randn(1, 4, 8, 8)
    sk, sn = torch.tensor(0.6), torch.tensor(0.45)
    x = ms.noise_scaling(sk, n, z0)
    stepped = x + (x - z0) * ((sn - sk) / sk)
    assert torch.allclose(stepped, ms.noise_scaling(sn, n, z0), atol=1e-6)
