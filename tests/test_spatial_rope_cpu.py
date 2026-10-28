"""Native-core CPU checks, called by tests/run_tests.py. No checkpoints needed."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

import comfy.ops
import comfy.model_patcher
from comfy.ldm.minimax import model as H
from h3_drakennodes_pkg import spatial_rope as S


class SpatialRoPETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(143)
        self.diffusion = H.MiniMaxH3Model(
            hidden_size=12, num_layers=2, token_refiner_num_layers=0,
            num_attention_heads=2, attention_head_dim=6, ffn_hidden_size=16,
            text_dim=12, timestep_input_dim=4, time_embed_hidden_size=12,
            time_embed_dim=6, rope_inv_freq_len=1, dtype=torch.float32,
            operations=comfy.ops.disable_weight_init)
        for p in self.diffusion.parameters():
            nn.init.normal_(p, std=0.3)
        self.diffusion.rope.inv_freq.fill_(0.3)
        base = nn.Module()
        base.diffusion_model = self.diffusion
        self.model = comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))

    def test_layout_scopes_and_cached_layout_isolation(self):
        refs = [{"kind": "image", "latent_h": 4, "latent_w": 6},
                {"kind": "video_audio", "latent_h": 2, "latent_w": 4,
                 "latent_t": 2, "ref_audio_t": 3}]
        layout = H.PackedLayout(3, 2, 4, 6, 3,
            keyframes=[{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 6)}], refs=refs)
        # Emulate the context handler's absolute window-time offset.
        for a, b, kind in layout.segments:
            if kind in ("video", "audio", "cond"):
                layout.position_ids[a:b, 0] += 85
        before = layout.position_ids.clone()
        for scope in S.SCOPES:
            for _ in range(2):
                out = S.scale_layout(layout, 0.85, scope)
                self.assertIsNot(out, layout)
                torch.testing.assert_close(out.position_ids[:, 0], before[:, 0], rtol=0, atol=0)
                for a, b, kind in layout.segments:
                    scaled = kind == "video" or (scope != "target_only" and kind == "cond") or (scope == "all_visual" and kind == "ref_img")
                    expected = 16 + 0.85 * (before[a:b, 1:] - 16) if scaled else before[a:b, 1:]
                    torch.testing.assert_close(out.position_ids[a:b, 1:], expected, rtol=0, atol=0)
                torch.testing.assert_close(layout.position_ids, before, rtol=0, atol=0)
                self.assertIs(out.img_update, layout.img_update)
                self.assertEqual(out.seq_len, layout.seq_len)
        self.assertIs(S.scale_layout(layout, 1, "all_visual"), layout)

    def test_ramp_uses_sigma_schedule_without_call_counter(self):
        options = {"sample_sigmas": torch.tensor([1., 0.8, 0.4, 0.1, 0.])}
        p = S.SpatialRoPEPatch(0.8, "smooth_ramp", 1/3, 1, "all_visual")
        self.assertEqual(p.scale_at(torch.tensor([1000.]), options), 1)
        self.assertAlmostEqual(p.scale_at(torch.tensor([800.]), options), 1)
        self.assertAlmostEqual(p.scale_at(torch.tensor([400.]), options), 0.9)
        self.assertAlmostEqual(p.scale_at(torch.tensor([100.]), options), 0.8)
        # Same sigma yields the same scale for CFG branches/windows/repeated calls.
        for _ in range(3):
            self.assertAlmostEqual(p.scale_at(torch.tensor([400.]), options), 0.9)
        self.assertAlmostEqual(S.sampling_progress(torch.tensor([600.]), options["sample_sigmas"]), 0.5)
        self.assertEqual(S.sampling_progress(torch.tensor([1000.]), torch.tensor([1., 0.])), 0)
        self.assertEqual(S.sampling_progress(torch.tensor([100.]), torch.tensor([0.8, 0.4, 0.1])), 1)
        self.assertEqual(S.SpatialRoPEPatch(0.85, "static", 0.4, 0.9, "all_visual").scale_at(None, {}), 0.85)
        with self.assertRaises(ValueError):
            p.scale_at(torch.tensor([400.]), {})
        for schedule in (torch.tensor([1., float("nan"), 0.]), torch.tensor([0.1, 0.5, 0.])):
            with self.assertRaises(ValueError):
                S.sampling_progress(torch.tensor([400.]), schedule)

    def test_wrapper_rebuild_padding_and_failure_cleanup(self):
        x = [torch.zeros(1, 24, 2, 3, 5), torch.zeros(1, 32, 2, 3)]
        context = torch.zeros(1, 3, 12)
        stale = H.PackedLayout(1, 1, 2, 2, 1)
        payload = {"layout": stale, "seed": 123}
        before = stale.position_ids.clone()
        options = {"arbitrary_option": 7}
        outer = self
        class Executor:
            class_obj = outer.diffusion
            def __call__(self, x, timestep, context, opts, **kw):
                layout = kw["minimax_payload"]["layout"]
                outer.assertEqual(layout.signature, (3, 2, 4, 6, 3))
                outer.assertEqual(kw["minimax_payload"]["seed"], 123)
                outer.assertEqual(kw["denoise_mask"], "sentinel")
                opts["minimax_h3_layout"] = layout
                raise RuntimeError("test failure")
        p = S.SpatialRoPEPatch(0.85, "static", 0.4, 0.9, "all_visual")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            p.wrap(Executor(), x, torch.tensor([500.]), context, options,
                   minimax_payload=payload, denoise_mask="sentinel")
        self.assertIs(payload["layout"], stale)
        self.assertEqual(options, {"arbitrary_option": 7})
        torch.testing.assert_close(stale.position_ids, before, rtol=0, atol=0)

    def test_validation_branch_isolation_and_conflicts(self):
        zero = S.patch_model(self.model, 1)
        self.assertFalse(zero.wrappers)
        active = S.patch_model(self.model, 0.85)
        self.assertFalse(self.model.wrappers)
        with self.assertRaisesRegex(ValueError, "already applied"):
            S.patch_model(active, 0.9)
        for kwargs in ({"spatial_scale": float("nan")}, {"spatial_scale": 0},
                       {"mode": "unknown"}, {"reference_scope": "unknown"},
                       {"ramp_start": 0.9, "ramp_end": 0.4}):
            with self.assertRaises(ValueError):
                S.patch_model(self.model, **kwargs)
        memory = self.model.clone()
        memory.add_wrapper_with_key("diffusion_model", "DrakenH3MotionMemory", lambda *a, **k: None)
        with self.assertRaisesRegex(ValueError, "Motion Memory"):
            S.patch_model(memory, 0.85)
        # Detect memory added AFTER this node too, before either wrapper does work.
        wrappers = {"wrappers": memory.wrappers}
        with self.assertRaisesRegex(ValueError, "Motion Memory"):
            S.SpatialRoPEPatch(0.85, "static", 0.4, 0.9, "all_visual").wrap(None, None, None, None, wrappers)
        from h3_drakennodes_pkg.nodes import NODE_CLASS_MAPPINGS
        self.assertIs(NODE_CLASS_MAPPINGS["DrakenH3SpatialRoPE"], S.H3SpatialRoPE)
        node_model, info = S.H3SpatialRoPE().apply(self.model)
        self.assertFalse(node_model.wrappers)
        self.assertIn("bypass", info)
        self.assertEqual(S.H3SpatialRoPE.INPUT_TYPES()["required"]["reference_scope"][1]["default"], "target_only")

    def test_native_forward_baseline_and_active_scaling(self):
        x = [torch.randn(1, 24, 2, 4, 6), torch.randn(1, 32, 2, 3)]
        t, context = torch.tensor([500.]), torch.randn(1, 3, 12)
        image = {"kind": "image", "latent_h": 2, "latent_w": 2}
        image_latent = torch.randn(1, 24, 1, 2, 2)
        guide = {"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 4, 6),
                 "audio_latent": torch.randn(1, 32, 2, 2)}
        layout = H.PackedLayout(3, 2, 4, 6, 3, keyframes=[guide], refs=[image])
        payload = {"refs": [image], "keyframes": [guide], "layout": layout,
                   "cond_video_latents": [guide["latent"], image_latent],
                   "cond_audio_latents": [guide["audio_latent"]]}
        saved = layout.position_ids.clone()

        def rms_rope(q, k, table, qw, kw, epsilon, rot_dim):
            # CPU equivalent of the optional compiled split-half rotation.
            def rotate(z, weight):
                z = torch.nn.functional.rms_norm(z[0], (z.shape[-1],), weight, epsilon)
                half = rot_dim // 2
                a, b = z[..., :half], z[..., half:rot_dim]
                cos, sin = table[0, :, :, :, 0, 0], table[0, :, :, :, 1, 0]
                return torch.cat((a*cos-b*sin, a*sin+b*cos, z[..., rot_dim:]), -1)[None]
            q.copy_(rotate(q, qw))
            k.copy_(rotate(k, kw))

        quant_type = comfy.ops.QuantizedTensor if isinstance(comfy.ops.QuantizedTensor, type) else type("UnusedQuantizedTensor", (), {})
        def run(model, timestep=t, **more):
            options = copy.copy(model.model_options["transformer_options"])
            options.update(more)
            options["wrappers"] = model.wrappers
            return self.diffusion(x, timestep, context, transformer_options=options, minimax_payload=payload)
        with torch.no_grad(), patch.object(comfy.ops, "QuantizedTensor", quant_type), patch.object(H.comfy.quant_ops.ck, "rms_rope_split_half_", side_effect=rms_rope):
            baseline = run(self.model)
            bypass = run(S.patch_model(self.model, 1))
            for a, b in zip(baseline, bypass):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            active = S.patch_model(self.model, 0.85)
            with patch.object(self.diffusion, "rope_freqs", wraps=self.diffusion.rope_freqs) as rope:
                out = run(active)
                self.assertEqual(rope.call_count, 1)
                torch.testing.assert_close(rope.call_args.args[0], S.scale_layout(layout, 0.85, "target_only").position_ids)
            repeat = run(active)
            self.assertFalse(torch.allclose(baseline[0], out[0], rtol=1e-6, atol=1e-7))
            for a, b, original in zip(out, repeat, x):
                self.assertEqual(a.shape, original.shape)
                self.assertTrue(torch.isfinite(a).all())
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Native branch is unchanged after active inference.
            for a, b in zip(baseline, run(self.model)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            ramp = S.patch_model(self.model, 0.85, "smooth_ramp")
            schedule = torch.tensor([1., 0.5, 0.1, 0.])
            for a, b in zip(run(self.model, timestep=torch.tensor([1000.])),
                            run(ramp, timestep=torch.tensor([1000.]), sample_sigmas=schedule)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(run(active, timestep=torch.tensor([100.])),
                            run(ramp, timestep=torch.tensor([100.]), sample_sigmas=schedule)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(layout.position_ids, saved, rtol=0, atol=0)

    def test_vae_workflow_contract(self):
        import json
        from pathlib import Path
        import nodes
        from comfy_extras import nodes_video
        classes = dict(nodes.NODE_CLASS_MAPPINGS)
        for cls in (nodes_video.LoadVideo, nodes_video.GetVideoComponents):
            classes[cls.define_schema().node_id] = cls
        path = Path(__file__).resolve().parents[1]/"example_workflows/h3_vae_roundtrip_api.json"
        graph = json.loads(path.read_text())
        schemas = {}
        for key, node in graph.items():
            cls = classes[node["class_type"]]
            if hasattr(cls, "GET_NODE_INFO_V1"):
                info = cls.GET_NODE_INFO_V1()
                inputs, outputs = info["input"], info["output"]
            else:
                inputs, outputs = cls.INPUT_TYPES(), cls.RETURN_TYPES
            self.assertTrue(set(inputs.get("required", {})) <= set(node["inputs"]))
            schemas[key] = {**inputs.get("required", {}), **inputs.get("optional", {})}, outputs
        for key, node in graph.items():
            for name, value in node["inputs"].items():
                self.assertIn(name, schemas[key][0])
                if isinstance(value, list):
                    self.assertEqual(schemas[value[0]][1][value[1]], schemas[key][0][name][0])


def test_spatial_rope():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SpatialRoPETests))
    assert result.wasSuccessful(), "Spatial RoPE CPU checks failed"
