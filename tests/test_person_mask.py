"""Person selection, video ordering, alpha compositing, and model isolation contracts."""

import importlib.util
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load("draken_person_mask_test", "person_mask.py")
backend = load("draken_person_backend_test", "person_backend.py")


class PersonMaskTests(unittest.TestCase):
    def test_preview_selection_and_frame_contract(self):
        images = torch.full((3, 40, 60, 3), 0.2)
        with tempfile.TemporaryDirectory() as directory:
            folder_paths = types.SimpleNamespace(get_temp_directory=lambda: directory)
            with mock.patch.dict("sys.modules", {"folder_paths": folder_paths}):
                selected = nodes.PersonSelection().select(images, 1, '{"box":[0.2,0.1,0.8,0.9],"points":[[0.5,0.5,1]]}')
            preview = selected["ui"]["person_selector"][0]
            self.assertTrue((pathlib.Path(directory) / preview["subfolder"] / preview["filename"]).is_file())
            self.assertEqual(preview["frame_index"], 1)
            person = selected["result"][0]
            fake = types.SimpleNamespace(segment_video=mock.Mock(return_value=torch.ones(3, 40, 60)))
            with mock.patch.object(nodes, "_backend", return_value=fake):
                mask, = nodes.PersonVideoMask().track(images, person,
                    corrections='[{"frame_index":2,"points":[[0.8,0.7,0]]}]')
            self.assertEqual(tuple(mask.shape), (3, 40, 60))
            prompts = fake.segment_video.call_args.args[1]
            self.assertEqual([p["frame_index"] for p in prompts], [1, 2])
            with self.assertRaisesRegex(ValueError, "same frame count"):
                nodes.PersonVideoMask().track(images[:2], person)
            person["prompt"] = {"frame_index": 1, "points": [], "box": None}
            with self.assertRaisesRegex(ValueError, "Select a person"):
                nodes.PersonVideoMask().track(images, person)

    def test_invalid_prompts_fail_before_loading_models(self):
        for data in ({"box": [0.8, 0, 0.2, 1]}, {"points": [[float("nan"), 0.5, 1]]},
                     {"points": [[0.5, 1.1, 1]]}, {"points": [[0.5, 0.5, 3]]}):
            with self.assertRaises(ValueError):
                nodes._prompt(data, 0)
        with self.assertRaisesRegex(ValueError, "zero-based"):
            nodes._frame_index(3, 3)

    def test_color_and_soft_alpha_do_not_modify_sources(self):
        images = torch.full((2, 3, 4, 3), 0.2)
        mask = torch.zeros(2, 3, 4)
        mask[0, 1, 1] = 1
        mask[0, 1, 2] = 0.5
        original = images.clone()
        preview, rgba, transparency = nodes.PersonMaskOverlay().apply(images, mask, opacity=1)
        torch.testing.assert_close(rgba[0, 1, 1, :3], torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(rgba[0, 1, 2, :3], torch.tensor([0.6, 0.1, 0.1]))
        self.assertTrue(torch.all(rgba[..., 3] == 1))
        self.assertTrue(torch.all(transparency == 0))
        self.assertTrue(torch.equal(preview[1], images[1]))
        preview, rgba, transparency = nodes.PersonMaskOverlay().apply(images, mask, mode="transparent", opacity=0.5)
        self.assertEqual(float(rgba[0, 1, 1, 3]), 0.5)
        self.assertEqual(float(rgba[0, 1, 2, 3]), 0.75)
        torch.testing.assert_close(transparency, mask * 0.5)
        self.assertTrue(torch.equal(rgba[..., :3], images))
        self.assertTrue(torch.equal(images, original))
        with self.assertRaisesRegex(ValueError, "match"):
            nodes.PersonMaskOverlay().apply(images, mask[:1])

    def test_existing_alpha_and_full_person_transparency(self):
        images = torch.full((1, 3, 4, 4), 0.2)
        mask = torch.zeros(1, 3, 4)
        mask[0, 1, 1] = 1
        _, rgba, transparency = nodes.PersonMaskOverlay().apply(images, mask, mode="transparent")
        self.assertEqual(float(rgba[0, 1, 1, 3]), 0)
        self.assertAlmostEqual(float(rgba[0, 0, 0, 3]), 0.2)
        self.assertEqual(float(transparency[0, 1, 1]), 1)
        _, rgba, _ = nodes.PersonMaskOverlay().apply(images, mask, opacity=0)
        self.assertTrue(torch.equal(rgba, images))

    def test_video_order_reverse_tracking_coordinates_and_fresh_state(self):
        class Predictor:
            def __init__(self):
                self.states = []
                self.inputs = []

            def init_state(self, frames, **kwargs):
                state = {"count": len(frames)}
                self.states.append(state)
                return state

            def add_new_points_or_box(self, state, **kwargs):
                self.inputs.append(kwargs)

            def propagate_in_video(self, state, start_frame_idx, reverse):
                order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, state["count"])
                for i in order:
                    logits = torch.full((1, 1, 4, 6), -1.)
                    logits[0, 0, i, i] = 1
                    yield i, [1], logits

        predictor = Predictor()
        images = torch.zeros(3, 4, 6, 3)
        prompt = {"frame_index": 1, "box": [0.25, 0.25, 0.75, 0.75], "points": [[0.5, 0.5, 1]]}
        progress = mock.Mock()
        mask = backend.track_masks(predictor, images, [prompt], progress=progress)
        self.assertEqual(progress.update.call_count, 3)
        self.assertEqual(predictor.inputs[0]["points"], [[3., 2.]])
        self.assertEqual(predictor.inputs[0]["box"], [1.5, 1., 4.5, 3.])
        self.assertEqual(float(mask.sum()), 3)
        for i in range(3):
            self.assertEqual(float(mask[i, i, i]), 1)
        backend.track_masks(predictor, images, [prompt])
        self.assertIsNot(predictor.states[0], predictor.states[1])

    def test_checksum_rejection_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "person_segmentation" / "edgetam.pt"
            path.parent.mkdir()
            path.write_bytes(b"bad checkpoint")
            with mock.patch.dict("sys.modules", {"folder_paths": types.SimpleNamespace(models_dir=directory)}):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    backend.ensure_checkpoint("EdgeTAM")
            self.assertEqual(path.read_bytes(), b"bad checkpoint")


if __name__ == "__main__":
    unittest.main()
