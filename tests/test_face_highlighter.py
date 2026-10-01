"""Face overlay geometry, blending, and frame alignment without model downloads."""

import importlib.util
import pathlib
import unittest

import torch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "face_highlighter.py"
SPEC = importlib.util.spec_from_file_location("draken_face_highlighter_test", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def record(mask, origin):
    return {"mask": torch.as_tensor(mask, dtype=torch.float32), "mask_origin": origin}


def face_data(images, frames):
    return {"schema_version": 1, "height": images.shape[1], "width": images.shape[2], "frames": frames}


class FaceHighlighterTests(unittest.TestCase):
    def setUp(self):
        self.node = module.FaceMaskHighlighter()

    def test_soft_edges_and_occlusion_hole(self):
        images = torch.full((1, 5, 6, 3), 0.2)
        crop = record([[1, 0.5, 1], [1, 0, 1]], (2, 1))
        data = face_data(images, [[crop]])
        output, = self.node.apply(images, data, "#00FF00", 0.5)
        torch.testing.assert_close(output[0, 1, 2], torch.tensor([0.1, 0.6, 0.1]))
        torch.testing.assert_close(output[0, 1, 3], torch.tensor([0.15, 0.4, 0.15]))
        torch.testing.assert_close(output[0, 2, 3], images[0, 2, 3])
        torch.testing.assert_close(output[0, 0], images[0, 0])
        torch.testing.assert_close(images, torch.full_like(images, 0.2))
        torch.testing.assert_close(crop["mask"], torch.tensor([[1, 0.5, 1], [1, 0, 1.0]]))

    def test_multiple_faces_overlaps_and_frame_order(self):
        images = torch.zeros((3, 4, 7, 3))
        data = face_data(images, [
            [record([[0.5, 1]], (1, 1)), record([[0.75, 1]], (2, 1)), record([[1]], (5, 2))],
            [], [record([[1]], (0, 3))],
        ])
        output, = self.node.apply(images, data, "#FF0000", 0.5)
        torch.testing.assert_close(output[0, 1, 1:4, 0], torch.tensor([0.25, 0.5, 0.5]))
        self.assertEqual(float(output[0, 2, 5, 0]), 0.5)
        torch.testing.assert_close(output[1], images[1])
        self.assertEqual(float(output[2, 3, 0, 0]), 0.5)
        self.assertEqual(float(output[2, :3].sum()), 0)
        # Record order must not affect the union or blend.
        data["frames"][0].reverse()
        torch.testing.assert_close(self.node.apply(images, data, "#FF0000", 0.5)[0], output)

    def test_crop_clipping_at_frame_borders(self):
        images = torch.zeros((1, 3, 4, 3))
        data = face_data(images, [[record([[0, 0], [0, 1]], (-1, -1)),
                                   record([[1, 1], [1, 1]], (3, 2)),
                                   record([[1]], (9, 9))]])
        output, = self.node.apply(images, data, "0000ff", 1)
        expected = images.clone()
        expected[0, 0, 0, 2] = 1
        expected[0, 2, 3, 2] = 1
        torch.testing.assert_close(output, expected)

    def test_opacity_endpoints_alpha_and_dtype(self):
        images = torch.full((1, 2, 3, 4), 0.25, dtype=torch.float64)
        images[..., 3] = 0.7
        data = face_data(images, [[record([[1]], (1, 0))]])
        unchanged, = self.node.apply(images, data, " #aBcD12 ", 0)
        torch.testing.assert_close(unchanged, images)
        output, = self.node.apply(images, data, " #aBcD12 ", 1)
        torch.testing.assert_close(output[0, 0, 1, :3], images.new_tensor([171 / 255, 205 / 255, 18 / 255]))
        torch.testing.assert_close(output[..., 3], images[..., 3])
        self.assertEqual(output.dtype, images.dtype)
        self.assertEqual(output.device, images.device)
        self.assertNotEqual(output.data_ptr(), images.data_ptr())

    def test_invalid_controls(self):
        images = torch.zeros((1, 2, 3, 3))
        data = face_data(images, [[]])
        for color in ("red", "#123", "#GG0000", "#FF000080"):
            with self.subTest(color=color), self.assertRaisesRegex(ValueError, "hex RGB"):
                self.node.apply(images, data, color)
        for opacity in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(opacity=opacity), self.assertRaisesRegex(ValueError, "opacity"):
                self.node.apply(images, data, opacity=opacity)

    def test_invalid_frame_data(self):
        images = torch.zeros((1, 2, 3, 3))
        for data, message in (({}, "face_masks output"),
                              (face_data(images, []), "frame count"),
                              ({**face_data(images, [[]]), "width": 9}, "height and width"),
                              (face_data(images, [[{"bbox": (0, 0, 1, 1)}]]), "mask crops")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.node.apply(images, data)
        with self.assertRaisesRegex(ValueError, "IMAGE batch"):
            self.node.apply(images[0], face_data(images, [[]]))


if __name__ == "__main__":
    unittest.main()
