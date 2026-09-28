"""Check per-frame colors, square sizing, clipping and frame matching."""

import importlib.util
import pathlib
import unittest

import torch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "face_squares.py"
SPEC = importlib.util.spec_from_file_location("draken_face_squares_test", SOURCE)
squares = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(squares)


def face_data(height, width, frames):
    return {"schema_version": 1, "height": height, "width": width,
            "frames": [[{"bbox": bbox} for bbox in boxes] for boxes in frames]}


class FaceSquareTests(unittest.TestCase):
    def test_multi_face_palette_and_frame_alignment(self):
        images = torch.full((2, 40, 90, 4), 0.2)
        data = face_data(40, 90, [[(60, 10, 70, 20), (10, 10, 20, 20), (35, 10, 45, 20)], []])
        result, = squares.FaceMaskSquares().draw(images, data, line_width=1)
        self.assertTrue(torch.equal(result[0, 10, 10, :3], torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.equal(result[0, 10, 35, :3], torch.tensor([0., 0., 1.])))
        self.assertTrue(torch.equal(result[0, 10, 60, :3], torch.tensor([0., 1., 0.])))
        self.assertTrue(torch.equal(result[1], images[1]))
        self.assertTrue(torch.equal(result[..., 3], images[..., 3]))
        self.assertTrue(torch.all(images == 0.2))
        self.assertTrue(torch.equal(result[0, 15, 15], images[0, 15, 15]))

    def test_square_scale_is_centered_on_mask_bounds(self):
        images = torch.zeros((1, 60, 60, 3))
        data = face_data(60, 60, [[(20, 10, 30, 30)]])
        fitted, = squares.FaceMaskSquares().draw(images, data, size_scale=1, line_width=1)
        smaller, = squares.FaceMaskSquares().draw(images, data, size_scale=0.5, line_width=1)
        larger, = squares.FaceMaskSquares().draw(images, data, size_scale=2, line_width=1)
        self.assertEqual(float(fitted[0, 10, 15, 0]), 1)
        self.assertEqual(float(fitted[0, 29, 34, 0]), 1)
        self.assertEqual(float(smaller[0, 15, 20, 0]), 1)
        self.assertEqual(float(smaller[0, 24, 29, 0]), 1)
        self.assertEqual(float(larger[0, 0, 5, 0]), 1)
        self.assertEqual(float(larger[0, 39, 44, 0]), 1)
        for result in (fitted, smaller, larger):
            self.assertEqual(float(result[0, 20, 25].sum()), 0)

    def test_clipping_does_not_wrap_or_invent_an_edge(self):
        images = torch.zeros((1, 30, 30, 3))
        data = face_data(30, 30, [[(0, 0, 10, 20)]])
        result, = squares.FaceMaskSquares().draw(images, data, line_width=1)
        self.assertEqual(float(result[0, 10, 14, 0]), 1)
        self.assertEqual(float(result[0, 10, 0].sum()), 0)
        self.assertEqual(float(result[0, :, 25:].sum()), 0)

    def test_mismatched_frame_batches_are_rejected(self):
        images = torch.zeros((2, 40, 90, 3))
        with self.assertRaisesRegex(ValueError, "frame count"):
            squares.FaceMaskSquares().draw(images, face_data(40, 90, [[]]))
        with self.assertRaisesRegex(ValueError, "height and width"):
            squares.FaceMaskSquares().draw(images, face_data(41, 90, [[], []]))


if __name__ == "__main__":
    unittest.main()
