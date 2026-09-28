"""Small geometry and ComfyUI batch contract tests; no model download needed."""

import importlib.util
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np
import torch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "face_occlusion.py"
SPEC = importlib.util.spec_from_file_location("draken_face_occlusion_test", SOURCE)
face_occlusion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = face_occlusion
SPEC.loader.exec_module(face_occlusion)


class HalfMaskSession:
    def run(self, _, inputs):
        assert inputs["input"].shape == (1, 256, 256, 3)
        mask = np.zeros((1, 256, 256, 1), dtype=np.float32)
        mask[:, :, :128] = 1
        return [mask]


class FaceOcclusionTests(unittest.TestCase):
    def setUp(self):
        self.face = np.zeros(15, dtype=np.float32)
        self.face[4:14] = (face_occlusion._FACE_TEMPLATE * 256).reshape(-1)
        self.face[14] = 1

    def test_alignment_preserves_occlusion_side(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = face_occlusion._mask_one_face(image, self.face, HalfMaskSession(), "input", 0.3)
        self.assertGreater(float(mask[128, 80]), 0.9)
        self.assertLess(float(mask[128, 176]), 0.1)
        self.assertLess(float(mask[0, 0]), 0.01)

    def test_batch_and_base_mask(self):
        images = torch.zeros((2, 256, 256, 3), dtype=torch.float32)
        base = torch.ones((1, 256, 256), dtype=torch.float32)
        base[:, :, :128] = 0
        with mock.patch.object(face_occlusion, "_xseg_session", return_value=(HalfMaskSession(), "input")), \
             mock.patch.object(face_occlusion, "_detect_faces", return_value=[self.face]):
            result, data = face_occlusion.FaceOcclusionMask().make_mask(images, base_mask=base)
        self.assertEqual(tuple(result.shape), (2, 256, 256))
        self.assertEqual(float(result.max()), 0)
        self.assertEqual(data["frames"], [[], []])

    def test_no_detected_face_returns_black(self):
        with mock.patch.object(face_occlusion, "_xseg_session", return_value=(HalfMaskSession(), "input")), \
             mock.patch.object(face_occlusion, "_detect_faces", return_value=[]):
            result, data = face_occlusion.FaceOcclusionMask().make_mask(torch.zeros((1, 32, 48, 3)))
        self.assertEqual(tuple(result.shape), (1, 32, 48))
        self.assertEqual(float(result.sum()), 0)
        self.assertEqual(data["frames"], [[]])

    def test_separate_faces_and_disconnected_mask_regions(self):
        left = np.zeros((40, 80), dtype=np.float32)
        left[10:20, 10:15] = 1
        left[10:20, 20:25] = 1  # An obstruction splits one face's visible mask.
        right = np.zeros_like(left)
        right[5:15, 50:60] = 1
        with mock.patch.object(face_occlusion, "_xseg_session", return_value=(HalfMaskSession(), "input")), \
             mock.patch.object(face_occlusion, "_detect_faces", return_value=[None, None]), \
             mock.patch.object(face_occlusion, "_mask_one_face", side_effect=[right, left]):
            combined, data = face_occlusion.FaceOcclusionMask().make_mask(torch.zeros((1, 40, 80, 3)))
        records = data["frames"][0]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["bbox"], (10, 10, 25, 20))
        self.assertEqual(records[1]["bbox"], (50, 5, 60, 15))
        reconstructed = torch.zeros_like(combined[0])
        for record in records:
            x, y = record["mask_origin"]
            mask = record["mask"]
            h, w = mask.shape
            reconstructed[y:y+h, x:x+w] = torch.maximum(reconstructed[y:y+h, x:x+w], mask)
        self.assertTrue(torch.equal(reconstructed, combined[0]))


if __name__ == "__main__":
    unittest.main()
