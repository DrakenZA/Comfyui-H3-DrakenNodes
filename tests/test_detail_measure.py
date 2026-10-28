"""Measurement-tool controls: python -m unittest discover -s tests -p test_detail_measure.py"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("detail_measure", Path(__file__).resolve().parents[1]/"scripts/measure_h3_detail.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class DetailMeasurementTests(unittest.TestCase):
    def test_blur_control_and_static_temporal_control(self):
        rng = np.random.default_rng(721)
        sharp = rng.random((64, 96, 3), dtype=np.float32)
        blurred = cv2.GaussianBlur(sharp, (0, 0), 2)
        a, b = M.spatial_metrics(sharp), M.spatial_metrics(blurred)
        for key in ("laplacian_variance", "sobel_energy", "highpass_rms"):
            self.assertGreater(a[key], b[key]*2)
        temporal = M.temporal_metrics(sharp, sharp)
        self.assertEqual(temporal["frame_difference_mae"], 0)
        # Dense flow has a numerical floor even for identical uint8 inputs.
        self.assertLess(temporal["flow_warp_residual_mae"], 1e-3)

    def test_reference_error_is_native_size_and_checks_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, reconstruction = root/"source", root/"reconstruction"
            source.mkdir()
            reconstruction.mkdir()
            for i in range(3):
                cv2.imwrite(str(source/f"{i:04d}.png"), np.full((32, 48, 3), 100, np.uint8))
                cv2.imwrite(str(reconstruction/f"{i:04d}.png"), np.full((32, 48, 3), 110, np.uint8))
            report = M.measure(reconstruction, (24, 16), 5, reference=source)
            self.assertEqual(report["frames_measured"], 3)
            self.assertAlmostEqual(report["aligned_reference"]["mse"], (10/255)**2)
            self.assertAlmostEqual(report["aligned_reference"]["psnr_db"], 28.130803608679106)
            perfect = M.measure(source, (24, 16), 3, reference=source)
            self.assertTrue(perfect["aligned_reference"]["perfect_match"])
            self.assertIsNone(perfect["aligned_reference"]["psnr_db"])
            cv2.imwrite(str(source/"0001.png"), np.full((32, 32, 3), 100, np.uint8))
            with self.assertRaisesRegex(ValueError, "matching"):
                M.measure(reconstruction, (24, 16), 5, reference=source)

    def test_aspect_crop_and_roi_are_explicit(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        output, fraction = M.prepare(image, (32, 32))
        self.assertEqual(output.shape, (32, 32, 3))
        self.assertEqual(fraction, 0.5)
        output, fraction = M.prepare(image, (32, 32), [0.25, 0, 0.5, 1])
        self.assertEqual(fraction, 0)
        with self.assertRaises(ValueError):
            M.prepare(image, (32, 32), [0.9, 0, 0.5, 1])


if __name__ == "__main__":
    unittest.main()
