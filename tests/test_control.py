"""Portable current-control math tests; synthetic data only, no hardware."""
import subprocess
import sys
import unittest

import numpy as np

from optical_lab.control import Burst, History, action, impulse_velocity, integrate
from optical_lab.effective_frame import EffectiveWindow, FilterSpec
from optical_lab.regions import layout_from_weights, subdivide_layout, weights_for_layout


def layout64():
    masks = np.zeros((8, 8, 8), dtype=np.uint8)
    for i in range(8):
        masks[i, i, :] = 1
    native = layout_from_weights(masks, dict(OffsetX=0, OffsetY=0, Width=8, Height=8))
    return subdivide_layout(native, 64)


class CurrentControlTests(unittest.TestCase):
    def test_nested_regions_cover_exactly_once(self):
        layout = layout64()
        weights = weights_for_layout(layout)
        np.testing.assert_array_equal(weights.sum(0), np.ones((8, 8)))
        self.assertEqual(len(layout['regions']), 64)

    def test_raw_integrals_conserve_energy(self):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        sums = integrate(image, layout64())
        self.assertEqual(sums.shape, (64,))
        self.assertEqual(sums.sum(), image.sum())

    def test_head_has_130_parameters(self):
        weights = np.arange(128).reshape(64, 2) * .001
        head = dict(weights_sum=weights.tolist(), bias=[.1, -.2])
        self.assertEqual(weights.size + len(head['bias']), 130)
        np.testing.assert_allclose(action(np.ones(64), head), weights.sum(0) + head['bias'])

    def test_wrong_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            integrate(np.ones((7, 8)), layout64())
        with self.assertRaises(ValueError):
            action(np.ones(8), dict(weights_sum=np.ones((64, 2)), bias=[0, 0]))

    def test_independent_mean_windows(self):
        window = EffectiveWindow(FilterSpec('mean', 2))
        self.assertIsNone(window.push(np.full((2, 2), 10, np.uint8)))
        np.testing.assert_array_equal(window.push(np.full((2, 2), 20, np.uint8)), 15)
        self.assertIsNone(window.push(np.full((2, 2), 50, np.uint8)))
        np.testing.assert_array_equal(window.push(np.full((2, 2), 50, np.uint8)), 50)

    def test_ema_resets_between_inputs(self):
        window = EffectiveWindow(FilterSpec('ema', 2, .5))
        window.push(np.full((2, 2), 10, np.uint8))
        np.testing.assert_array_equal(window.push(np.full((2, 2), 30, np.uint8)), 20)
        window.push(np.zeros((2, 2), np.uint8))
        np.testing.assert_array_equal(window.push(np.zeros((2, 2), np.uint8)), 0)

    def test_four_temporal_samples_are_independent_copies(self):
        history = History()
        image = np.zeros((1, 96, 96), np.uint8)
        for i in range(10):
            image[:] = 0
            image[0, 0, i] = 255
            frames = history.push(image)
        self.assertEqual(frames.shape, (4, 1, 96, 96))
        self.assertEqual([int(np.argmax(f[0, 0])) for f in frames], [0, 3, 6, 9])

    def test_impulse_conversion(self):
        np.testing.assert_allclose(impulse_velocity(.01, 90, .05), [0, .2], atol=1e-15)

    def test_burst_rejects_old_generation(self):
        burst = Burst()
        old = burst.arm('camera', 1, {}, FilterSpec('mean', 2), (2, 2))
        current = burst.arm('camera', 1, {}, FilterSpec('mean', 2), (2, 2))
        image = np.ones((2, 2), np.uint8)
        burst.feed(old, image, {'TimeStamp': 1}, 'camera', 1, {})
        self.assertEqual(burst.snapshot()[1], 0)
        burst.feed(current, image, {'TimeStamp': 2}, 'camera', 1, {})
        burst.feed(current, image, {'TimeStamp': 3}, 'camera', 1, {})
        self.assertEqual(burst.snapshot()[1], 2)
        self.assertIsNotNone(burst.snapshot()[0])

    def test_math_imports_do_not_load_hardware_or_gui(self):
        subprocess.run([sys.executable, '-c',
            "import sys; from optical_lab import control, effective_frame, regions; "
            "assert not any(x.startswith(('PySpin','PyQt','torch','gi')) for x in sys.modules)"], check=True)


if __name__ == '__main__':
    unittest.main()
