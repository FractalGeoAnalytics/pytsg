import unittest

from src.pytsg import feature
from numpy.typing import NDArray
import numpy as np


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.wavelength = np.arange(-10, 10)
        self.signal = feature.gaussian(self.wavelength, -1, 0, 5).reshape(1, -1)

    def test_band_statistics(self):
        br = feature.band_extractor(self.signal)
        expectation = np.asarray([10, -1, -11.953864]).reshape(1, 3)
        np.testing.assert_allclose(br, expectation, atol=10e-3)

    def test_sqm(self):
        results, _ = feature.sqm(self.wavelength, self.signal)
        expectation = np.asanyarray([-0.12984183, -0.9112505, 19.70058995]).reshape(
            1, 3
        )
        np.testing.assert_allclose(results, expectation, atol=10e-3)

    def test_gaussian(self):
        results = feature.fit_gaussian(self.wavelength, self.signal, [1, 9, -5])
        expectation = np.asarray([-1, 0, 5]).reshape(1, 3)
        np.testing.assert_allclose(results, expectation, atol=10e-3)

    def tearDown(self):
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
