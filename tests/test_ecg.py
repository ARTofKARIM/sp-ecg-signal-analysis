"""Tests for ECG analysis."""
import unittest
import numpy as np
from src.preprocessing import ECGPreprocessor
from src.r_peak_detector import RPeakDetector
from src.hrv_analysis import HRVAnalyzer

class TestPreprocessor(unittest.TestCase):
    def test_normalize(self):
        pp = ECGPreprocessor(360)
        signal = np.random.randn(1000)
        result = pp.normalize(signal)
        self.assertAlmostEqual(np.mean(result), 0, places=5)
        self.assertAlmostEqual(np.std(result), 1, places=5)

class TestRPeakDetector(unittest.TestCase):
    def test_detect_synthetic(self):
        fs = 360
        t = np.arange(0, 5, 1/fs)
        ecg = np.zeros_like(t)
        peaks_true = np.arange(0.5, 5, 0.8)
        for p in peaks_true:
            idx = int(p * fs)
            if idx < len(ecg):
                ecg[idx] = 1.0
        from scipy.ndimage import gaussian_filter1d
        ecg = gaussian_filter1d(ecg, sigma=3)
        detector = RPeakDetector(fs)
        peaks = detector.simple_threshold(ecg, 0.3)
        self.assertGreater(len(peaks), 0)

class TestHRV(unittest.TestCase):
    def test_time_domain(self):
        hrv = HRVAnalyzer()
        rr = np.random.normal(800, 50, 100)
        metrics = hrv.time_domain(rr)
        self.assertIn("mean_rr", metrics)
        self.assertIn("sdnn", metrics)
        self.assertGreater(metrics["mean_hr"], 0)

if __name__ == "__main__":
    unittest.main()
