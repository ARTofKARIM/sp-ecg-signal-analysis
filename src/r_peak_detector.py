"""R-peak detection algorithms for ECG signals."""
import numpy as np
from scipy.signal import find_peaks

class RPeakDetector:
    def __init__(self, sample_rate=360):
        self.fs = sample_rate

    def pan_tompkins(self, ecg):
        diff = np.diff(ecg)
        squared = diff ** 2
        window = int(0.15 * self.fs)
        integrated = np.convolve(squared, np.ones(window)/window, mode="same")
        threshold = 0.6 * np.max(integrated)
        min_distance = int(0.3 * self.fs)
        peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)
        return peaks

    def hamilton(self, ecg):
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [8/(self.fs/2), 16/(self.fs/2)], btype="band")
        filtered = filtfilt(b, a, ecg)
        diff = np.abs(np.diff(filtered))
        window = int(0.08 * self.fs)
        ma = np.convolve(diff, np.ones(window)/window, mode="same")
        threshold = np.mean(ma) + 0.5 * np.std(ma)
        min_dist = int(0.3 * self.fs)
        peaks, _ = find_peaks(ma, height=threshold, distance=min_dist)
        return peaks

    def simple_threshold(self, ecg, threshold_factor=0.6):
        threshold = threshold_factor * np.max(ecg)
        min_dist = int(0.3 * self.fs)
        peaks, _ = find_peaks(ecg, height=threshold, distance=min_dist)
        return peaks
