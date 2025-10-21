"""ECG signal preprocessing and filtering."""
import numpy as np
from scipy import signal as sig

class ECGPreprocessor:
    def __init__(self, sample_rate=360):
        self.fs = sample_rate

    def bandpass_filter(self, ecg, low=0.5, high=45, order=4):
        nyq = self.fs / 2
        b, a = sig.butter(order, [low/nyq, high/nyq], btype="band")
        return sig.filtfilt(b, a, ecg)

    def notch_filter(self, ecg, freq=50, quality=30):
        b, a = sig.iirnotch(freq / (self.fs / 2), quality)
        return sig.filtfilt(b, a, ecg)

    def remove_baseline(self, ecg, window=0.6):
        window_size = int(window * self.fs)
        if window_size % 2 == 0:
            window_size += 1
        from scipy.ndimage import median_filter
        baseline = median_filter(ecg, size=window_size)
        return ecg - baseline

    def normalize(self, ecg):
        return (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

    def preprocess(self, ecg):
        ecg = self.bandpass_filter(ecg)
        ecg = self.notch_filter(ecg)
        ecg = self.remove_baseline(ecg)
        ecg = self.normalize(ecg)
        return ecg
