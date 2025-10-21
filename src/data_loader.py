"""ECG data loading from various formats."""
import numpy as np
import os

class ECGDataLoader:
    def __init__(self, sample_rate=360):
        self.sample_rate = sample_rate
        self.signal = None
        self.annotations = None

    def load_csv(self, filepath, column=0):
        data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
        self.signal = data[:, column] if data.ndim > 1 else data
        print(f"Loaded {len(self.signal)} samples ({len(self.signal)/self.sample_rate:.1f}s)")
        return self.signal

    def load_wfdb(self, record_path):
        try:
            import wfdb
            record = wfdb.rdrecord(record_path)
            self.signal = record.p_signal[:, 0]
            self.sample_rate = record.fs
            ann = wfdb.rdann(record_path, "atr")
            self.annotations = ann.sample
            print(f"WFDB: {len(self.signal)} samples, {self.sample_rate} Hz")
        except Exception as e:
            print(f"WFDB loading failed: {e}")
        return self.signal

    def get_segment(self, start_sec, duration_sec):
        start = int(start_sec * self.sample_rate)
        end = int((start_sec + duration_sec) * self.sample_rate)
        return self.signal[start:end]
