"""Heart Rate Variability analysis."""
import numpy as np
from scipy import signal as sig

class HRVAnalyzer:
    def __init__(self, sample_rate=360):
        self.fs = sample_rate

    def compute_rr_intervals(self, r_peaks):
        rr = np.diff(r_peaks) / self.fs * 1000  # ms
        return rr

    def time_domain(self, rr_intervals):
        rr = rr_intervals
        nn_diff = np.abs(np.diff(rr))
        return {
            "mean_rr": np.mean(rr),
            "sdnn": np.std(rr),
            "rmssd": np.sqrt(np.mean(nn_diff ** 2)),
            "pnn50": np.sum(nn_diff > 50) / len(nn_diff) * 100,
            "mean_hr": 60000 / np.mean(rr),
            "std_hr": np.std(60000 / rr),
        }

    def frequency_domain(self, rr_intervals):
        rr = rr_intervals / 1000  # convert to seconds
        t = np.cumsum(rr)
        t_uniform = np.arange(t[0], t[-1], 1/4)  # 4 Hz interpolation
        rr_interp = np.interp(t_uniform, t, rr)
        rr_interp -= np.mean(rr_interp)
        freqs, psd = sig.welch(rr_interp, fs=4, nperseg=min(256, len(rr_interp)))
        vlf = np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)], freqs[(freqs >= 0.003) & (freqs < 0.04)])
        lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)])
        return {"vlf_power": vlf, "lf_power": lf, "hf_power": hf, "lf_hf_ratio": lf / (hf + 1e-10), "total_power": vlf + lf + hf}

    def detect_arrhythmia(self, rr_intervals, threshold=0.2):
        mean_rr = np.mean(rr_intervals)
        deviations = np.abs(rr_intervals - mean_rr) / mean_rr
        abnormal_indices = np.where(deviations > threshold)[0]
        return {"abnormal_beats": len(abnormal_indices), "abnormal_ratio": len(abnormal_indices) / len(rr_intervals),
                "indices": abnormal_indices.tolist()}
