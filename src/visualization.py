"""ECG visualization module."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class ECGVisualizer:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def plot_ecg(self, ecg, sample_rate, r_peaks=None, title="ECG Signal", save=True):
        t = np.arange(len(ecg)) / sample_rate
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, ecg, linewidth=0.5, color="steelblue")
        if r_peaks is not None:
            ax.scatter(r_peaks / sample_rate, ecg[r_peaks], color="red", s=30, zorder=5, label="R-peaks")
            ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        if save:
            fig.savefig(f"{self.output_dir}ecg_signal.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_hrv_time(self, rr_intervals, save=True):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(rr_intervals, linewidth=0.8, color="steelblue")
        ax1.set_ylabel("RR Interval (ms)")
        ax1.set_title("RR Interval Tachogram")
        hr = 60000 / rr_intervals
        ax2.plot(hr, linewidth=0.8, color="coral")
        ax2.set_ylabel("Heart Rate (bpm)")
        ax2.set_xlabel("Beat Number")
        ax2.set_title("Heart Rate")
        if save:
            fig.savefig(f"{self.output_dir}hrv_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_poincare(self, rr_intervals, save=True):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(rr_intervals[:-1], rr_intervals[1:], s=5, alpha=0.5, color="steelblue")
        lims = [min(rr_intervals) * 0.9, max(rr_intervals) * 1.1]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("RR(n) (ms)")
        ax.set_ylabel("RR(n+1) (ms)")
        ax.set_title("Poincaré Plot")
        if save:
            fig.savefig(f"{self.output_dir}poincare.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
