"""Main pipeline for ECG signal analysis."""
import argparse
import yaml
import numpy as np
from src.data_loader import ECGDataLoader
from src.preprocessing import ECGPreprocessor
from src.r_peak_detector import RPeakDetector
from src.hrv_analysis import HRVAnalyzer
from src.visualization import ECGVisualizer

def main():
    parser = argparse.ArgumentParser(description="ECG Signal Analysis")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = ECGDataLoader(config["data"]["sample_rate"])
    ecg = loader.load_csv(args.data)
    pp = ECGPreprocessor(config["data"]["sample_rate"])
    ecg_clean = pp.preprocess(ecg)
    detector = RPeakDetector(config["data"]["sample_rate"])
    r_peaks = detector.hamilton(ecg_clean)
    print(f"Detected {len(r_peaks)} R-peaks")
    hrv = HRVAnalyzer(config["data"]["sample_rate"])
    rr = hrv.compute_rr_intervals(r_peaks)
    time_metrics = hrv.time_domain(rr)
    freq_metrics = hrv.frequency_domain(rr)
    arrhythmia = hrv.detect_arrhythmia(rr)
    print(f"Mean HR: {time_metrics['mean_hr']:.1f} bpm, SDNN: {time_metrics['sdnn']:.1f} ms")
    print(f"LF/HF: {freq_metrics['lf_hf_ratio']:.2f}")
    print(f"Abnormal beats: {arrhythmia['abnormal_beats']}")
    viz = ECGVisualizer()
    viz.plot_ecg(ecg_clean, config["data"]["sample_rate"], r_peaks)
    viz.plot_hrv_time(rr)
    viz.plot_poincare(rr)

if __name__ == "__main__":
    main()
