# ECG Signal Analysis

A comprehensive ECG signal processing pipeline with R-peak detection, Heart Rate Variability (HRV) analysis, and arrhythmia detection.

## Architecture
```
sp-ecg-signal-analysis/
├── src/
│   ├── data_loader.py      # CSV/WFDB ECG data loading
│   ├── preprocessing.py    # Bandpass, notch, baseline removal
│   ├── r_peak_detector.py  # Pan-Tompkins, Hamilton algorithms
│   ├── hrv_analysis.py     # Time/frequency domain HRV metrics
│   └── visualization.py    # ECG plots, tachogram, Poincaré
├── config/config.yaml
├── tests/test_ecg.py
└── main.py
```

## HRV Metrics
| Domain | Metrics |
|--------|---------|
| Time | Mean RR, SDNN, RMSSD, pNN50, Mean HR |
| Frequency | VLF, LF, HF power, LF/HF ratio |

## Installation
```bash
git clone https://github.com/mouachiqab/sp-ecg-signal-analysis.git
cd sp-ecg-signal-analysis && pip install -r requirements.txt
```

## Usage
```bash
python main.py --data data/ecg_recording.csv
```

## Technologies
- Python 3.9+, SciPy, wfdb, biosppy, matplotlib











