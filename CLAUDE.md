# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**vcaml** is a research pipeline that estimates WebRTC video QoE metrics (frames/sec, bitrate, jitter, frame height) for Google Meet, Microsoft Teams, and Webex by analyzing network traffic — without relying on application-layer headers. It was published at IMC 2023.

## Setup

```bash
# Install Python dependencies (uses uv; installs core + notebook deps)
uv sync

# PCAP → CSV conversion requires tshark to be installed separately
# tshark must be on PATH before running pcap2csv.py
```

## Running the Pipeline

### Step 1: Collect data (optional — raw captures)
Automated data collection scripts live in `src/data_collection/`:
- `in-lab/` — Selenium + tshark automation for in-lab VCA calls (Google Meet, Webex, Zoom)
- `real-world/` — headless browser automation for real-world Zoom/Meet calls

Run from `src/data_collection/in-lab/` via `python -m vcqoe -h` for options.

### Step 2: Convert PCAPs to CSVs (if starting from raw captures)
```python
from vcaml.util.pcap2csv import convert
convert('/path/to/directory/with/pcaps')
```

### Step 3: Train and evaluate models
```bash
make train                         # in-lab dataset
make train-rw                      # real-world dataset
make train DATASET=data/my_data    # custom path
make train ARGS='--metrics framesReceivedPerSecond --methods ip-udp-ml'
```
All progress and per-experiment results are printed to the terminal. No log files are written.

### Step 4: Analyze results
Open and run the notebooks in `notebooks/` (In_Lab_Analysis, Real_World_Analysis, Sensitivity_Analysis).

## Architecture

### Data Flow
```
PCAP files → tshark → CSV files
                           ↓
WebRTC dump JSON ──→ FileProcessor (links CSVs to WebRTC JSON ground truth)
                           ↓
                    KfoldCVOverFiles (5-fold CV splits, validates files via FileValidator)
                           ↓
                    ModelRunner (orchestrates training and evaluation)
                           ↓
              FeatureExtractor + Estimator → predictions saved as .pkl
```

### Estimation Methods

There are four estimation methods, controlled by the `estimation_method` string:

| Method | Class | Uses RTP headers? | Algorithm |
|---|---|---|---|
| `ip-udp-heuristic` | `IP_UDP_Heuristic` | No | Rule-based |
| `ip-udp-ml` | `IP_UDP_ML` | No | Random Forest |
| `rtp-heuristic` | `RTP_Heuristic` | Yes | Rule-based |
| `rtp-ml` | `RTP_ML` | Yes | Random Forest |

`frameHeight` prediction is only supported by ML methods (not heuristic). FPS metrics (e.g. `framesReceivedPerSecond`) are evaluated with ±2 frames/sec tolerance accuracy in addition to MAE. All other metrics use MAE only.

### Feature Subsets (`src/vcaml/features/feature_extraction.py`)

`FeatureExtractor` computes features over 1-second sliding windows. Feature sets are combined via the `feature_subset` list:

- `LSTATS` — statistical summaries of packet lengths (mean, std, min, max, Q1/Q2/Q3, num_pkts, num_bytes, num_unique)
- `TSTATS` — statistical summaries of inter-arrival times + burst count
- `SIZE` — raw per-packet sizes padded/truncated to `n_features_size` per window
- `IAT` — raw inter-arrival times padded/truncated to `n_features_iat` per window

`RTP_ML` additionally calls `extract_rtp_features()` which extracts RTP-specific stats (buffer time, unique timestamps, OOO sequence numbers, RTP lag).

### Source Module Map

| Path | Purpose |
|---|---|
| `train.py` | Sole entry point — do not run `src/vcaml/models/run_model.py` directly |
| `src/vcaml/config.py` | Loads `config.yaml` and exposes `project_config` for all modules |
| `src/vcaml/models/run_model.py` | `ModelRunner` — orchestrates fold training and evaluation |
| `src/vcaml/models/base_ml_estimator.py` | `BaseMLEstimator` — shared Random Forest training logic |
| `src/vcaml/models/ip_udp_ml.py` / `rtp_ml.py` | ML estimator subclasses |
| `src/vcaml/models/ip_udp_heuristic.py` / `rtp_heuristic.py` | Heuristic estimator subclasses |
| `src/vcaml/features/feature_extraction.py` | `FeatureExtractor` — all feature computation |
| `src/vcaml/util/file_processor.py` | `FileProcessor` — file discovery and CSV/JSON linking |
| `src/vcaml/util/data_splitter.py` | `KfoldCVOverFiles` — k-fold splitting over files |
| `src/vcaml/util/validator.py` | `FileValidator` — filters out malformed/unusable file pairs |
| `src/vcaml/util/webrtc_reader.py` | `WebRTCReader` — parses WebRTC internals JSON |
| `src/vcaml/util/helper_functions.py` | Shared utilities: `filter_video_frames`, `mark_video_frames`, `read_net_file`, `mergeWithWebrtc`, `_FPS_METRICS` |
| `src/vcaml/util/pcap2csv.py` | PCAP → CSV conversion via tshark |
| `src/data_collection/in-lab/` | Automated in-lab call + capture scripts (Selenium + tshark) |
| `src/data_collection/real-world/` | Automated real-world call scripts (Selenium) |

### Trial ID Format

`ModelRunner` generates a `trialId` used to name the output directory under `*_intermediates/`:

```
{metric}_{estimationMethod}_{featureTag}_{datasetName}_cv_{cvIndex}
# e.g. framesReceivedPerSecond_ip-udp-ml_LSTATS-TSTATS_in_lab_data_cv_1
```

### Configuration (`config.yaml` + `src/vcaml/config.py`)

`config.yaml` at the project root is the single source of truth. `src/vcaml/config.py` loads it and exposes `project_config` for all modules. Key fields:
- `prediction_window` (default 1 second)
- `video_ptype` / `rtx_ptype` — per-dataset, per-VCA RTP payload type strings
- `n_features_size/iat` — per-VCA feature vector lengths for raw SIZE/IAT feature modes
- `video_thresh` (306) — UDP length threshold to filter non-video packets
- `training.*` — default metrics, methods, feature subsets, and k-folds for `train.py`

**When adding a new dataset**, add entries to `video_ptype` and `rtx_ptype` in `config.yaml` keyed by the dataset directory basename, then update `FileProcessor._getInLabFiles()` or `_getRealWorldFiles()` as needed.

### Coding Conventions

- **Variable names**: camelCase throughout (e.g. `fileTuple`, `vcaModels`, `datasetName`)
- **Method names**: camelCase for private helpers (e.g. `_filterFiles`, `_buildIntervals`); snake_case kept for public API methods used by notebooks (`get_webrtc`, `extract_features`, `validate`)
- **Class names**: UpperCamelCase
- **No log files**: all output goes to stdout via Python `logging`

### Input File Format

Each experiment is a pair:
- **CSV** (14-column tshark export): `frame.time_relative`, `frame.time_epoch`, `ip.src`, `ip.dst`, `ip.proto`, `ip.len`, `udp.srcport`, `udp.dstport`, `udp.length`, `rtp.ssrc`, `rtp.timestamp`, `rtp.seq`, `rtp.p_type`, `rtp.marker`
- **JSON** (WebRTC internals dump): Chrome's `chrome://webrtc-internals` export with `PeerConnections[*].stats` containing time-series for `IT01V*` stats

### Ground Truth (`src/vcaml/util/webrtc_reader.py`)

`WebRTCReader.get_webrtc()` parses the WebRTC JSON to extract per-second time-series. Cumulative stats (e.g., `framesReceived`, `totalFreezesDuration`) are differenced to get per-second values. The most active inbound video stream is selected by highest cumulative `framesPerSecond`.

### File Discovery (`src/vcaml/util/file_processor.py`)

`FileProcessor` discovers files differently by dataset type:
- **in_lab_data** — `<data_dir>/<date>_<vca>_*/*.csv` + `*.json`
- **real_world_data** — `<data_dir>/<device>/<timestamp>-<vca>-*.csv` + matching `.json`

## Data Directory Structure

```
data/
├── in_lab_data/           # In-lab traces (VCA name encoded in subdir name)
│   └── <exp>_<vca>_*/
│       ├── trace.csv
│       └── webrtc.json
├── in_lab_data_intermediates/   # Output: model.pkl, predictions_*.pkl, cv_splits.pkl
├── real_world_data/       # Real-world traces
│   └── <device>/
│       ├── <ts>-<vca>-*.csv
│       └── <ts>-<vca>-*.json
└── real_world_data_intermediates/
```

## MLOps Roadmap

The following improvements are planned to make vcaml reproducible and deployable beyond the research prototype:

1. **Proper packaging** — replace `sys.path.insert` hacks in `train.py` and `src/vcaml/models/` with a proper installable `vcaml` package, so all modules are importable without path manipulation.
2. **Test suite** — add `pytest` unit tests for `FeatureExtractor`, `WebRTCReader`, and `FileValidator`; these are pure functions with well-defined inputs/outputs.
3. **CI (GitHub Actions)** — lint (`ruff`) + test on every PR.
4. **Experiment tracking (MLflow)** — log metric, method, feature subset, per-fold MAE/accuracy, and model artifacts instead of bare `.pkl` files.
5. **Data + pipeline versioning (DVC)** — version-track PCAP/CSV/JSON datasets with a remote; define pipeline stages (`pcap→csv`, `csv→train`) so `dvc repro` only re-runs what changed.
6. **Inference entrypoint** — a `predict.py` CLI (or FastAPI app) that loads a trained model and runs inference on a new CSV, returning per-second QoE estimates.
7. **Docker** — a `Dockerfile` that installs `tshark` + Python deps via `uv`, making the full pipeline self-contained.
