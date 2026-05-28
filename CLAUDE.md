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

### Step 1: Convert PCAPs to CSVs (if starting from raw captures)
```python
# From within src/util/
from pcap2csv import convert
convert('/path/to/directory/with/pcaps')
```

### Step 2: Train and evaluate models
```bash
make train                         # in-lab dataset
make train-rw                      # real-world dataset
make train DATASET=data/my_data    # custom path
make train ARGS='--metrics framesReceivedPerSecond --methods ip-udp-ml'
```
All progress and per-experiment results are printed to the terminal. No log files are written.

### Step 3: Analyze results
Open and run the notebooks in `notebooks/` (In_Lab_Analysis, Real_World_Analysis, Sensitivity_Analysis).

## Architecture

### Data Flow
```
PCAP files → tshark → CSV files
                           ↓
WebRTC dump JSON ──→ FileProcessor (links CSVs to WebRTC JSON ground truth)
                           ↓
                    KfoldCVOverFiles (5-fold CV splits, filters bad files via FileValidator)
                           ↓
                    ModelRunner (orchestrates training and evaluation)
                           ↓
              FeatureExtractor + Estimator → predictions saved as .pkl
```

### Estimation Methods

There are four estimation methods, controlled by the `estimation_method` string:

| Method | Class | Uses RTP headers? |
|---|---|---|
| `ip-udp-heuristic` | `IP_UDP_Heuristic` | No |
| `ip-udp-ml` | `IP_UDP_ML` | No |
| `rtp-heuristic` | `RTP_Heuristic` | Yes |
| `rtp-ml` | `RTP_ML` | Yes |

`frameHeight` prediction is only supported by ML methods (not heuristic).

### Feature Subsets (`src/features/feature_extraction.py`)

`FeatureExtractor` computes features over 1-second sliding windows. Feature sets are combined via the `feature_subset` list:

- `LSTATS` — statistical summaries of packet lengths (mean, std, min, max, Q1/Q2/Q3, num_pkts, num_bytes, num_unique)
- `TSTATS` — statistical summaries of inter-arrival times + burst count
- `SIZE` — raw per-packet sizes padded/truncated to `n_features_size` per window
- `IAT` — raw inter-arrival times padded/truncated to `n_features_iat` per window

`RTP_ML` additionally calls `extract_rtp_features()` which extracts RTP-specific stats (buffer time, unique timestamps, OOO sequence numbers, RTP lag).

### Configuration (`config.yaml` + `src/models/config.py`)

`config.yaml` at the project root is the single source of truth. `src/models/config.py` loads it and exposes `project_config` for all modules. Key fields:
- `prediction_window` (default 1 second)
- `video_ptype` / `rtx_ptype` — per-dataset, per-VCA RTP payload type strings
- `n_features_size/bps/iat` — per-VCA feature vector lengths for raw feature modes
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

### Ground Truth (`src/util/webrtc_reader.py`)

`WebRTCReader.get_webrtc()` parses the WebRTC JSON to extract per-second time-series. Cumulative stats (e.g., `framesReceived`, `totalFreezesDuration`) are differenced to get per-second values. The most active inbound video stream is selected by highest cumulative `framesPerSecond`.

### File Discovery (`src/util/file_processor.py`)

`FileProcessor` discovers files differently by dataset type:
- **in_lab_data** — `<data_dir>/<date>_<vca>_*/*.csv` + `*.json`
- **real_world_data** — `<data_dir>/<device>/<timestamp>-<vca>-*.csv` + matching `.json`

### Entry Point

`train.py` at the project root is the sole entry point. It adds `src/` and `src/models/` to `sys.path`, loads `config.yaml`, and calls `ModelRunner.trainModel()` / `ModelRunner.getTestSetPredictions()`. Do not run `src/models/run_model.py` directly.

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
