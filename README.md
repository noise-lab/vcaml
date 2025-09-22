vcaml
==============================

A Machine Learning (ML) pipeline designed to estimate QoE for WebRTC-based video conferencing applications (VCAs) without using application layer headers.

# 0. Prerequisites

## 0.0 Clone this repository

```
git clone https://github.com/noise-lab/vcaml.git
cd vcaml
```

## 0.1 Install Wireshark/Tshark

Wireshark is required to extract RTP/RTCP packets from the network traffic captures.

### MacOS
```
brew install wireshark 
```
### Ubuntu/Debian

```
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tshark  # accept the dumpcap permissions prompt
```

## 0.2 Install uv

`uv` is a fast Python package and project manager, written in Rust.

```
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS: brew install uv
```

## 0.3 Create a virtual environment and install dependencies

```
uv new -r python=3.11 vcaml-env
uv activate vcaml-env
pip install -r requirements.txt
```

# 1. Download datasets
We use the following datasets in our experiments:

- **In-Lab**: A dataset collected in a controlled lab environment with known network conditions emulated using `tc` (Linux Traffic Control). This dataset includes various network impairments such as varying bandwidth, packet loss, latency, and jitter to simulate real-world network conditions. [Click here to download the dataset.](https://example.com)
- **Real-world**: A dataset collected from real-world residential networks from calls conducted between Raspberry Pi devices and a server hosted on the University of Chicago campus. [Click here to download the dataset.](https://example.com)

Both these datasets are available as raw PCAP files. Download and extract them to a local directory. Assuming that you extract them to `/data/vcaml`, you should have the following directory structure:

```
/data/vcaml
    /raw
        /lab
            <pcap files>
        /real-world
            <pcap files>
```

# 2. Create a `.env` file

Create a `.env` file in the root directory of the repository with the following content:

```
RAW_BASE=/data/vcaml/raw  # Holds the raw PCAP files
INTERIM_BASE=/data/vcaml/interim  # Holds intermediate processed data
FEATURES_BASE=/data/vcaml/features  # Holds extracted features
PREDICTIONS_BASE=/data/vcaml/predictions  # Holds model predictions
```

Make sure to replace `/data/vcaml` with the actual path where you extracted the datasets. Keep the rest of the directory structure as is.