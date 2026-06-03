PYTHON   := uv run python
DATAROOT ?= /data/taveesh/vca
DATASET  ?= $(DATAROOT)/in_lab_data

.PHONY: help install download-data train train-rw mlflow-ui

help:
	@echo "vcaml — WebRTC QoE estimation pipeline"
	@echo ""
	@echo "  make install               Install the vcaml package and all dependencies"
	@echo "  make download-data         Download in-lab and real-world datasets (run after install)"
	@echo "  make download-data DATAROOT=<path>  Download to a custom root directory"
	@echo "  make download-data ARGS='--dataset in_lab_data'  Download a single dataset"
	@echo ""
	@echo "  make train                 Train on in-lab dataset ($(DATAROOT)/in_lab_data)"
	@echo "  make train-rw              Train on real-world dataset ($(DATAROOT)/real_world_data)"
	@echo "  make train DATASET=<path>  Train on a custom dataset path"
	@echo ""
	@echo "Override metrics or methods at the command line:"
	@echo "  make train ARGS='--metrics framesReceivedPerSecond --methods ip-udp-ml'"
	@echo ""
	@echo "  make mlflow-ui             Launch MLflow UI (backend URI read from config.yaml)"

install:
	uv sync

download-data:
	$(PYTHON) download_data.py --path $(DATAROOT) $(ARGS)

train:
	$(PYTHON) train.py --dataset $(DATASET) $(ARGS)

train-rw:
	$(PYTHON) train.py --dataset $(DATAROOT)/real_world_data $(ARGS)

mlflow-ui:
	$(PYTHON) -c "from vcaml.config import mlflow_tracking_uri; import os; os.execlp('mlflow', 'mlflow', 'ui', '--backend-store-uri', mlflow_tracking_uri)"
