PYTHON := uv run python
DATASET ?= data/in_lab_data

.PHONY: help install train train-rw

help:
	@echo "vcaml — WebRTC QoE estimation pipeline"
	@echo ""
	@echo "  make train                 Train on in-lab dataset (data/in_lab_data)"
	@echo "  make train-rw              Train on real-world dataset (data/real_world_data)"
	@echo "  make train DATASET=<path>  Train on a custom dataset path"
	@echo ""
	@echo "Override metrics or methods at the command line:"
	@echo "  make train ARGS='--metrics framesReceivedPerSecond --methods ip-udp-ml'"
	@echo ""
	@echo "  make install               Install the vcaml package and all dependencies"

install:
	uv sync

train:
	$(PYTHON) train.py --dataset $(DATASET) $(ARGS)

train-rw:
	$(PYTHON) train.py --dataset data/real_world_data $(ARGS)
