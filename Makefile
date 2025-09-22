# Makefile
# Load variables from .env
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Default dataset type (can be overridden: make ingest TYPE=real)
TYPE ?= lab

RAW_DIR        := $(RAW_BASE)/$(TYPE)
INTERIM_DIR    := $(INTERIM_BASE)/$(TYPE)
FEATURES_DIR   := $(FEATURES_BASE)/$(TYPE)
PREDICTIONS_DIR:= $(PREDICTIONS_BASE)/$(TYPE)

.PHONY: ingest features predict all

ingest:
	uv run vcaml ingest dir $(RAW_DIR) -o $(INTERIM_DIR) -R --include-rtp

features:
	uv run vcaml features $(INTERIM_DIR) -o $(FEATURES_DIR) --config configs/features.yaml --dataset $(TYPE)

predict:
	uv run vcaml predict $(FEATURES_DIR) -o $(PREDICTIONS_DIR)

all: ingest features predict
