#!/bin/bash

source .venv/bin/activate
export PYTHONPATH=src

python experiments/exp_batch.py
