#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o xtrace

python run_htm_network.py --config configs/body_acc_x.no_sequences.yml
python run_htm_network.py --config configs/debug.sequences.yml
python run_htm_network.py --config configs/body_acc_x.sequences.yml
python run_htm_network.py --config configs/synthetic_control.sequences.yml
python run_htm_network.py --config configs/test1.sequences.yml