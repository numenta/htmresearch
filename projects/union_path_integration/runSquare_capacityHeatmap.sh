#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_capacity_heatmap.py --bumpType square --resultName results/capacityHeatmap_square.json --repeat 3

python plot_capacity_heatmap.py --inFile results/capacityHeatmap_square.json --outFile capacityHeatmap_square.pdf
