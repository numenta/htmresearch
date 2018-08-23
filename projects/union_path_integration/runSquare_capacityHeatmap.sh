#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_capacity_heatmap.py --bumpType square --resultName results/capacityHeatmap_square.json --repeat 1

python plot_cells_features_heatmap.py --inFile results/capacityHeatmap_square.json --outFile capacityHeatmap_square.pdf
