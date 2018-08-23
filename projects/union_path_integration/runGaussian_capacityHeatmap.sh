#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_capacity_heatmap.py --bumpType gaussian --resultName results/capacityHeatmap_gaussian.json --repeat 1

python plot_cells_features_heatmap.py --inFile results/capacityHeatmap_gaussian.json --outFile capacityHeatmap_gaussian.pdf
