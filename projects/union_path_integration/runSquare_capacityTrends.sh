#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_capacity_trends.py --bumpType square --resultName results/capacityTrends_square.json --repeat 1

python plot_capacity_trends.py --inFile results/capacityTrends_square.json --outFile capacityTrends_square.pdf
