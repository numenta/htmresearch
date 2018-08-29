#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_capacity_trends.py --bumpType gaussian --resultName results/capacityTrends_gaussian.json --repeat 1

python plot_capacity_trends.py --inFile results/capacityTrends_gaussian.json --outFile capacityTrends_gaussian.pdf
