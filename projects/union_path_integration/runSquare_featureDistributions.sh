#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_feature_distributions.py --bumpType square --resultName results/featureDistributions_square.json --repeat 1

python plot_feature_distributions.py --inFile results/featureDistributions_square.json --outFile featureDistributions_square.pdf
