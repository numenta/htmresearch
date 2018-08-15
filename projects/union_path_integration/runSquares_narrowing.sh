#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --numObjects 50 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 100 --seed2 357627 --logCellActivity --resultName results/narrowing_40_feats_50_objects.json

python convergence_simulation.py --numObjects 75 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 100 --seed2 357627 --logCellActivity --resultName results/narrowing_40_feats_75_objects.json

python convergence_simulation.py --numObjects 100 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 100 --seed2 357627 --logCellActivity --resultName results/narrowing_40_feats_100_objects.json

python convergence_simulation.py --numObjects 125 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 100 --seed2 357627 --logCellActivity --resultName results/narrowing_40_feats_125_objects.json
