#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

repetitions=1

python capacity_simulation.py --numUniqueFeatures 100 --locationModuleWidth 10 --thresholds 0 -1 --numModules 5 10 15 20 25 30 35 40 --resultName results/capacity_100_feats_100_cpm.json --appendResults --repeat $repetitions

python capacity_simulation.py --numUniqueFeatures 100 --locationModuleWidth 6 8 9 10 11 12 13 14  --numModules 10 --resultName results/capacity_100_feats_10_modules.json --appendResults --repeat $repetitions

python capacity_simulation.py --numUniqueFeatures 5 10 20 50 75 100 150 200 --locationModuleWidth 10 --numModules 10 --resultName results/capacity_10_modules_100_cpm.json --appendResults --repeat $repetitions
