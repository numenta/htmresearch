#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

repetitions=3

python convergence_simulation.py --numObjects 3 20 40 60 80 100 120 140 160 180 200 220 --numUniqueFeatures 100 --locationModuleWidth 6 --numModules 6 12 18 --resultName results/squares_varyNumModules_100_feats_6_moduleWidth.json --repeat $repetitions --appendResults

python convergence_simulation.py --numObjects 3 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 --numUniqueFeatures 100 --locationModuleWidth 9 --numModules 6 12 18 --resultName results/squares_varyNumModules_100_feats_9_moduleWidth.json --repeat $repetitions --appendResults

python convergence_simulation.py --numObjects 3 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540 560 580 600 --numUniqueFeatures 100 --locationModuleWidth 12 --numModules 6 12 18 --resultName results/squares_varyNumModules_100_feats_12_moduleWidth.json --repeat $repetitions --appendResults
