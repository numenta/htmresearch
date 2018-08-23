#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --bumpType square --numObjects 50 75 100 125 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 42 --seed2 42 --logCellActivity --resultName results/narrowing_square.json

python plot_narrowing.py --inFile results/narrowing_square.json --outFile1 narrowing_singleTrials_square.pdf --outFile2 narrowing_aggregated_square.pdf --exampleObjectCount 100 --aggregateObjectCounts 50 75 100 125 --exampleObjectNumbers 66 65 76
