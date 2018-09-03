#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --bumpType square --numModules 10 --numObjects 60 85 100 115 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 42 --seed2 42 --logCellActivity --resultName results/narrowing_square.json

python plot_narrowing.py --inFile results/narrowing_square.json --outFile1 narrowing_singleTrials_square.pdf --outFile2 narrowing_aggregated_square.pdf --exampleObjectCount 100 --aggregateObjectCounts 60 85 100 115 --exampleObjectNumbers 3 28 4 --aggregateYlim -0.02 0.3
