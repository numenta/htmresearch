#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --bumpType gaussian --numModules 10 --numObjects 40 50 60 70 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 42 --seed2 42 --logCellActivity --resultName results/narrowing_gaussian.json

python plot_narrowing.py --inFile results/narrowing_gaussian.json --outFile1 narrowing_singleTrials_gaussian.pdf --outFile2 narrowing_aggregated_gaussian.pdf --exampleObjectCount 50 --aggregateObjectCounts 40 50 60 70 --exampleObjectNumbers 28 27 30
