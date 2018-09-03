#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --bumpType square --numModules 10 --numObjects 100 --numUniqueFeatures 10 --locationModuleWidth 19 21 40 --resultName results/comparisonToIdeal_square.json --repeat 10

python ideal_sim.py &
python bof_sim.py &
wait

python plot_comparison_to_ideal.py --inFile results/comparisonToIdeal_square.json --outFile comparisonToIdeal_square.pdf --locationModuleWidth 19 21 40 --legendPosition 1.05 0.36
