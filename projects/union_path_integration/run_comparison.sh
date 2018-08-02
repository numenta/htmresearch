#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --numObjects 3350 3400 3450 3500 3550 3600 3650 --numUniqueFeatures 400 --locationModuleWidth 20 --resultName results/comparison_400_cpm_400_feats.json &
python convergence_simulation.py --numObjects 2300 2350 2400 2450 2500 2550 2600 2650 --numUniqueFeatures 400 --locationModuleWidth 17 --resultName results/comparison_289_cpm_400_feats.json &
python convergence_simulation.py --numObjects 1700 1750 1800 1850 1900 1950 2000 --numUniqueFeatures 400 --locationModuleWidth 14 --resultName results/comparison_196_cpm_400_feats.json &
python convergence_simulation.py --numObjects 950 1000 1050 1100 1150 --numUniqueFeatures 400 --locationModuleWidth 10 --resultName results/comparison_100_cpm_400_feats.json &

python convergence_simulation.py --numObjects 2600 2650 2700 2750 2800 2850 --numUniqueFeatures 300 --locationModuleWidth 20 --resultName results/comparison_400_cpm_300_feats.json &
python convergence_simulation.py --numObjects 1800 1850 1900 1950 2000 2050 2100 --numUniqueFeatures 300 --locationModuleWidth 17 --resultName results/comparison_289_cpm_300_feats.json &
python convergence_simulation.py --numObjects 1300 1350 1400 1450 1500 1550 1600 --numUniqueFeatures 300 --locationModuleWidth 14 --resultName results/comparison_196_cpm_300_feats.json &
python convergence_simulation.py --numObjects 700 750 800 850 --numUniqueFeatures 300 --locationModuleWidth 10 --resultName results/comparison_100_cpm_300_feats.json &

python convergence_simulation.py --numObjects 1750 1800 1850 1900 1950 2000 --numUniqueFeatures 200 --locationModuleWidth 20 --resultName results/comparison_400_cpm_200_feats.json &
python convergence_simulation.py --numObjects 1200 1250 1300 1350 1400 1450 1500 --numUniqueFeatures 200 --locationModuleWidth 17 --resultName results/comparison_289_cpm_200_feats.json &
python convergence_simulation.py --numObjects 750 800 850 900 950 1000 --numUniqueFeatures 200 --locationModuleWidth 14 --resultName results/comparison_196_cpm_200_feats.json &
python convergence_simulation.py --numObjects 500 550 600 650 --numUniqueFeatures 200 --locationModuleWidth 10 --resultName results/comparison_100_cpm_200_feats.json &

python convergence_simulation.py --numObjects 900 950 1000 1050 1100 1150 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/comparison_400_cpm_100_feats.json &
python convergence_simulation.py --numObjects 600 650 700 750 800 850 --numUniqueFeatures 100 --locationModuleWidth 17 --resultName results/comparison_289_cpm_100_feats.json &
python convergence_simulation.py --numObjects 400 450 500 550 600 650 --numUniqueFeatures 100 --locationModuleWidth 14 --resultName results/comparison_196_cpm_100_feats.json &
python convergence_simulation.py --numObjects 250 300 350 400 --numUniqueFeatures 100 --locationModuleWidth 10 --resultName results/comparison_100_cpm_100_feats.json &


wait

python plot_comparison.py
