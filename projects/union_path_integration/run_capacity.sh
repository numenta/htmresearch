#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd


python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 --numUniqueFeatures 500 --locationModuleWidth 10 --resultName results/capacity_1_500_feats_100_cpm.json &
python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 --numUniqueFeatures 500 --locationModuleWidth 14 --resultName results/capacity_1_500_feats_196_cpm.json &
python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 --numUniqueFeatures 500 --locationModuleWidth 20 --resultName results/capacity_1_500_feats_400_cpm.json &
#wait


#python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 --numUniqueFeatures 50 --locationModuleWidth 20 --resultName results/capacity_50_feats_400_cpm.json &
python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/capacity_100_feats_400_cpm.json &
python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 --numUniqueFeatures 200 --locationModuleWidth 20 --resultName results/capacity_200_feats_400_cpm.json &
python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 --numUniqueFeatures 500 --locationModuleWidth 20 --resultName results/capacity_500_feats_400_cpm.json &
wait

python plot_capacity.py
