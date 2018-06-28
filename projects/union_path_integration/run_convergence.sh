#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

# Convergence, comparing # unique features
python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 50 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_50_feats.json &
python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_100_feats.json &
python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 5000 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_5000_feats.json &
wait

# Convergence vs. number of modules, comparing CPM
python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 5 --numModules 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --resultName results/convergence_vs_num_modules_100_feats_25_cpm.json --repeat 10
python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 10 --numModules 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --resultName results/convergence_vs_num_modules_100_feats_100_cpm.json --repeat 10
python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 20 --numModules 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --resultName results/convergence_vs_num_modules_100_feats_400_cpm.json --repeat 10

# Cumulative convergence
python convergence_simulation.py --numObjects 100 --numUniqueFeatures 10 --locationModuleWidth 20 --resultName results/cumulative_convergence_400_cpm_10_feats_100_objs.json --repeat 10 &
python convergence_simulation.py --numObjects 100 --numUniqueFeatures 10 --locationModuleWidth 10 --resultName results/cumulative_convergence_100_cpm_10_feats_100_objs.json --repeat 10 &
python ideal_sim.py &
python bof_sim.py &
wait

python plot_convergence.py
