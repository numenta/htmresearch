#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --numObjects 2500 3000 3500 4000 --numUniqueFeatures 200 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats.json &
python convergence_simulation.py --numObjects 1500 2000 2500 3000 --numUniqueFeatures 200 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats.json &
python convergence_simulation.py --numObjects 1000 1500 2000 2500 --numUniqueFeatures 200 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats.json &
python convergence_simulation.py --numObjects 500 1000 1500 --numUniqueFeatures 200 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats.json &

python convergence_simulation.py --numObjects 1500 2000 2500 3000 --numUniqueFeatures 150 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats.json &
python convergence_simulation.py --numObjects 1500 2000 2500 --numUniqueFeatures 150 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats.json &
python convergence_simulation.py --numObjects 1000 1500 2000 --numUniqueFeatures 150 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats.json &
python convergence_simulation.py --numObjects 500 1000 1500 --numUniqueFeatures 150 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats.json &
wait

python convergence_simulation.py --numObjects 1500 2000 2500 --numUniqueFeatures 100 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats.json &
python convergence_simulation.py --numObjects 1000 1500 2000 --numUniqueFeatures 100 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats.json &
python convergence_simulation.py --numObjects 800 900 1000 1100 --numUniqueFeatures 100 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats.json &
python convergence_simulation.py --numObjects 400 500 600 --numUniqueFeatures 100 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats.json &

python convergence_simulation.py --numObjects 500 600 700 800 900 1000 --numUniqueFeatures 50 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats.json &
python convergence_simulation.py --numObjects 400 500 600 700 800 --numUniqueFeatures 50 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats.json &
python convergence_simulation.py --numObjects 300 400 500 600 700 --numUniqueFeatures 50 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats.json &
python convergence_simulation.py --numObjects 200 300 400 --numUniqueFeatures 50 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats.json &
wait

python plot_comparison.py
