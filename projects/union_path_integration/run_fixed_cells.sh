#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

# 5000 cells

python convergence_simulation.py --numObjects 700 750 800 850 900 --numUniqueFeatures 100 --numModules 1 --thresholds 1 --locationModuleWidth 70 --resultName results/fixed_cells_5000_cells_1_modules.json &
python convergence_simulation.py --numObjects 1250 1300 1350 1400 --numUniqueFeatures 100 --numModules 2 --thresholds 2 --locationModuleWidth 50 --resultName results/fixed_cells_5000_cells_2_modules.json &
python convergence_simulation.py --numObjects 1550 1600 1700 1750 --numUniqueFeatures 100 --numModules 3 --thresholds 3 --locationModuleWidth 40 --resultName results/fixed_cells_5000_cells_3_modules.json &
python convergence_simulation.py --numObjects 1650 1700 1750 1800 --numUniqueFeatures 100 --numModules 4 --thresholds 4 --locationModuleWidth 35 --resultName results/fixed_cells_5000_cells_4_modules.json &
python convergence_simulation.py --numObjects 950 1000 1050 1100 1150 1200 --numUniqueFeatures 100 --numModules 5 --thresholds 5 --locationModuleWidth 31 --resultName results/fixed_cells_5000_cells_5_modules.json &
python convergence_simulation.py --numObjects 950 1000 1050 1100 1150 1200 --numUniqueFeatures 100 --numModules 6 --thresholds 6 --locationModuleWidth 28 --resultName results/fixed_cells_5000_cells_6_modules.json &
python convergence_simulation.py --numObjects 950 1000 1050 1100 1150 1200 --numUniqueFeatures 100 --numModules 7 --thresholds 7 --locationModuleWidth 26 --resultName results/fixed_cells_5000_cells_7_modules.json &
python convergence_simulation.py --numObjects 1100 1150 1200 1250 1300 --numUniqueFeatures 100 --numModules 8 --thresholds 8 --locationModuleWidth 25 --resultName results/fixed_cells_5000_cells_8_modules.json &

wait

python convergence_simulation.py --numObjects 800 900 1000 1200 1400 1600 --numUniqueFeatures 100 --numModules 1 --thresholds 1 --locationModuleWidth 100 --resultName results/fixed_cells_5000_cells_1_modules.json &
python convergence_simulation.py --numObjects 900 1000 1200 1400 1600 1800 --numUniqueFeatures 100 --numModules 2 --thresholds 2 --locationModuleWidth 70 --resultName results/fixed_cells_5000_cells_1_modules.json &
python convergence_simulation.py --numObjects 1000 1200 1400 1600 2000 3400 3800 --numUniqueFeatures 100 --numModules 3 --thresholds 3 --locationModuleWidth 57 --resultName results/fixed_cells_5000_cells_2_modules.json &
python convergence_simulation.py --numObjects 1400 2000 2500 3000 3500 4000 --numUniqueFeatures 100 --numModules 4 --thresholds 4 --locationModuleWidth 50 --resultName results/fixed_cells_5000_cells_2_modules.json &
python convergence_simulation.py --numObjects 1000 1500 2000 2500 3000 3500 4000 5000 --numUniqueFeatures 100 --numModules 5 --thresholds 5 --locationModuleWidth 44 --resultName results/fixed_cells_5000_cells_5_modules.json &
python convergence_simulation.py --numObjects 1000 1500 2000 2500 3000 4000 5000 --numUniqueFeatures 100 --numModules 6 --thresholds 6 --locationModuleWidth 40 --resultName results/fixed_cells_5000_cells_6_modules.json &
python convergence_simulation.py --numObjects 1000 2000 3000 4000 --numUniqueFeatures 100 --numModules 7 --thresholds 7 --locationModuleWidth 37 --resultName results/fixed_cells_5000_cells_7_modules.json &
python convergence_simulation.py --numObjects 1000 2000 3000 --numUniqueFeatures 100 --numModules 8 --thresholds 8 --locationModuleWidth 35 --resultName results/fixed_cells_5000_cells_8_modules.json &

wait

python plot_fixed_cells.py
