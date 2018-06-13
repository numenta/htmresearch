# Union Path Integration Narrowing Simulations

These simulations learn objects using an input layer and grid cell
location layer.

## Columns Plus Paper

### Convergence

For the convergence vs. number of objects, comparing number of unique features

    python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 50 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_50_feats.json
    python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_100_feats.json
    python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 5000 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_5000_feats.json


For the convergence vs. number of modules figure:

    python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 5 --numModules 1 2 3 4 5 6 7 8 9 10 --resultName results/convergence_vs_num_modules_100_feats_25_cpm.json --repeat 10
    python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 10 --numModules 1 2 3 4 5 6 7 8 9 10 --resultName results/convergence_vs_num_modules_100_feats_100_cpm.json --repeat 10
    python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 20 --numModules 1 2 3 4 5 6 7 8 9 10 --resultName results/convergence_vs_num_modules_100_feats_400_cpm.json --repeat 10

### Module Size vs. Number of Unique Features

    python convergence_simulation.py --numObjects 2000 2500 3000 3500 --numUniqueFeatures 200 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats.json; python convergence_simulation.py --numObjects 1500 2000 2500 --numUniqueFeatures 200 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats.json; python convergence_simulation.py --numObjects 1000 1500 2000 --numUniqueFeatures 200 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats.json; python convergence_simulation.py --numObjects 200 400 600 800 1000 --numUniqueFeatures 200 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats.json

    python convergence_simulation.py --numObjects 4000 4500 5000 --numUniqueFeatures 200 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats_2.json
    python convergence_simulation.py --numObjects 3000 3500 4000 --numUniqueFeatures 200 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats_2.json
    python convergence_simulation.py --numObjects 2500 3000 3500 --numUniqueFeatures 200 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats_2.json
    python convergence_simulation.py --numObjects 1500 2000 2500 --numUniqueFeatures 200 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats_2.json

### Capacity

The following will run the simulations for the capacity tests:

    python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 5000 --locationModuleWidth 5 --resultName results/convergence_vs_num_objs_25_cpm.json
    python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 5000 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm.json
    python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 5000 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_400_cpm.json
