#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o xtrace

target_dir=$1
exp_settings=$2

orig_project_dir=../classification
project_dir=$target_dir/classification-$exp_settings

# copy into new folder
rm -rf $project_dir || true
cp -r $orig_project_dir $project_dir
# copy setting file
cp $orig_project_dir/settings/exps/$exp_settings $project_dir/settings/htm_network.py
# copy data
rm -rf $target_dir/data* || true
cp -r $orig_project_dir/../data $target_dir/data

sensortag_z='traces/sensortag_z_sp=True_tm=True_tp=False_SDRClassifier'
binary_noise1='traces/binary_ampl=10.0_mean=0.0_noise=1.0_sp=True_tm=True_tp=False_SDRClassifier'
binary_noise0='traces/binary_ampl=10.0_mean=0.0_noise=0.0_sp=True_tm=True_tp=False_SDRClassifier'

# cleanup
rm $project_dir/data/*.csv $project_dir/data/*.png || true
rm -rf $project_dir/results || true
rm -rf $project_dir/results/$sensortag_z || true
rm -rf $project_dir/results/$binary_noise0 || true
rm -rf $project_dir/results/$binary_noise1 || true

# generate data
python $project_dir/generate_acc_data.py
python $project_dir/generate_artificial_data.py

# run network
python $project_dir/run_htm_network.py

# accuracy plots
python $project_dir/plot_accuracy_results.py -f $project_dir/results/$sensortag_z.csv || true
python $project_dir/plot_accuracy_results.py -f $project_dir/results/$binary_noise0.csv || true
python $project_dir/plot_accuracy_results.py -f $project_dir/results/$binary_noise1.csv || true

# distances plots
python $project_dir/plot_tm_distances.py -f $project_dir/results/$sensortag_z.csv || true
python $project_dir/plot_tm_distances.py -f $project_dir/results/$binary_noise0.csv || true
python $project_dir/plot_tm_distances.py -f $project_dir/results/$binary_noise1.csv || true

