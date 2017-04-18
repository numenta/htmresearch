# Baseline V1

## About
This folder contains scripts to establish a supervised baseline for the 
classification of UCI accelerometer data. The input data can be:
* SDRs coming out of the temporal memory
* Union of SDRs coming out of the TM (average SDRs over fixed-size windows)

## Prerequisites
1. To download and prepare the data, go to `capybara/datasets` and follow the README.
2. To generate the HTM traces (what will be classified), go to `capybara/htm` and follow the README.

## Run the supervised baseline 
Run `python run_baseline --config configs/uci.yml`

## Plot the results
* Run `plot_results.py`.