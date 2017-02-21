Online Sequence Classification
========

How to run the experiments?

# 1. Generate data

## Artificial data
Run `generate_acc_data.py`. You can edit the parameters to generate different
 types of artificial data in `settings/artificial_data.py`

## Accelerometer data

### How to collect data and visualize data
The sensor used to record accelerometer data is the TI SensorTag CC2541. 
* To collect data, run: `data/sensortag/record_sensortag_data.js`.
* To visualize data, run: 'plot_raw_sensortag_data.csv'

### Datasets
The datasets were created by recording accelerometer data during the following 
tasks:
* 5min of walking
* 5min of running
* 5min of jumping
* 5min of going up stairs
* 5min of going down stairs
* 5min of stumbling around
* 5min of sitting


### Running experiments on the accelerometer data
Run `generate_acc_data.py`. You can edit the parameters to clean up and 
concatenate the accelerometer data in `settings/acc_data.py`

# 2. Run network 
* Run: `python run_htm_network`.

# 3. Plot results

## Plot Temporal Memory sequences
To visualize inter-sequence / inter-cluster distances and 2D projections of TM 
states , run `python plot_tm_distances.py -f <filepath>` where `filepath` 
can be:
* `results/traces/binary_ampl=10.0_mean=0.0_noise=0.0.csv`
* `results/traces/sensortag_z.csv`

## Plot clustering and classification accuracy results VS TM states
To visualize clustering or classification accuracy results, run 
`python plot_accuracy_results.py -f <filename>` where `filename` can be:
* `results/traces/binary_ampl=10.0_mean=0.0_noise=0.0.csv`
* `results/traces/sensortag_z.csv`
