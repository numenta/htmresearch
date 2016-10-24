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

Edit `settings/htm_network.py` and make sure you will run the network with the 
appropriate `INPUT_FILES` - the data that you just generated at step 1. Then 
run `run_htm_network`.

# 3. Plot results

## Plot Temporal Memory sequences
To visualize 2D projections of TM sequences and inter-sequence distances, run   
`python plot_tm_sequences.py -f <filepath>` where `filepath` can be:
* `results/traces_binary_ampl=10.0_mean=0.0_noise=0.0_sp=False_tm=True_tp=False_SDRClassifier.csv`
* `results/traces_sensortag_z_sp=False_tm=True_tp=False_SDRClassifier.csv`

## Plot clustering and classification accuracy results VS TM states
To visualize clustering or classification accuracy results, run 
`python plot_accuracy_results.py -f <filename>` where `filename` can be:
* `results/traces_binary_ampl=10.0_mean=0.0_noise=0.0_sp=False_tm=True_tp=False_SDRClassifier.csv`
* `results/traces_sensortag_z_sp=False_tm=True_tp=False_SDRClassifier.csv`