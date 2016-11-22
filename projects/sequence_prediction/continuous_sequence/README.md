# Description

Run HTM and other competing algorithms on continuous time series prediction problem

This experiment is used in 

Cui Y, Ahmad S, Hawkins J. Continuous online sequence learning with an unsupervised neural network model. (2016) *Neural Computation,* 28(11) 2474-2504.  DOI: 10.1162/NECO_a_00893


# Example Usage

run HTM sequence memory on continuous time series prediction problems

	python run_tm_model.py -d DATASET

Existing dataset includes nyc_taxi, nyc_taxi_perturb, sine, rec-center-hourly, 

run LSTM on dataset (requires pyBrain)

    python run_lstm_suite.py -e EXPERIMENT_NAME -d

EXPREIMENT_NAME is defined in experiments.cfg (e.g., nyc_taxi_experiment_continuous)
The -d flag removes existing experiment logs and starts a new experiments

run ARIMA on dataset (requires R)

	Rscript run_arima.R

run Extreme Learning Machine model (requires hpelm)

	python run_elm.py -d DATASET

run time-delayed neural network (TDNN) (requires pyBrain)

	python run_tdnn.py -d DATASET

run adaptive filter (requires adaptfilt
	
	python run_adaptive_filter.py -d DATASET

run Echo State Network model (requires Matlab and ESNToolbox)

    run run_esn_model.m in Matlab

compare HTM performance with other algorithms

	python plotPerformance.py.py


