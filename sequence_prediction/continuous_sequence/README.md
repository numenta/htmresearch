# Description

Run HTM on continuous time series prediction problem

# Example Usage

run swarm on continous time series prediction problems

	python run_swarm.py -d DATASET

run swarm using the delta encoder

	python run_swarm.py -d DATASET -f True

Existing DATASET includes sine, MackeyGlass, SantaFe_A, rec-center-hourly, nyc_taxi

run swarm using custom permutation settings (e.g. different error metrics)

	python run_swarm.py swarm_hotgym/permutations.py --maxWorkers 8

run nupic model using existing model parameters

	python run_model_timestamp.py -d rec-center-hourly
	python run_model.py -d sine


run ARIMA on dataset

	Rscript run_arima.R

run LSTM on dataset

    python run_lstm.py -d sine -n 5 -r 30

compare TM performance with a trivial shift predictor and ARIMA 

	python comparePerformance.py -d DATASET
