# Description

Run HTM and other competing algorithms on continuous time series prediction problem

# Example Usage

run swarm on continuous time series prediction problems

	python run_swarm.py -d DATASET

Existing dataset includes sine, rec-center-hourly, nyc_taxi, MackeyGlass, SantaFe_A

run swarm using custom permutation settings (e.g. different error metrics)

	python run_swarm.py swarm/nyc_taxi/permutations.py --maxWorkers 8

run nupic model using existing model parameters (located in model_params)

	python run_model_new.py -d nyc_taxi.py

compare TM performance with shift predictor, ARIMA and LSTM

	python comparePerformance.py -d DATASET

run ARIMA on dataset

	Rscript run_arima.R

run LSTM on dataset

    python run_lstm.py -d nyc_taxi -r 30
