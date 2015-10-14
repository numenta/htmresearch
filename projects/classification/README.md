Sequence Classification
=======================

# Sensor Data

The goal of the sensor data experiments is to learn and classify successfully
 spatio-temporal sequences. Two types of sensor data are being classified here:
* Artificially generated sensor data.
* Real-data collected from the TI Sensor Tag.

## Classification of Artificial Data

Three types of signals are generated. Each signal has the following 
characteristics:
- Non-noisy data: sequence of 3 sine waves with different amplitude and frequency
- Data with spatial noise: same sequence of distinct sine waves, but with with white noise (spatial noise)
- Data with temporal noise: same sequence of distinct sine waves, but with 
random variation of the signal frequency (temporal noise)

* Change to `nupic.research/projects/classification/sensor/artificial_data`
* To visualize the artificial data, run `python generate_and_plot_data.py`
* To run the experiments, run: `python run_experiments.py`

## Classification of Real Data

### Benchmark datasets
The datasets were created by recording accelerometer data during the following tasks:
* 5min of walking
* 5min of running
* 5min of jumping
* 5min of going up stairs
* 5min of going down stairs
* 5min of stumbling around
* 5min of sitting

The sensor used to record accelerometer data is the TI SensorTag CC2541. 

### Data analysis of the SensorTag data
* Change to `nupic.research/projects/classification/sensor/sensortag_data`
* Run `python trajectory_converter.py` to convert the input files and generate the model params.
* Run `python run_models.py` to run the models on the converted data and generate the output results
* Run `python plot_results.py` to plot the converted data and associated anomalies.
* If all went well, the generated output will be located in `plot_results` 

