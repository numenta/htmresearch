Sequence Classification
=======================

#Classification of Artificial Data

Generate the data by running: `python generate_data.py`. This will generate the following datasets:
- Non-noisy data: sequence of 3 sine waves with different amplitude and frequency
- Data with spatial noise: same sequence of distinct sine waves, but with with white noise (spatial noise)
- Data with temporal noise: [TODO]

[Optional] Plot the data: `python plot.py` 

Generate the model params for all the datasets: `python generate_model_params.py`

Classify the data: `python classify.py`. The classification accuracy will be printed out. 


#Classification of Real Data

## Benchmark datasets
The datasets were created by recording accelerometer data during the following tasks:
* 5min of walking
* 5min of running
* 5min of jumping
* 5min of going up stairs
* 5min of going down stairs
* 5min of stumbling around
* 5min of sitting

The sensor used to record accelerometer data is the TI SensorTag CC2541

To connect to the SensorTag and write data to a file: enable bluetooth, and run `node sensortag.js '<filename>.csv'`
