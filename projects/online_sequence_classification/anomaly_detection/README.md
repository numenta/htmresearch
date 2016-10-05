Anomaly Detection 
=======================
The goal of these experiments is to detect anomalies on the SensorTag data 
(accelerometer data)

# Anomaly Detection of the SensorTag data
* Change to `nupic.research/projects/classification/sensor/sensortag_data`
* Run `python trajectory_converter.py` to convert the input files and generate the model params.
* Run `python run_models.py` to run the models on the converted data and generate the output results
* Run `python plot_results.py` to plot the converted data and associated anomalies.
* If all went well, the generated output will be located in `plot_results` 

