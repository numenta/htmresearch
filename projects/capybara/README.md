Capybara: Online Sequence Classification
===
![capybaras](http://vignette4.wikia.nocookie.net/rio/images/c/c2/Capybaras.png/revision/latest?cb=20141219163253)

## Goal

The goal of the Capybara project is to create a canonical example for 
online and unsupervised classification using a HTM network. You can think 
of it as a form of temporal pooling for HTM networks.

## How to run
Follow the `README` instruction for each sub-folder, in this order:
1. Download and format the datasets: `capybara/datasets`.
2. Run the HTM on the datasets: `capybara/htm`.
3. Run the supervised baseline experiments (1st approach): `capybara/supervised_baseline/v1_no_sequences`
4. Run the supervised baseline experiments(2nd approach): `capybara/supervised_baseline/v1_with_sequences`
5. Run the clustering experiments: `capybara/clustering`

## Project file structure
* `datasets`: contains scripts to download, format and visualize the input 
datasets. See README in folder.

* `htm`: contains the code to run the HTM network and generate TM traces. See 
README in folder for more details.

* `baseline`: contains scripts to establish a supervised baseline for the 
classification of sensor data.

* `clustering`: contains the clustering code. See README in 
folder for more details.

* `cla_vs_sdr_classifier`: Compares results between the SDRClassifier and 
CLAClassifier on the same datasets.

* `anomaly_detection`: The goal of these experiments is to detect anomalies in 
the SensorTag data (accelerometer data). These were preliminary experiments 
before diving into online classification. See README in folder for more 
details.

* `sandbox`: contains a bunch of scripts used for quick prototyping. These 
scripts are not used in any of the main experiments and are not maintained. 
