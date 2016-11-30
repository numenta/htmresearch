Capybara: Online Sequence Classification
===
![capybaras](http://vignette4.wikia.nocookie.net/rio/images/c/c2/Capybaras.png/revision/latest?cb=20141219163253)

## Goal

The goal of the Capybara project is to create a canonical example for 
online and unsupervised classification using a HTM network. You can think 
of it as a form of temporal pooling for HTM networks.

## Project file structure

* `anomaly_detection`: The goal of these experiments is to detect anomalies on 
the SensorTag data (accelerometer data). These were preliminary experiments 
before diving into the network API setup. See README in folder for more details.
 
* `cla_vs_sdr_classifier`: Compares results between the SDRClassifier and 
CLAClassifier on the same datasets.

* `classification`: contains the code to generate data and run the htm 
network to generate TM traces and visualize them. The clustering code in there 
is deprecated and has been replaced by the code in `clustering`. See README in 
folder for more details.

* `sandbox`: contains an example of an online clustering algorithm with 
scalar data, as well as a demo the 2D projection of SDR clusters.

* `clustering`: contains the bulk of the clustering code. See README in 
folder for more details.