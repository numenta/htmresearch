Online Sequence Classification
===

The online sequence classification projects has a couple of sub-folders:

* `anomaly_detection`: The goal of these experiments is to detect anomalies on 
the SensorTag data (accelerometer data). These were preliminary experiments 
before diving into the network API setup. See README in folder for more details.
 
* `cla_vs_sdr_classifier`: Compares results between the SDRClassifier and 
CLAClassifier on the same datasets.

* `classification`: contains the code to generate data and run the htm 
network to generate traces and visualize them. The clustering code in there 
is deprecated and has been replaced by the code in `clustering`. See README in 
folder for more details.

* `demo`: contains a demo of an online clustering algorithm with 
scalar data, as well as a demo the 2D projection of SDR clusters.

* `clustering`: contains the bulk of the clustering code. See README in 
folder for more details.