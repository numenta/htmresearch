Online Sequence CLassification
===

The online sequence classification projects has a couple of sub-folders:

* `anomaly_detection`: The goal of these experiments is to detect anomalies on 
the SensorTag data (accelerometer data). These were preliminary experiments 
before diving into the network API setup. See README in `anomaly_detection` 
folder for more details.
 
* `cla_vs_sdr_classifier`: Compares results between the SDRClassifier and 
CLAClassifier on the same datasets.

* `classification`: this folder contains the bulk of the online sequence 
classification work. Artificial and real-world data (accelerometer) data are 
used to test various online classification algorithms. See README in 
`classification` folder for more details.

* `clustering`: contains a demo of an online clustering algorithm with 
scalar data, as well as a demo the 2D projection of SDR clusters.