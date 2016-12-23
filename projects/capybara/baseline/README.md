## Baseline
Contains scripts to establish a supervised baseline for the classification 
of UCI accelerometer data. The input data can be:
* SDRs coming out of the temporal memory
* Union of SDRs coming out of the TM (average SDRs over fixed-size windows)

### Scripts
* `tensorflow`: contains tensorflow examples. If I have time to look more 
at TF, the idea is to start with a simple network for the classifier and 
then to add more layers to see how it impacts prediction accuracy. 
TensorBoard can be started for all the examples to visualize the structure 
of the networks.
* `scikit`: shallow neural net classifiers (or other types of classifiers) 