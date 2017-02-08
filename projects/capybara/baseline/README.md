## Baseline
Contains scripts to establish a supervised baseline for the classification 
of UCI accelerometer data. The input data can be:
* SDRs coming out of the temporal memory
* Union of SDRs coming out of the TM (average SDRs over fixed-size windows)

### How to run the baseline
* The baseline scripts live in `capybara/baseline/keras`. 
* Download and pre-process UCI data, go to `capybara/datasets/uci_har` and 
follow the README instructions. 
* To run the baseline: `python run_baseline.py`

### Other scripts
* `tensorflow`: contains tensorflow classifier examples. 
* `scikit`: shallow neural net classifier examples. 
