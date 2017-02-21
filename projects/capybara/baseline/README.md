## Baseline
This folder contains scripts to establish a supervised baseline for the 
classification of UCI accelerometer data. The input data can be:
* SDRs coming out of the temporal memory
* Union of SDRs coming out of the TM (average SDRs over fixed-size windows)

### Run the baseline
* Download and pre-process UCI data. Go to `capybara/datasets/uci_har` and 
follow the `README` instructions. 
* Run the HTM on the UCI data. Go to `capybara/htm` and run `python 
htm_network_runner.py`. This will generate the HTM traces.
* Finally, run the supervised baseline with `baseline_runner.py`.

## Plot the results
* After running `baseline_runner.py`, run `baseline_runner.py`.