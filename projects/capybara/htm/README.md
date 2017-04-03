# HTM traces analysis
 
## Prerequisite
To download and prepare the data, go to `capybara/datasets` and follow the README.

## Usage
To run the HTM network on all datasets, run: `./run_all.sh`. 

## Additional information
This section focuses on the property `timeIndexed` in the `config.yml` files 
located in `capybara/htm/configs/`.
    
### Case 1: run the HTM network on time series data (`timeIndexed: True`)

#### Input data structure
* In this case, each row of the input file listed in `config.yml` contains 
the values of metrics at each timesteps. The metric header is the first row of 
the input file. In the config file, you must specify which metric you are interested to
 run the HTM network on.
* Example for N time steps (from `t0` to `tN`), M metrics and K labels:
```
t |  metric0   | ... |  metricM   | label
--|------------|-----|------------|--------
t0| metric0(t0)| ... | metricM(t0)| label0
t1| metric0(t1)| ... | metricM(t1)| label0
 ...
tN| metric0(tN)| ... | metricM(tN)| labelK

```

#### How to run
* To run the HTM network and save the traces: `python 
run_htm_network.py --config config.yml`
* Make sure that `timeIndexed` is set to `True` in the `config.yml` that you 
are passing. This means that you will run the HTM network on 
time series data (i.e. "time indexed", not "sequence indexed").  


### Case 2: run the HTM network on sequences (`timeIndexed: False`)

#### Input data structure
* In this case, each row is a time series (sequence). The first value of the 
row is the `label`.
* Example for N time steps (from `t0` to `tN`), M sequence samples and K 
labels: 
```
label0, sequence0(t0), ..., sequence0(tN)
label0, sequence1(t0), ..., sequence1(tN)
 ...
labelK, sequenceM(t0), ..., sequenceM(tN)
```

#### How to run
* To run the HTM network and save the traces: `python 
run_htm_network.py --config config.yml`
* Make sure that `timeIndexed` is set to `False` in the `config.yml` that you 
are passing. This means that you will run the HTM network on 
sequences (i.e. "sequence indexed", not "time indexed").  


