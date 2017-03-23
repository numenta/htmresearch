# HTM traces analysis
 
## Prepare the data
Go to `nupic.research/projects/capybara/datasets` and run the ipython 
notebook `convert_and_plot_sequences.ipynb`.

## Generate and analyze HTM traces
The main notebook to generate and analyze the HTM traces is 
`sequence_traces_analysis.py`. It contains the analysis results of the HTM 
traces for multiple datasets.

## Additional standalone scripts (Optional, just FYI)

### Run HTM network on sequences

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
htm_network_runner_sequences.py`
* You can pass another config file with the `--config` flag. The  default 
config is `config.yml`.


### Run HTM network on a specific metric

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
* To run the HTM network and save the traces: `python htm_network_runner_metric.py`
* You can pass another config file with the `--config` flag. The  default 
config is `config.yml`.
