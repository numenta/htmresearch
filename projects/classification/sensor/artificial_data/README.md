#Notes 

## Run network on accelerometer data
Make sure that:
* `settings.py` has the right `INPUT_FILE` and that `USE_REAL_DATA=True`
* The default `--fileName` in `analyze_sdr_clusters.py` and `plot_network_traces
.py`is the one you want. Otherwise, specify it via command line args.

## Plot inter-cluster distances
1. `generate_acc_data.py`
2. `run_experiments_v2.py`
3. `analyze_sdr_clusters.py`