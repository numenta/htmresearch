# Datasets
This folder contains scripts to download, format and analyze the input 
datasets. 

## Download, format and analyze the data
Run the ipython notebook `convert_and_plot_sequences.ipynb`. 

## Details on how to get individual datasets

### UCI
Follow the `README` in `nupic.research/projects/capybara/datasets/uci_har/` 
and run the scripts to download and pre-process the data.

### UCR
* There is a password, so you'll have to download it manually.
* Download the zipped data [here](http://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip)
* Password to unzip: `attempttopredict`. Make sure to unzip it in 
`nupic.research/projects/capybara/datasets/`.

### Synthetic data
* Go to `SyntheticData` and run `python generate_synthetic_data.py`

### Sensortag data
 * Go to `sensortag` and run `python convert_acc_data.py`
 
### Artificial unit test data
* Go to `artificial` and run `python generate_artificial_data.py`



