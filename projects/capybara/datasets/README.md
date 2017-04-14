# Datasets
This folder contains scripts to download, format and analyze the input 
datasets. 

## Before you start
The `UCI Human Activity data` is the most interesting dataset for 
someone looking to evaluate the classification of time series with HTM. 
This is a good temporal dataset to start with and it was the most challenging 
to classify. The data lives under `nupic.research/projects/capybara/datasets/uci_har`.

## Usage
To download and format the data, run: `./run.sh`. The UCR data is password 
protected, so you'll have to download it manually:
* Download the zipped data [here](http://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip)
* Password to unzip: `attempttoclassify`. Make sure to unzip it in 
`nupic.research/projects/capybara/datasets/`.

## Explore the data
Visualizations for data exploration are in the the notebook: `explore_datasets.ipynb`
* Start a jupyter notebook: `jupyter notebok`
* Open and run all the cells in the notebook `explore_datasets.ipynb`

## More details about the datasets

### UCI
Follow the `README` in `nupic.research/projects/capybara/datasets/uci_har/` 
and run the scripts to download and pre-process the data.

### Motifs data
* Go to `SyntheticData` and run `python generate_synthetic_data.py`

### Sensortag data
 * Go to `sensortag` and run `python convert_acc_data.py`
 
### Artificial unit test data
* Go to `artificial` and run `python generate_artificial_data.py`

 ### Synapse data
* Go to `synapse` and refer to the `README`.



