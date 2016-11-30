# How to run the scripts
## Install `synapseclient`
* `pip install synapseclient --user`
* More info here: http://python-docs.synapse.org

## Download the data
* Export environment variables (your synapse.org credentials):
```
export SYN_EMAIL='<you_synapse.org_email>'
export SYN_PWD='<you_synapse.org_password>'
```
* Run `download_synapse_data.py`


## Visualize the data
* Start Jupyter notebook. Run `jupyter notebook`.
* Open `analysis_plotly_offline.ipynb`


