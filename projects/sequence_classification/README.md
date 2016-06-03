Sequence classification
========================

This repository contains experimental application of HTM algorithm to time series
classification problems. The problem involves predicting class labels of 
sequences given a set of training sequences and the training class labels. 

We considered two datasets: the UCR Time Series Classification Archive and a 
synthetic datasets with sequence motifs. To use the scripts in this repo, you need
to first download the UCR dataset www.cs.ucr.edu/~eamonn/time_series_data/ or 
generate the synthetic data by running "python generate_synthetic_data.py"

Several different approaches were evaluated. We tried to combine 1-NN classifier
with the following distance metrics
* Euclidean distance (as reported in the UCR benchmark)
* SDR overlap distance with RDSE encoder (run_encoder_only.py)
* SDR overlap distance with RDSE encoder + union pooler (run_encoder_with_union.py)
* SDR overlap distance with RDSE encoder + TM + union pooler (run_sequence_classification_experiment.py)

On the UCR benchmark, we found that the RDSE encoder gives slightly better performance than Euclidean 
distance (1.7% improvement), and RDSE encoder + union gives a bigger improvement (7.4% improvement). 
Adding the temporal memory component, however, does not contribute to performance improvement on the 
UCR benchmark. 

On the synthetic data, using temporal memory greatly improved prediction performance. This is 
because the representation reflect presence of specific motifs in the sequence. 