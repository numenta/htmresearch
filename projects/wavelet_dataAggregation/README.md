# Description

This directory contains code for an experiment algorithm that automatically selects
data aggregation window using wavelet transformations. The algorithm is validated on
the Numenta Anomaly Benchmark (NAB).

# Dependency
    SciPy (for wavelet transformation)
    NAB
    NuPIC

# Algorithm
    Step 1: Calculate continuous wavelet transformations of the signal
    Step 2: Calculate the variance of wavelet coefficients for each frequency band (time scale)
    Step 3: Apply a threshold to the cumulative distribution of the function obtained in Step 2
            Find the corresponding time scale at the threshold (dt)
    Step 4: The suggested aggregation time scale is chosen as max(dt0, dt/10.0), dt0 is the
            original time scale

# Results

Anomaly score is not affected by the aggregation for threshold < 20%, which means on average
we can aggregate the data by 50% without affecting the anomaly score.

![WeeklyPattern](https://github.com/ywcui1990/nupic.research/blob/master/projects/wavelet_dataAggregation/AnomalyScore_Vs_AggregationThreshold.png)


