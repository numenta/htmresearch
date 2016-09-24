# randomSDR, fixed sparsity
`run train_sp.py -b 1 -d randomSDR --spatialImp py --runClassification 1 --trackOverlapCurve 1 -e 50`

# randomSDR, varying sparsity
`run train_sp.py -b 1 -d randomSDRVaryingSparsity --spatialImp cpp --runClassification 0 --trackOverlapCurve 1 -e 50`

# two input fields
`run train_sp.py -b 1 –d correlatedSDRPairs`
 
# Continuous learning experiment
Train SP on random SDR dataset until converge
Then switch to a different dataset
The SP should adapt to the new dataset

`run train_sp.py -b 1 -d randomSDRVaryingSparsity --trackOverlapCurve 1 --name continuous_learning_without_topology --spatialImp cpp --changeDataSetAt 80 -e 220` 

# Fault tolerance experiment (no topology)
Train SP on random SDR dataset until converge
kill a fraction of the SP columns
`run train_sp.py -b 1 --name trauma_boosting_without_topology --spatialImp faulty_sp --killCellsAt 50 --killCellPrct 0.5 --runClassification 1 --trackOverlapCurve 1`

# Topology experiment (random bar sets)
`run train_sp_topology.py -b 1 --name random_bars_with_topology --spatialImp monitored_sp --changeDataSetContinuously 1 --boosting 1`

# Fault tolerance experiment (with topology)
`run train_sp_topology.py -b 1 --name trauma_boosting_with_topology --changeDataSetContinuously 1 --spatialImp faulty_sp --killCellsAt 50 --runClassification 0 --trackOverlapCurve 1 -e 100`

# NYC Taxi experiment
## Run with random SP (learning off)
`run run_sp_tm_model.py --trainSP 0`
## Run with learning SP, but without boosting
`run run_sp_tm_model.py --trainSP 1 --maxBoost 1`
## Run with learning SP, and boosting
`run run_sp_tm_model.py --trainSP 1 --maxBoost 20`
## plot results