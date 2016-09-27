# Spatial Pooler Without Topology
## randomSDR, fixed sparsity
`run train_sp.py -b 1 -d randomSDR --spatialImp py --runClassification 1 --trackOverlapCurve 1 -e 50`

## randomSDR, varying sparsity
`run train_sp.py -b 1 -d randomSDRVaryingSparsity --spatialImp cpp --runClassification 1 --trackOverlapCurve 1 -e 100 --name randomSDRVaryingSparsityNoTopology`
 
## Continuous learning experiment
Train SP on random SDR dataset until converge
Then switch to a different dataset
The SP should adapt to the new dataset

`run train_sp.py -b 1 -d randomSDRVaryingSparsity --trackOverlapCurve 1 --name continuous_learning_without_topology --spatialImp cpp --changeDataSetAt 80 -e 220` 

## Fault tolerance experiment (no topology)
Train SP on random SDR dataset until converge
kill a fraction of the SP columns
`run train_sp.py -b 1 --name trauma_boosting_without_topology --spatialImp faulty_sp --killCellsAt 50 --killCellPrct 0.5 --runClassification 1 --trackOverlapCurve 1`

## Random Bar Pairs Vs. Random Cross (No Topology)
`run train_sp.py -d randomBarPairs --spatialImp cpp -e 200 -b 1`
`run train_sp.py -d randomCross --spatialImp cpp -e 200 -b 1`

## two input fields
`run train_sp.py -b 1 –d correlatedSDRPairs`

# Spatial Pooler With Topology

## random SDRs with varying sparsity
`run train_sp_topology.py -b 1 -d randomSDRVaryingSparsity --spatialImp cpp --runClassification 1 --trackOverlapCurve 1 -e 100 --name randomSDRVaryingSparsity`

## continuous learning experiment
`run train_sp_topology.py -b 1 -d randomSDRVaryingSparsity --spatialImp cpp --runClassification 0 --trackOverlapCurve 1 -e 100 --changeDataSetAt 50 --name randomSDRVaryingSparsityContinuousLearning `
## Random Bar Pairs Vs. Random Cross (With Topology)
`run train_large_sp_topology.py -d randomCross --spatialImp py -e 200 -b 1`
`run train_sp_topology.py -d randomBarPairs --spatialImp py -e 200 -b 1`

# Topology experiment (random bar sets)
`run train_sp_topology.py -d randomBarSets -b 1 --name random_bars_with_topology --spatialImp monitored_sp --changeDataSetContinuously 1 --boosting 1`

# Fault tolerance experiment (with topology)
## Train faulty_SP on random bar set dataset
`run train_sp_topology.py -d randomBarSets -b 1 --name trauma_boosting_with_topology --changeDataSetContinuously 1 --spatialImp faulty_sp --killCellsAt 180 --trackOverlapCurve 1 -e 600 --checkTestInput 0 --checkRFCenters 1`
## analyze results
`run analyze_trauma_experiment.py`
## make trauma movie (requires ffmpeg)
 In figures/traumaMovie/
 `ffmpeg -start_number 100 -i frame_%03d.png traumaMovie.mp4`

# NYC Taxi experiment
## Run with random SP (learning off)
`run run_sp_tm_model.py --trainSP 0`
## Run with learning SP, but without boosting
`run run_sp_tm_model.py --trainSP 1 --maxBoost 1`
## Run with learning SP, and boosting
`run run_sp_tm_model.py --trainSP 1 --maxBoost 20`
## plot results