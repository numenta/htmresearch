import numpy
import copy
from itertools import izip as zip, count
from nupic.bindings.algorithms import TemporalMemory as TM
from scipy.spatial.distance import cosine
from data_tools import generate_evenly_distributed_data_sparse


def convert_cell_lists_to_dense(dim, cell_list, add_1 = False):
  if add_1:
    dense_cell_list = numpy.zeros((len(cell_list), dim + 1))
  else:
    dense_cell_list = numpy.zeros((len(cell_list), dim))
  for i, datapoint in enumerate(cell_list):
    for cell in datapoint:
      dense_cell_list[i, int(cell)] = 1
    if add_1:
      dense_cell_list[i, dim] = 1

  return dense_cell_list


def run_tm_dim_experiment(test_dims = range(2900, 3100, 200),
                          cellsPerColumn=1,
                          num_active = 10,
                          activationThreshold=5,
                          initialPermanence=0.8,
                          connectedPermanence=0.50,
                          minThreshold=5,
                          maxNewSynapseCount=20,
                          permanenceIncrement=0.05,
                          permanenceDecrement=0.00,
                          predictedSegmentDecrement=0.000,
                          maxSegmentsPerCell=255,
                          maxSynapsesPerSegment=255,
                          seed=42,
                          num_samples = 1000,
                          sequence_length = 20,
                          training_iters = 1,
                          automatic_threshold = True):

  """
  Run an experiment tracking the performance of the temporal memory given
  different input dimensions.  The number of active cells is kept fixed, so we
  are in effect varying the sparsity of the input.   We track performance by
  comparing the cells predicted to be active with the cells actually active in
  the sequence without noise at every timestep, and averaging across timesteps.
  Three metrics are used, correlation (Pearson's r, by numpy.corrcoef), set similarity (Jaccard index) and cosine
  similarity (using scipy.spatial.distance.cosine).

  Output is written to tm_dim_{num_active}.txt, including sample size.
  """
  if automatic_threshold:
    activationThreshold = min(num_active/2, maxNewSynapseCount/2)
    minThreshold = min(num_active/2, maxNewSynapseCount/2)

  for dim in test_dims:
    print dim
    tm = TM(columnDimensions = (dim,),
            cellsPerColumn= cellsPerColumn,
            activationThreshold= activationThreshold,
            initialPermanence=initialPermanence,
            connectedPermanence=connectedPermanence,
            minThreshold=minThreshold,
            maxNewSynapseCount=maxNewSynapseCount,
            permanenceIncrement=permanenceIncrement,
            permanenceDecrement=permanenceDecrement,
            predictedSegmentDecrement=predictedSegmentDecrement,
            maxSegmentsPerCell=maxSegmentsPerCell,
            maxSynapsesPerSegment=maxSynapsesPerSegment,
            seed=seed)

    datapoints = []
    canonical_active_cells = []

    for sample in range(num_samples):
      if (sample + 1) % 10 == 0:
        print sample + 1
      data = generate_evenly_distributed_data_sparse(dim = dim, num_active = num_active, num_samples = sequence_length)
      datapoints.append(data)
      for i in range(training_iters):
        for j in range(data.nRows()):
          activeColumns = set(data.rowNonZeros(j)[0])
          tm.compute(activeColumns, learn = True)
        tm.reset()

      current_active_cells = []
      for j in range(data.nRows()):
        activeColumns = set(data.rowNonZeros(j)[0])
        tm.compute(activeColumns, learn = True)
        current_active_cells.append(tm.getActiveCells())
      canonical_active_cells.append(current_active_cells)
      tm.reset()

    # Now that the TM has been trained, check its performance on each sequence with noise added.
    correlations = []
    similarities = []
    csims = []
    for datapoint, active_cells in zip(datapoints, canonical_active_cells):
      data = copy.deepcopy(datapoint)
      predicted_cells = []

      for j in range(data.nRows()):
        activeColumns = set(data.rowNonZeros(j)[0])
        tm.compute(activeColumns, learn = False)
        predicted_cells.append(tm.getPredictiveCells())
      tm.reset()
        
      similarity = [(0.+len(set(predicted) & set(active)))/len((set(predicted) | set(active))) for predicted, active in zip (predicted_cells[:-1], active_cells[1:])]
      dense_predicted_cells = convert_cell_lists_to_dense(dim*cellsPerColumn, predicted_cells[:-1])
      dense_active_cells = convert_cell_lists_to_dense(dim*cellsPerColumn, active_cells[1:])

      correlation = [numpy.corrcoef(numpy.asarray([predicted, active]))[0, 1] for predicted, active in zip(dense_predicted_cells, dense_active_cells)]

      csim = [1 - cosine(predicted, active) for predicted, active in zip(dense_predicted_cells, dense_active_cells)]

      correlation = numpy.nan_to_num(correlation)
      csim = numpy.nan_to_num(csim)
      correlations.append(numpy.mean(correlation))
      similarities.append(numpy.mean(similarity))
      csims.append(numpy.mean(csim))



    print correlations
    correlation = numpy.mean(correlations)
    similarity = numpy.mean(similarities)
    csim = numpy.mean(csims)
    print dim, correlation, similarity, csim
    #with open("tm_dim_{}.txt".format(num_active), "a") as f:
    #  f.write(str(dim)+", " + str(correlation) + ", " + str(similarity) + ", " + str(csim) + ", " + str(num_samples) + "\n")

if __name__ == "__main__":
  run_tm_dim_experiment()
