# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy
import copy
from itertools import izip as zip, count
from nupic.bindings.algorithms import TemporalMemory as TM
from scipy.spatial.distance import cosine
from htmresearch.frameworks.poirazi_neuron_model.data_tools import generate_evenly_distributed_data_sparse
numpy.random.seed(19)

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

def run_tm_union_experiment(dim = 2000,
                            cellsPerColumn=1,
                            num_active = 40,
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
                            num_branches_range = range(50, 51, 1),
                            onset_length = 5,
                            training_iters = 10,
                            num_trials = 10000,
                            automatic_threshold = True,
                            save_results = True):
  """
  Run an experiment tracking the performance of the temporal memory given
  different input dimensions.  The number of active cells is kept fixed, so we
  are in effect varying the sparsity of the input.   We track performance by
  comparing the cells predicted to be active with the cells actually active in
  the sequence without noise at every timestep, and averaging across timesteps.
  Three metrics are used, correlation (Pearson's r, by numpy.corrcoef),
  set similarity (Jaccard index) and cosine similarity (using
  scipy.spatial.distance.cosine).  The Jaccard set similarity is the
  canonical metric used in the paper, but all three tend to produce very similar
  results.

  Output is written to tm_dim_{num_active}.txt, including sample size.
  """
  if automatic_threshold:
    activationThreshold = min(num_active/2, maxNewSynapseCount/2)
    minThreshold = min(num_active/2, maxNewSynapseCount/2)

  for num_branches in num_branches_range:
    overlaps = []
    surprises = []
    csims = []
    for trial in range(num_trials):
      if (trial + 1) % 100 == 0:
        print trial + 1
      tm = TM(columnDimensions=(dim,),
              cellsPerColumn=cellsPerColumn,
              activationThreshold=activationThreshold,
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
      onset = generate_evenly_distributed_data_sparse(dim = dim, num_active = num_active, num_samples = onset_length)

      for branch in range(num_branches):
        datapoint = numpy.random.choice(dim, num_active, replace = False)
        datapoints.append(datapoint)
        for i in range(training_iters):
          for j in range(onset.nRows()):
            activeColumns = set(onset.rowNonZeros(j)[0])
            tm.compute(activeColumns, learn = True)
          tm.compute(datapoint, learn=True)
          tm.reset()

      for j in range(onset.nRows()):
        activeColumns = set(onset.rowNonZeros(j)[0])
        tm.compute(activeColumns, learn = False)
      predicted_cells = tm.getPredictiveCells()

      datapoint = numpy.random.choice(dim, num_active, replace = False)
      overlap = (1. * len(set(predicted_cells) & set(datapoint)))/len(datapoint)
      surprise = len(datapoint) - len(set(predicted_cells) & set(datapoint))
      dense_predicted_cells = numpy.zeros((dim*cellsPerColumn,))
      for cell in predicted_cells:
        dense_predicted_cells[cell] = 1.
      dense_active_cells = numpy.zeros((dim*cellsPerColumn,))
      for cell in datapoint:
        dense_active_cells[cell] = 1.
      csim = 1 - cosine(dense_predicted_cells, dense_active_cells)
      csim = numpy.nan_to_num(csim)
      overlaps.append(overlap)
      surprises.append(surprise)
      csims.append(csim)

    overlap = numpy.mean(overlaps)
    surprise = numpy.mean(surprises)
    csim = numpy.mean(csims)
    print dim, overlap, surprise, csim
    if save_results:
      with open("tm_union_n{}_a{}_c{}.txt".format(dim, num_active, cellsPerColumn), "a") as f:
        f.write(str(num_branches)+", " + str(overlap) + ", " + str(surprise) + ", " + str(csim) + ", " + str(num_trials) + "\n")

if __name__ == "__main__":
  run_tm_union_experiment()
