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
from nupic.bindings.math import *
from scipy.signal import convolve2d


def get_biased_correlations(data, threshold= 10):
  """
  Gets the highest few correlations for each bit, across the entirety of the
  data.  Meant to provide a comparison point for the pairwise correlations
  reported in the literature, which are typically between neighboring neurons
  tuned to the same inputs.  We would expect these neurons to be among the most
  correlated in any region, so pairwise correlations between most likely do not
  provide an unbiased estimator of correlations between arbitrary neurons.
  """
  data = data.toDense()
  correlations = numpy.corrcoef(data, rowvar = False)
  highest_correlations = []
  for row in correlations:
    highest_correlations += sorted(row, reverse = True)[1:threshold+1]
  return numpy.mean(highest_correlations)

def get_pattern_correlations(data):
  """
  Gets the average correlation between all bits in patterns, across the entire
  dataset.  Assumes input is a sparse matrix.  Weighted by pattern rather than
  by bit; this is the average pairwise correlation for every pattern in the
  data, and is not the average pairwise correlation for all bits that ever
  cooccur.  This is a subtle but important difference.
  """

  patterns = [data.rowNonZeros(i)[0] for i in range(data.nRows())]
  dense_data = data.toDense()
  correlations = numpy.corrcoef(dense_data, rowvar = False)
  correlations = numpy.nan_to_num(correlations)
  pattern_correlations = []
  for pattern in patterns:
    pattern_correlations.append([correlations[i, j]
        for i in pattern for j in pattern if i != j])
  return numpy.mean(pattern_correlations)

def generate_correlated_data(dim = 2000,
                             num_active = 40,
                             num_samples = 1000,
                             num_cells_per_cluster_size = [2000]*8,
                             cluster_sizes = range(2, 10)):
  """
  Generates a set of data drawn from a uniform distribution, but with bits
  clustered to force correlation between neurons.  Clusters are randomly chosen
  to form an activation pattern, in such a way as to maintain sparsity.

  The number of clusters of each size is defined by num_cells_per_cluster_size,
  as cluster_size*num_clusters = num_cells. This parameter can be thought of as
  indicating how thoroughly the clusters "cover" the space. Typical values are
  1-3 times the total dimension.  These are specificied independently for each
  cluster size, to allow for more variation.
  """

  # Chooses clusters to
  clusters = []
  cells = set(range(dim))

  # Associate cluster sizes with how many cells they should have, then generate
  # clusters.
  for size, num_cells in zip(cluster_sizes, num_cells_per_cluster_size):

    # Must have (num_cells/size) clusters in order to have num_cells cells
    # across all clusters with this size.
    for i in range(int(1.*num_cells/size)):
      cluster = tuple(numpy.random.choice(dim, size, replace = False))
      clusters.append(cluster)

  # Now generate a list of num_samples SDRs
  datapoints = []
  for sample in range(num_samples):

    # This conditional is necessary in the case that very, very few clusters are
    # created, as otherwise numpy.random.choice can attempt to choose more
    # clusters than is possible.  Typically this case will not occur, but it is
    # possible if we are attempting to generate extremely correlated data.
    if len(clusters) > num_active/2:
      chosen_clusters = numpy.random.choice(len(clusters),
          num_active/2, replace = False)
      current_clusters = [clusters[i] for i in chosen_clusters]
    else:
      current_clusters = clusters

    # Pick clusters until doing so would exceed our target num_active.
    current_cells = set()
    for cluster in current_clusters:
      if len(current_cells) + len(cluster) < num_active:
        current_cells |= set(cluster)
      else:
        break

    # Add random cells to the SDR if we still don't have enough active.
    if len(current_cells) < num_active:
      possible_cells = cells - current_cells
      new_cells = numpy.random.choice(tuple(possible_cells),
          num_active - len(current_cells), replace = False)
      current_cells |= set(new_cells)
    datapoints.append(list(current_cells))

  data = SM32()
  data.reshape(num_samples, dim)
  for sample, datapoint in enumerate(datapoints):
    for i in datapoint:
      data[sample, i] = 1.
  return data

def apply_noise(data, noise):
  """
  Applies noise to a sparse matrix.  Noise can be an integer between 0 and
  100, indicating the percentage of ones in the original input to move, or
  a float in [0, 1), indicating the same thing.
  The input matrix is modified in-place, and nothing is returned.
  This operation does not affect the sparsity of the matrix, or of any
  individual datapoint.
  """
  if noise >= 1:
    noise = noise/100.

  for i in range(data.nRows()):
    ones = data.rowNonZeros(i)[0]
    replace_indices = numpy.random.choice(ones,
        size = int(len(ones)*noise), replace = False)
    for index in replace_indices:
      data[i, index] = 0

    new_indices = numpy.random.choice(data.nCols(),
        size = int(len(ones)*noise), replace = False)

    for index in new_indices:
      while data[i, index] == 1:
        index = numpy.random.randint(0, data.nCols())
      data[i, index] = 1

def shuffle_sparse_matrix_and_labels(matrix, labels):
  """
  Shuffles a sparse matrix and set of labels together.
  Resorts to densifying and then re-sparsifying the matrix, for
  convenience.  Still very fast.
  """
  print "Shuffling data"
  new_matrix = matrix.toDense()
  rng_state = numpy.random.get_state()
  numpy.random.shuffle(new_matrix)
  numpy.random.set_state(rng_state)
  numpy.random.shuffle(labels)

  print "Data shuffled"
  return SM32(new_matrix), numpy.asarray(labels)

def split_sparse_matrix(matrix, num_categories):
  """
  An analog of numpy.split for our sparse matrix.  If the number of
  categories does not divide the number of rows in the matrix, all
  overflow is placed in the final bin.

  In the event that there are more categories than rows, all later
  categories are considered to be an empty sparse matrix.
  """
  if matrix.nRows() < num_categories:
    return [matrix.getSlice(i, i+1, 0, matrix.nCols())
        for i in range(matrix.nRows())] + [SM32()
        for i in range(num_categories - matrix.nRows())]
  else:
    inc = matrix.nRows()/num_categories
    divisions = [matrix.getSlice(i*inc, (i+1)*inc, 0, matrix.nCols())
        for i in range(num_categories - 1)]

    # Handle the last bin separately.  All overflow goes into it.
    divisions.append(matrix.getSlice((num_categories - 1)*inc, matrix.nRows(),
        0, matrix.nCols()))

    return divisions

def generate_evenly_distributed_data_sparse(dim = 2000,
                                            num_active = 40,
                                            num_samples = 1000):
  """
  Generates a set of data drawn from a uniform distribution.  The binning
  structure from Poirazi & Mel is ignored, and all (dim choose num_active)
  arrangements are possible.
  """
  indices = [numpy.random.choice(dim, size = num_active, replace = False)
      for i in range(num_samples)]
  data = SM32()
  data.reshape(num_samples, dim)
  for sample, datapoint in enumerate(indices):
    for index in datapoint:
      data[sample, index] = 1.

  return data


def generate_evenly_distributed_data(dim = 2000,
                                     num_active = 40,
                                     num_samples = 1000,
                                     num_negatives = 500):
  """
  Generates a set of data drawn from a uniform distribution.  The binning
  structure from Poirazi & Mel is ignored, and all (dim choose num_active)
  arrangements are possible. num_negatives samples are put into a separate
  negatives category for output compatibility with generate_data, but are
  otherwise identical.
  """

  sparse_data = [numpy.random.choice(dim, size = num_active, replace = False)
      for i in range(num_samples)]
  data = [[0 for i in range(dim)] for i in range(num_samples)]
  for datapoint, sparse_datapoint in zip(data, sparse_data):
    for i in sparse_datapoint:
      datapoint[i] = 1

  negatives = data[:num_negatives]
  positives = data[num_negatives:]
  return positives, negatives

def generate_data(dim = 40, num_samples = 30000, num_bins = 10, sparse = False):
  """
  Generates data following appendix V of (Poirazi & Mel, 2001).
  Positive and negative examples are drawn from the same distribution, but
  are multiplied by different square matrices, one of them uniform on [-1, 1
  and one the sum of a uniform matrix and a normal one.  It is assumed that half
  the samples are negative in this case.

  Initially samples of dimension dim are produced, with values in each
  dimension being floats, but they are binned into discrete categories, with
  num_bins bins per dimension.  This binning produces an SDR.
  """

  positives, negatives = [], []
  positive, negative = generate_matrices(dim)

  for i in range(num_samples):
    phase_1 = generate_phase_1(dim)
    phase_2 = generate_phase_2(phase_1, dim)


    if i < num_samples/2:
      positives.append(numpy.dot(positive, phase_2))
    else:
      negatives.append(numpy.dot(negative, phase_2))

  binned_data = bin_data(positives + negatives, dim, num_bins)
  positives = binned_data[:len(binned_data)/2]
  negatives = binned_data[len(binned_data)/2:]

  if sparse:
    positives = SM32(positives)
    negatives = SM32(negatives)

  return positives, negatives

def generate_phase_1(dim = 40):
  """
  The first step in creating datapoints in the Poirazi & Mel model.
  This returns a vector of dimension dim, with the last four values set to
  1 and the rest drawn from a normal distribution.
  """
  phase_1 = numpy.random.normal(0, 1, dim)
  for i in range(dim - 4, dim):
    phase_1[i] = 1.0
  return phase_1

def generate_phase_2(phase_1, dim = 40):
  """
  The second step in creating datapoints in the Poirazi & Mel model.
  This takes a phase 1 vector, and creates a phase 2 vector where each point
  is the product of four elements of the phase 1 vector, randomly drawn with
  replacement.
  """
  phase_2 = []
  for i in range(dim):
    indices = [numpy.random.randint(0, dim) for i in range(4)]
    phase_2.append(numpy.prod([phase_1[i] for i in indices]))
  return phase_2

def generate_matrices(dim = 40):
  """
  Generates the matrices that positive and negative samples are multiplied
  with.  The matrix for positive samples is randomly drawn from a uniform
  distribution, with elements in [-1, 1].  The matrix for negative examples
  is the sum of the positive matrix with a matrix drawn from a normal
  distribution with mean 0 variance 1.
  """
  positive = numpy.random.uniform(-1, 1, (dim, dim))
  negative = positive + numpy.random.normal(0, 1, (dim, dim))
  return positive, negative

def generate_RF_bins(data, dim = 40, num_bins = 10):
  """
  Generates bins for the encoder.  Bins are designed to have equal frequency,
  per Poirazi & Mel (2001), which requires reading the data once.
  Bins are represented as the intervals dividing them.
  """
  intervals = []
  for i in range(dim):
    current_dim_data = [data[x][i] for x in range(len(data))]
    current_dim_data = numpy.sort(current_dim_data)
    intervals.append([current_dim_data[int(len(current_dim_data)*x/num_bins)]
        for x in range(1, num_bins)])
  return intervals

def bin_number(datapoint, intervals):
  """
  Given a datapoint and intervals representing bins, returns the number
  represented in binned form, where the bin including the value is
  set to 1 and all others are 0.

  """
  index = numpy.searchsorted(intervals, datapoint)
  return [0 if index != i else 1 for i in range(len(intervals) + 1)]

def bin_data(data, dim = 40, num_bins = 10):
  """
  Fully bins the data generated by generate_data, using generate_RF_bins and
  bin_number.
  """
  intervals = generate_RF_bins(data, dim, num_bins)
  binned_data = [numpy.concatenate([bin_number(data[x][i], intervals[i])
      for i in range(len(data[x]))]) for x in range(len(data))]
  return binned_data

if __name__ == "__main__":
  """
  Generates a set of correlated test data and prints some correlation
  statistics.  This is only meant for testing; ordinarily, experiments using
  tools in this class will generate fresh data directly for each trial.
  """
  data = generate_evenly_distributed_data_sparse(dim = 2000,
                                                 num_active = 32,
                                                 num_samples = 100000)
  data2 = data.toDense()
  correlations = numpy.corrcoef(data2, rowvar = False)
  corrs = []
  for i in xrange(2000):
    for j in xrange(i - 1):
      corrs.append(correlations[i, j])
  print numpy.mean(corrs)
  print get_pattern_correlations(data)
  print get_biased_correlations(data, threshold = 10)
  print get_biased_correlations(data, threshold = 25)
  print get_biased_correlations(data, threshold = 50)
  print get_biased_correlations(data, threshold = 100)
