# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

"""Data splitting is used to partition data into train and test sets."""

import random

# from nupic.bindings.math import Random



class DataSplit(object):
  """Base class for splitting data into train/test partitions."""


  def split(self, samples):
    """Split the given samples into train/test sets.

    @param samples        (list)          Sample elements of any type.
    @return               (list)          Splits where each split is 2-tuple
                                          (training, test) where each element is
                                          a list of elements from samples.
    """
    return NotImplementedError()



class KFolds(DataSplit):
  """Implementation of k-folds cross validation algorithm.

  Sample usage:

      data = [
          ("My manager is bad", "management"),
          ("the equipment needs to be replaced", "facilities"),
          ("I'm not getting paid enough", "compensation"),
          ...,
      ]
      kfolds = KFolds(5)
      splits = kfolds.split(data)
      for trainSamples, testSamples in splits:
        results = runExperiment(trainSamples, testSamples)
        ...

  """


  def __init__(self, k):
    if not isinstance(k, int):
      raise TypeError("k must be integer type, not %r" % type(k))
    if k < 2:
      raise ValueError("k must be 2 or greater, not %i" % k)

    self.k = k


  def split(self, samples, randomize=False, seed=42):
    """Split the given samples into k train/test sets.

    Each train/test split will have len(samples)/k elements in the test set
    and the rest in the train set. Each fold has a distinct, non-overlapping
    test set from the other folds. The samples themselves can be any type.

    @param samples        (list)          Sample elements of any type.
    @param randomize      (bool)          Randomize the order.
    @param seed           (int)           Random seed.
    @return               (list)          Splits where each split is 2-tuple
                                          (training, test) where each element is
                                          a list of elements from samples. Each
                                          training/test pair contains all
                                          elements from samples.
    """
    if len(samples) < self.k:
      raise ValueError(
          "Must have as many samples as number of folds %i" % self.k)

    if randomize:
      random.seed(seed)
      random.shuffle(samples)

    # Aggregate each train/test set to return
    trainTestSplits = []

    # Make sure we have an indexable list
    samples = list(samples)

    numTest = len(samples) / self.k
    for i in xrange(self.k):
      # Determine the range for the test data for this fold
      start = i * numTest
      end = (i + 1) * numTest

      # Split the samples into train and test sets
      testSamples = samples[start:end]
      trainSamples = samples[:start] + samples[end:]

      trainTestSplits.append((trainSamples, testSamples))

    return trainTestSplits



class StandardSplit(DataSplit):
  """Implementation of standard train/test splitting."""


  def __init__(self, trainPortion=0.8):
    if trainPortion < 0.0 or trainPortion > 1.0:
      raise ValueError("trainPortion must be between 0.0 and 1.0, not %.2f."
                       % trainPortion)

    self.trainPortion = trainPortion


  def split(self, samples, randomize=False, seed=42):
    """Split the given samples in one train/test set, where the first n-portion
    of the samples are designated for training.

    @param samples        (list)          Sample elements of any type.
    @param randomize      (bool)          Randomize the order.
    @param seed           (int)           Random seed.
    @return               (tuple)         Split samples into (training, test),
                                          where each element of the 2-tuple is a
                                          list of samples.
    """
    if len(samples) < 2:
      raise ValueError("Must have at least two samples for train/test split.")

    if randomize:
      random.seed(seed)
      random.shuffle(samples)

    # Make sure we have an indexable list
    samples = list(samples)

    sliceIdx = int(self.trainPortion*len(samples))

    return (samples[:sliceIdx], samples[sliceIdx:])



class Buckets(DataSplit):
  """Split data for the 'buckets' experiment."""

  def split(self, bucketSizes, numInference=10, randomize=False, seed=42):
    """Split the given samples into train/test sets.

    @param bucketSizes    (list)
    @param numInference   (int)           Size of first set of data.
    @param randomize      (bool)          Randomize the order.
    @param seed           (int)           Random seed.
    @return               (list)          Splits where each split is 2-tuple
                                          (training, test), and each element is
                                          a list of elements from samples.
    """
    if not all(x > numInference for x in bucketSizes):
      print "Warning: not all buckets have sufficient size for inference."

    # Aggregate each train/test set to return
    trainTestSplits = []

    for b in bucketSizes:
      indices = range(b)
      if randomize:
        random.seed(seed)
        random.shuffle(indices)
      trainTestSplits.append((indices[:numInference], indices[numInference:]))

    return trainTestSplits
