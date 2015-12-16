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

import numpy

"""
This module contains
"""

def evaluateResults(classifications, references, idx):
  """
  Calculate statistics for the predicted classifications against the actual.

  @param classifications  (tuple)     Two lists: (0) predictions and
      (1) actual classifications. Items in the predictions list are numpy
      arrays of ints or [None], and items in actual classifications list
      are numpy arrays of ints.

  @param references       (list)            Classification label strings.

  @param idx              (list)            Indices of test samples.

  @return                 (tuple)           Returns a 2-item tuple w/ the
      accuracy (float) and confusion matrix (numpy array).
  """
  accuracy = calculateAccuracy(classifications)
  cm = calculateConfusionMatrix(classifications, references)

  return (accuracy, cm)


def calculateClassificationResults(classifications):
  """
  Calculate the classification accuracy for each category.

  @param classifications  (list)          Two lists: (0) predictions and (1)
      actual classifications. Items in the predictions list are lists of
      ints or None, and items in actual classifications list are ints.

  @return                 (list)          Tuples of class index and accuracy.
  """
  if len(classifications[0]) != len(classifications[1]):
    raise ValueError("Classification lists must have same length.")

  if len(classifications[1]) == 0:
    return []

  # Get all possible labels
  labels = list(set([l for actual in classifications[1] for l in actual]))

  labelsToIdx = {l: i for i,l in enumerate(labels)}
  correctClassifications = numpy.zeros(len(labels))
  totalClassifications = numpy.zeros(len(labels))
  for actual, predicted in zip(classifications[1], classifications[0]):
    for a in actual:
      idx = labelsToIdx[a]
      totalClassifications[idx] += 1
      if a in predicted:
        correctClassifications[idx] += 1

  return zip(labels, correctClassifications / totalClassifications)


def calculateAccuracy(classifications):
  """
  @param classifications    (tuple)     First element is list of predicted
      labels, second is list of actuals; items are numpy arrays.

  @return                   (float)     Correct labels out of total labels,
      where a label is correct if it is amongst the actuals.
  """
  if len(classifications[0]) != len(classifications[1]):
    raise ValueError("Classification lists must have same length.")

  if len(classifications[1]) == 0:
    return None

  accuracy = 0.0
  for actual, predicted in zip(classifications[1], classifications[0]):
    commonElems = numpy.intersect1d(actual, predicted)
    accuracy += len(commonElems)/float(len(actual))

  return accuracy/len(classifications[1])


def calculateConfusionMatrix(classifications, references):
  """
  Returns confusion matrix as a pandas dataframe.
  """
  # TODO: Figure out better way to report multilabel outputs--only handles
  # single label now. So for now return empty array.
  return numpy.array([])

  # if len(classifications[0]) != len(classifications[1]):
  #   raise ValueError("Classification lists must have same length.")
  #
  # total = len(references)
  # cm = numpy.zeros((total, total+1))
  # for actual, predicted in zip(classifications[1], classifications[0]):
  #   if predicted is not None:
  #     cm[actual[0]][predicted[0]] += 1
  #   else:
  #     # No predicted label, so increment the "(none)" column.
  #     cm[actual[0]][total] += 1
  # cm = numpy.vstack((cm, numpy.sum(cm, axis=0)))
  # cm = numpy.hstack((cm, numpy.sum(cm, axis=1).reshape(total+1,1)))
  #
  # cm = pandas.DataFrame(data=cm,
  #                       columns=references+["(none)"]+["Actual Totals"],
  #                       index=references+["Prediction Totals"])
  #
  # return cm
