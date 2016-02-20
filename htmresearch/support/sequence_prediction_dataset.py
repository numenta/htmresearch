#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

"""
Dataset used for sequence prediction task
"""
import random

from htmresearch.support.reberGrammar import generateSequencesNumber

class Dataset(object):
  def generateSequence(self, iteration):
    """
    :return: A two-tuple with
     sequence: a sequence of input elements
     targets: a sequence of possible next inputs
    """
    pass



class ReberDataset(Dataset):
  def __init__(self, maxLength=None):
    if maxLength is None:
      raise "maxLength not specified"

    self.maxLength = maxLength
    self.numSymbols = 8


  def generateSequence(self, iteration):
    (sequence, target) = generateSequencesNumber(self.maxLength, iteration)
    target.append(None)
    return (sequence, target)



class SimpleDataset(Dataset):
  def __init__(self):
    self.sequences = [
      [6, 8, 7, 4, 2, 3, 5],
      [1, 8, 7, 4, 2, 3, 0],
    ]
    self.numSymbols = max(max(self.sequences))

  def generateSequence(self, iteration):
    random.seed(iteration)
    sequence = list(random.choice(self.sequences))
    target = sequence[1:] + [None]
    return (sequence, target)



class HighOrderDataset(Dataset):
  def __init__(self, numPredictions=1):
    self.numPredictions = numPredictions
    self.numSymbols = max(max(self.sequences(numPredictions, perturbed=False)))

  def sequences(self, numPredictions, perturbed):
    if numPredictions == 1:
      if perturbed:
        return [
          [6, 8, 7, 4, 2, 3, 5],
          [1, 8, 7, 4, 2, 3, 0],
          [6, 3, 4, 2, 7, 8, 0],
          [1, 3, 4, 2, 7, 8, 5],
          [0, 9, 7, 8, 5, 3, 4, 6],
          [2, 9, 7, 8, 5, 3, 4, 1],
          [0, 4, 3, 5, 8, 7, 9, 1],
          [2, 4, 3, 5, 8, 7, 9, 6]
        ]
      else:
        return [
          [6, 8, 7, 4, 2, 3, 0],
          [1, 8, 7, 4, 2, 3, 5],
          [6, 3, 4, 2, 7, 8, 5],
          [1, 3, 4, 2, 7, 8, 0],
          [0, 9, 7, 8, 5, 3, 4, 1],
          [2, 9, 7, 8, 5, 3, 4, 6],
          [0, 4, 3, 5, 8, 7, 9, 6],
          [2, 4, 3, 5, 8, 7, 9, 1]
        ]

    elif numPredictions == 2:
      if perturbed:
        return [
          [4, 8, 3, 10, 9, 6, 0],
          [4, 8, 3, 10, 9, 6, 7],
          [5, 8, 3, 10, 9, 6, 1],
          [5, 8, 3, 10, 9, 6, 2],
          [4, 6, 9, 10, 3, 8, 2],
          [4, 6, 9, 10, 3, 8, 1],
          [5, 6, 9, 10, 3, 8, 7],
          [5, 6, 9, 10, 3, 8, 0],
          [4, 3, 8, 6, 1, 10, 11, 0],
          [4, 3, 8, 6, 1, 10, 11, 7],
          [5, 3, 8, 6, 1, 10, 11, 9],
          [5, 3, 8, 6, 1, 10, 11, 2],
          [4, 11, 10, 1, 6, 8, 3, 2],
          [4, 11, 10, 1, 6, 8, 3, 9],
          [5, 11, 10, 1, 6, 8, 3, 7],
          [5, 11, 10, 1, 6, 8, 3, 0]
        ]
      else:
        return [
          [4, 8, 3, 10, 9, 6, 1],
          [4, 8, 3, 10, 9, 6, 2],
          [5, 8, 3, 10, 9, 6, 0],
          [5, 8, 3, 10, 9, 6, 7],
          [4, 6, 9, 10, 3, 8, 7],
          [4, 6, 9, 10, 3, 8, 0],
          [5, 6, 9, 10, 3, 8, 2],
          [5, 6, 9, 10, 3, 8, 1],
          [4, 3, 8, 6, 1, 10, 11, 9],
          [4, 3, 8, 6, 1, 10, 11, 2],
          [5, 3, 8, 6, 1, 10, 11, 0],
          [5, 3, 8, 6, 1, 10, 11, 7],
          [4, 11, 10, 1, 6, 8, 3, 7],
          [4, 11, 10, 1, 6, 8, 3, 0],
          [5, 11, 10, 1, 6, 8, 3, 2],
          [5, 11, 10, 1, 6, 8, 3, 9]
        ]

    elif numPredictions == 4:
      if perturbed:
        return [
          [7, 4, 12, 5, 14, 1, 13],
          [7, 4, 12, 5, 14, 1, 10],
          [7, 4, 12, 5, 14, 1, 6],
          [7, 4, 12, 5, 14, 1, 8],
          [11, 4, 12, 5, 14, 1, 2],
          [11, 4, 12, 5, 14, 1, 3],
          [11, 4, 12, 5, 14, 1, 0],
          [11, 4, 12, 5, 14, 1, 9],
          [7, 1, 14, 5, 12, 4, 9],
          [7, 1, 14, 5, 12, 4, 0],
          [7, 1, 14, 5, 12, 4, 3],
          [7, 1, 14, 5, 12, 4, 2],
          [11, 1, 14, 5, 12, 4, 8],
          [11, 1, 14, 5, 12, 4, 6],
          [11, 1, 14, 5, 12, 4, 10],
          [11, 1, 14, 5, 12, 4, 13],
          [9, 4, 5, 15, 6, 1, 12, 14],
          [9, 4, 5, 15, 6, 1, 12, 11],
          [9, 4, 5, 15, 6, 1, 12, 7],
          [9, 4, 5, 15, 6, 1, 12, 8],
          [13, 4, 5, 15, 6, 1, 12, 2],
          [13, 4, 5, 15, 6, 1, 12, 3],
          [13, 4, 5, 15, 6, 1, 12, 0],
          [13, 4, 5, 15, 6, 1, 12, 10],
          [9, 1, 12, 6, 15, 4, 5, 10],
          [9, 1, 12, 6, 15, 4, 5, 0],
          [9, 1, 12, 6, 15, 4, 5, 3],
          [9, 1, 12, 6, 15, 4, 5, 2],
          [13, 1, 12, 6, 15, 4, 5, 8],
          [13, 1, 12, 6, 15, 4, 5, 7],
          [13, 1, 12, 6, 15, 4, 5, 11],
          [13, 1, 12, 6, 15, 4, 5, 14]
        ]
      else:
        return [
          [7, 4, 12, 5, 14, 1, 2],
          [7, 4, 12, 5, 14, 1, 3],
          [7, 4, 12, 5, 14, 1, 0],
          [7, 4, 12, 5, 14, 1, 9],
          [11, 4, 12, 5, 14, 1, 13],
          [11, 4, 12, 5, 14, 1, 10],
          [11, 4, 12, 5, 14, 1, 6],
          [11, 4, 12, 5, 14, 1, 8],
          [7, 1, 14, 5, 12, 4, 8],
          [7, 1, 14, 5, 12, 4, 6],
          [7, 1, 14, 5, 12, 4, 10],
          [7, 1, 14, 5, 12, 4, 13],
          [11, 1, 14, 5, 12, 4, 9],
          [11, 1, 14, 5, 12, 4, 0],
          [11, 1, 14, 5, 12, 4, 3],
          [11, 1, 14, 5, 12, 4, 2],
          [9, 4, 5, 15, 6, 1, 12, 2],
          [9, 4, 5, 15, 6, 1, 12, 3],
          [9, 4, 5, 15, 6, 1, 12, 0],
          [9, 4, 5, 15, 6, 1, 12, 10],
          [13, 4, 5, 15, 6, 1, 12, 14],
          [13, 4, 5, 15, 6, 1, 12, 11],
          [13, 4, 5, 15, 6, 1, 12, 7],
          [13, 4, 5, 15, 6, 1, 12, 8],
          [9, 1, 12, 6, 15, 4, 5, 8],
          [9, 1, 12, 6, 15, 4, 5, 7],
          [9, 1, 12, 6, 15, 4, 5, 11],
          [9, 1, 12, 6, 15, 4, 5, 14],
          [13, 1, 12, 6, 15, 4, 5, 10],
          [13, 1, 12, 6, 15, 4, 5, 0],
          [13, 1, 12, 6, 15, 4, 5, 3],
          [13, 1, 12, 6, 15, 4, 5, 2]
        ]


  def generateSequence(self, iteration, perturbed=False):
    random.seed(iteration)
    sequence = list(random.choice(self.sequences(self.numPredictions, perturbed)))
    target = sequence[1:] + [None]

    return (sequence, target)
