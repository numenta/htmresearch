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
import copy
import random

from htmresearch.support.reberGrammar import generateSequencesNumber

def scrambleSequence(sequences, seed):
  numSymbols = max(max(e) for e in sequences) + 1
  symbolSet = range(numSymbols)
  shuffledSymbolSet = copy.copy(symbolSet)
  random.seed(seed)
  random.shuffle(shuffledSymbolSet)
  symbolMapping = dict()
  for i in range(numSymbols):
    symbolMapping[symbolSet[i]] = shuffledSymbolSet[i]

  shufledSequences = []
  for sequence in sequences:
    newSequence = []
    for symbol in sequence:
      newSequence.append(symbolMapping[symbol])
    shufledSequences.append(newSequence)
  return shufledSequences



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


  def generateSequence(self, seed):
    (sequence, target) = generateSequencesNumber(self.maxLength, seed)
    target.append(None)
    return (sequence, target)



class SimpleDataset(Dataset):
  def __init__(self):
    self.sequences = [
      [6, 8, 7, 4, 2, 3, 5],
      [1, 8, 7, 4, 2, 3, 0],
    ]
    self.numSymbols = max(max(e) for e in self.sequences) + 1

  def generateSequence(self, seed):
    random.seed(seed)
    sequence = list(random.choice(self.sequences))
    target = sequence[1:] + [None]
    return (sequence, target)



class HighOrderDataset(Dataset):
  def __init__(self, numPredictions=1, seed=1, smallAlphabet=False):
    self.numPredictions = numPredictions
    self.seed = seed
    self.smallAlphabet = smallAlphabet

    self.sequences = self.generateSequenceSet(numPredictions, False)
    self.perturbedSequences = self.generateSequenceSet(numPredictions, True)
    self.numSymbols = max(max(e) for e in self.sequences) + 1

    # scramble sequences
    self.sequences = scrambleSequence(self.sequences, seed)
    self.perturbedSequences = scrambleSequence(self.perturbedSequences, seed)


  def generateSequenceSet(self, numPredictions, perturbed=False):
    if numPredictions == 1:
      if perturbed:
        if self.smallAlphabet:
          return [
            [6, 3, 4, 3, 4, 3, 5],
            [1, 3, 4, 3, 4, 3, 0],
            [6, 4, 3, 3, 4, 3, 0],
            [1, 4, 3, 3, 4, 3, 5],
            [0, 4, 4, 3, 3, 4, 3, 6],
            [2, 4, 4, 3, 3, 4, 3, 1],
            [0, 3, 3, 4, 4, 3, 4, 1],
            [2, 3, 3, 4, 4, 3, 4, 6]
          ]
        else:
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
        if self.smallAlphabet:
          return [
            [6, 3, 4, 3, 4, 3, 0],
            [1, 3, 4, 3, 4, 3, 5],
            [6, 4, 3, 3, 4, 3, 5],
            [1, 4, 3, 3, 4, 3, 0],
            [0, 4, 4, 3, 3, 4, 3, 1],
            [2, 4, 4, 3, 3, 4, 3, 6],
            [0, 3, 3, 4, 4, 3, 4, 6],
            [2, 3, 3, 4, 4, 3, 4, 1]
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


  def generateSequence(self, seed, perturbed=False):
    random.seed(seed)
    if perturbed:
      sequence = list(random.choice(self.perturbedSequences))
    else:
      sequence = list(random.choice(self.sequences))

    target = sequence[1:] + [None]
    return (sequence, target)


class LongHighOrderDataset(Dataset):

  def __init__(self, sequenceLength=10, seed=1):
    self.sequenceSeed = seed
    self.sequenceLength = sequenceLength
    self.symbolPoolSize = sequenceLength*2
    self.sequences = self.generateSequenceSet(2, sequenceLength, seed)
    self.numSymbols = max(max(e) for e in self.sequences) + 1

  def generateSequenceSet(self, numSequenceGroups, sequenceLength, seed):
    sequences = []
    random.seed(seed)
    symbolPool = range(self.symbolPoolSize)

    for i in range(numSequenceGroups):
      shuffledPool = copy.copy(symbolPool)
      random.shuffle(shuffledPool)
      startElement1 = [shuffledPool[0]]
      startElement2 = [shuffledPool[1]]
      endElement1 = [shuffledPool[2]]
      endElement2 = [shuffledPool[3]]
      sequenceElements = shuffledPool[4:(4+sequenceLength-2)]

      sharedSubsequence1 = copy.copy(sequenceElements)
      sharedSubsequence2 = copy.copy(sequenceElements)
      while sharedSubsequence1 == sharedSubsequence2:
        random.shuffle(sharedSubsequence1)
        random.shuffle(sharedSubsequence2)

      sequences.append(startElement1+sharedSubsequence1+endElement1)
      sequences.append(startElement2+sharedSubsequence1+endElement2)
      # sequences.append(startElement1+sharedSubsequence2+endElement2)
      # sequences.append(startElement2+sharedSubsequence2+endElement1)
    return sequences

  def generateSequence(self, seed, perturbed=False):
    random.seed(seed)
    sequence = list(random.choice(self.sequences))
    target = sequence[1:] + [None]
    return (sequence, target)
