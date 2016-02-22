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

import random
import numpy as np

np.random.seed(1)

chars='BTSXPVE'
categoryList = ['B', 'T', 'S', 'X', 'P', 'V', 'E']

graph = [[(1,5),('T','P')] , [(1,2),('S','X')], \
           [(3,5),('S','X')], [(6,),('E')], \
           [(3,2),('V','P')], [(4,5),('V','T')] ]


def in_grammar(word):
    if word[0] != 'B':
        return False
    node = 0
    for c in word[1:]:
        transitions = graph[node]
        try:
            node = transitions[0][transitions[1].index(c)]
        except ValueError: # using exceptions for flow control in python is common
            return False
    return True

def sequenceToWord(sequence):
    """
    converts a sequence (one-hot) in a reber string
    """
    reberString = ''
    for i in xrange(len(sequence)):
        index = np.where(sequence[i]==1.)[0][0]
        reberString += chars[index]
    return reberString


def generateSequences(maxLength):
    """
    @param minLength (int): length of the sequence
    @return inchars (array): a generated reber Grammer
            outchars (array): possible next elements at each step
    """

    inchars = 'B'
    node = 0
    outchars = []
    while node != 6:
        # all possible transitions
        transitions = graph[node]

        outchars.append(transitions[1])

        i = np.random.randint(0, len(transitions[0]))
        inchars += transitions[1][i]

        node = transitions[0][i]

    inchars = inchars[:-1]
    if len(inchars) > maxLength:
        inchars = inchars[:maxLength]
        outchars = outchars[:maxLength]

    return inchars, outchars


def generateSequencesNumber(maxLength, seed):
    """
    @param maxLength (int): maximum length of the sequence
    @return inchars (array): a generated reber Grammer
            outchars (array): possible next elements at each step
    """

    inchars = [0]
    node = 0
    outchars = []
    random.seed(seed)
    while node != 6:
        # all possible transitions
        transitions = graph[node]

        i = random.randint(0, len(transitions[0]))
        inchars.append(transitions[0][i])
        outchars.append(transitions[0])
        node = transitions[0][i]

    if len(inchars) > maxLength:
        inchars = inchars[:maxLength]
        outchars = outchars[:maxLength]

    return inchars, outchars


def generateSequencesVector(maxLength):
    inchars, outchars = generateSequences(maxLength)
    inseq = []
    outseq= []
    for i,o in zip(inchars, outchars):
        inpt = np.zeros(7)
        inpt[chars.find(i)] = 1.
        outpt = np.zeros(7)
        for oo in o:
            outpt[chars.find(oo)] = 1.
        inseq.append(inpt)
        outseq.append(outpt)
    return inseq, outseq


def get_char_one_hot(char):
    char_oh = np.zeros(7)
    for c in char:
      char_oh[chars.find(c)] = 1.
    return [char_oh]

def get_n_examples(n, minLength=10):
    examples = []
    for i in xrange(n):
        examples.append(generateSequencesVector(minLength))
    return examples

emb_chars = "TP"


def get_one_embedded_example(minLength=10):
    i, o = generateSequencesVector(minLength)
    emb_char = emb_chars[np.random.randint(0, len(emb_chars))]
    new_in = get_char_one_hot(('B',))
    new_in += get_char_one_hot((emb_char,))
    new_out= get_char_one_hot(emb_chars)
    new_out+= get_char_one_hot('B',)
    new_in += i
    new_out += o
    new_in += get_char_one_hot(('E',))
    new_in += get_char_one_hot((emb_char,))
    new_out += get_char_one_hot((emb_char, ))
    new_out += get_char_one_hot(('E',))
    return new_in, new_out


def get_n_embedded_examples(n, minLength=10):
    examples = []
    for i in xrange(n):
        examples.append(get_one_embedded_example(minLength))
    return examples


def checkPrediction(possibleInputs, prediction):
  for i in xrange(len(possibleInputs)):
    if possibleInputs[i] == prediction:
      return True
  return False


def checkPrediction2(possibleOutcome, predictedOutcome):
  """
  :param possibleOutcome: list of all possible outcomes
  :param predictedOutcome: list of all predicted outomes
  :return missN: number of misses (a possible outcome not predicted)
            fpN: number of false positives (a predicted outcome is not possible to happen)
  """
  missN = 0
  for i in xrange(len(possibleOutcome)):
    miss = 1
    for j in xrange(len(predictedOutcome)):
      if predictedOutcome[j] == possibleOutcome[i]:
        miss = 0
    missN += miss

  fpN = 0
  for i in xrange(len(predictedOutcome)):
    fp = 1
    for j in xrange(len(possibleOutcome)):
      if predictedOutcome[i] == possibleOutcome[j]:
        fp = 0
    fpN += fp

  return (missN, fpN)


def getMatchingElements(overlap, thresh=20):
  matchIndex = np.where(np.greater(overlap, thresh))[0].astype('int')
  matchElement = []
  for _ in xrange(len(matchIndex)):
    matchElement.append(categoryList[matchIndex[_]])
  return matchElement

