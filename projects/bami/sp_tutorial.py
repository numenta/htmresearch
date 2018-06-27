#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

""" A simple tutorial that shows some features of the Spatial Pooler.

The following program has the purpose of presenting some
basic properties of the Spatial Pooler. It reproduces Figs.
5, 7 and 9 from this paper: http://arxiv.org/abs/1505.02142

To learn more about the Spatial Pooler have a look at BAMI:
http://numenta.com/biological-and-machine-intelligence/

or at its class reference in the NuPIC documentation:
http://numenta.org/docs/nupic/classnupic_1_1research_1_1spatial__pooler_1_1_spatial_pooler.html
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import random

from nupic.research.spatial_pooler import SpatialPooler as SP



SP_PARAMS = {
  "inputDimensions": (1000, 1),
  "columnDimensions": (1024, 1),
  "potentialRadius": 1024,
  "potentialPct": 0.5,
  "globalInhibition": True,
  "localAreaDensity": .02,
  "numActiveColumnsPerInhArea": -1,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 1.0,
  "seed": 1936
}


def pctOverlap(x1, x2, size):
  """
  Computes the percentage of overlap between vectors x1 and x2.
  Parameters:
  ----------
  x1: binary vector whose size is specified in parameter size
  x2: binary vector whose size is specified in parameter size
  size: length of binary vectors
  """
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)
  minX1X2 = min(nonZeroX1, nonZeroX2)
  pctOverlap = 0
  if minX1X2 > 0:
    pctOverlap = float(np.dot(x1, x2))/float(minX1X2)
  return pctOverlap



def corruptVector(vector, noiseLevel):
  """
  Corrupts a binary vector by swaping its inner values depending
  on the amount of noise specified by the user.
  Parameters:
  ----------
  vector: binary vector to be corrupted
  noiseLevel: Amount of noise to be applied on the vector.
  Real value from interval [0,1]. 0 means no noise and the vector
  remains the same. 1 is perfect corruption and all bits in the
  binary vector are swapped.
  """
  size = len(vector)
  for i in range(size):
    rnd = random.random()
    if rnd < noiseLevel:
      if vector[i] == 1:
        vector[i] = 0
      else:
        vector[i] = 1



def resetVector(x1, x2):
  """
  Copies the contents of vector x1 into vector x2.
  Parameters:
  ----------
  x1: binary vector to be copied
  x2: binary vector where x1 is copied
  """
  size = len(x1)
  for i in range(size):
    x2[i] = x1[i]


def run():
  # Initialize the SP
  inputSize = 1000;
  SP_PARAMS["inputDimensions"] = (inputSize, 1)
  SP_PARAMS["potentialRadius"] = inputSize
  sp = SP(**SP_PARAMS)
  print
  print "Spatial Pooler created with the following parameters:"
  pprint(SP_PARAMS)

  np.random.seed(42)

  # Part 1:
  # -------
  # A column connects to a region of the input vector (specified
  # by both the potentialRadius and potentialPct). The overlap score
  # for a column is the number of connections to the input that become
  # active when presented with a vector. When learning is 'on' in the SP,
  # the active connections are reinforced, whereas those inactive are
  # depressed (according to parameters synPermActiveInc and synPermInactiveDec.
  # In order for the SP to create a sparse representation of the input, it
  # will select a small percentage (usually 2%) of its most active columns,
  # ie. columns with the largest overlap score.
  # In this first part, we will create a histogram showing the overlap scores
  # of the Spatial Pooler (SP) after feeding it with a random binary
  # input. As well, the histogram will show the scores of those columns
  # that are chosen to build the sparse representation of the input.

  uintType = "uint32"
  inputArray = np.random.randint(0, 2, size=inputSize, dtype=uintType)
  outputCols = np.zeros(SP_PARAMS["columnDimensions"][0], dtype=uintType)
  learning = False
  sp.compute(inputArray, learning, outputCols)
  # import pdb; pdb.set_trace()
  overlaps = sp.getOverlaps()
  activeColsScores = [overlaps[i] for i in outputCols.nonzero()]

  print ""
  print "---------------------------------"
  print "Figure 1 shows an histogram of the overlap scores"
  print "from all the columns in the spatial pooler, as well as the"
  print "overlap scores of those columns that were selected to build a"
  print "sparse representation of the input (shown in green)."
  print "The SP chooses 2% of the columns with the largest overlap score"
  print "to make such sparse representation."
  print "---------------------------------"
  print ""

  bins = np.linspace(min(overlaps), max(overlaps), 50)
  plt.hist(overlaps, bins, alpha=0.5, label='All cols')
  plt.hist(activeColsScores, bins, alpha=0.5, label='Active cols')
  plt.legend(loc='upper right')
  plt.xlabel("Overlap scores")
  plt.ylabel("Frequency")
  plt.title("Figure 1")
  plt.show()

  import pdb; pdb.set_trace()

  # Part 2a:
  # -------
  # The input overlap between two binary vectors is defined as their dot product. In order
  # to normalize this value we divide it by the number of minimum number of active inputs
  # (in either vector). This means we are considering the sparser vector as reference.
  # Two identical binary vectors will have an input overlap of 1, whereas two complete
  # different vectors (one is the logical NOT of the other) will yield an overlap of 0.
  # In this section we will see how the input overlap of two binary vectors decrease as we
  # add noise to one of them.

  inputX1 = np.zeros(inputSize, dtype=uintType)
  inputX2 = np.zeros(inputSize, dtype=uintType)
  outputX1 = np.zeros(columnNumber, dtype=uintType)
  outputX2 = np.zeros(columnNumber, dtype=uintType)

  inputArray = np.random.randint(0, 2, size=inputSize, dtype=uintType)

  x = []
  y = []
  for noiseLevel in np.arange(0, 1.1, 0.1):
    resetVector(inputX1, inputX2)
    corruptVector(inputX2, noiseLevel)
    x.append(noiseLevel)
    y.append(pctOverlap(inputX1, inputX2, inputSize))

  print ""
  print "---------------------------------"
  print "Figure 2 shows the input overlap between 2 identical binary"
  print "vectors in function of the noise applied to one of them."
  print "0 noise level means that the vector remains the same, whereas"
  print "1 means that the vector is the logical negation of the original"
  print "vector."
  print "The relationship between overlap and noise level is practically"
  print "linear and monotonically decreasing."
  print "---------------------------------"
  print ""

  plt.plot(x, y)
  plt.xlabel("Noise level")
  plt.ylabel("Input overlap")
  plt.title("Figure 2")
  plt.show()

  # Part 2b:
  # -------
  # The output overlap between two binary input vectors is the overlap of the
  # columns that become active once they are fed to the SP. In this part we
  # turn learning off, and observe the output of the SP as we input two binary
  # input vectors with varying level of noise.
  # Starting from two identical vectors (that yield the same active columns)
  # we would expect that as we add noise to one of them their output overlap
  # decreases.
  # In this part we will show how the output overlap behaves in function of the
  # input overlap between two vectors.

  x = []
  y = []
  for noiseLevel in np.arange(0, 1.1, 0.1):
    resetVector(inputX1, inputX2)
    corruptVector(inputX2, noiseLevel)

    sp.compute(inputX1, False, outputX1)
    sp.compute(inputX2, False, outputX2)

    x.append(pctOverlap(inputX1, inputX2, inputSize))
    y.append(pctOverlap(outputX1, outputX2, columnNumber))

  print ""
  print "---------------------------------"
  print "Figure 3 shows the output overlap between two sparse representations"
  print "in function of their input overlap. Starting from two identical binary vectors"
  print "(which yield the same active columns) we add noise two one of them"
  print "feed it to the SP, and estimate the output overlap between the two"
  print "representations in terms of the common active columns between them."
  print "As expected, as the input overlap decrease, so does the output overlap."
  print "However, this relationship is sublinear, implying that the SP is very"
  print "sensitive to noise."
  print "---------------------------------"
  print ""

  plt.plot(x, y)
  plt.xlabel("Input overlap")
  plt.ylabel("Output overlap")
  plt.title("Figure 3")
  plt.show()

  # Part 3:
  # -------
  # As we saw in the last section, an untrained SP is very sensitive to noise in a binary vector
  # in such a way that the output overlap between a vector and a corrupted version of it decreases
  # dramatically as the amount of noise grows. To improve this situation we train the SP by
  # turning learning on, and by exposing it to a variety of random binary vectors.
  # We will expose the SP to a repetition of input patterns in order to make it learn and distinguish
  # them once learning is over. This will result in robustness to noise in the inputs.
  # In this section we will reproduce the plot in the last section after the SP has learned a series
  # of inputs. Here we will see how the SP exhibits some resilience to noise after learning.

  # We will present 10 random vectors to the SP, and repeat this 30 times.
  # Try changing the number of times we do this to see how it changes the last plot.
  # Then, you could also modify the number of examples to see how the SP behaves.
  # Is there a relationship between the number of examples and the number of times that
  # we expose them to the SP?

  numExamples = 10
  inputVectors = np.zeros((numExamples, inputSize), dtype=uintType)
  outputColumns = np.zeros((numExamples, columnNumber), dtype=uintType)

  for i in range(numExamples):
    for j in range(inputSize):
      inputVectors[i][j] = random.randrange(2)

  # This is the number of times that we will present the input vectors to the SP
  epochs = 30

  for _ in range(epochs):
    for i in range(numExamples):
      #Feed the examples to the SP
      sp.compute(inputVectors[i][:], True, outputColumns[i][:])

  inputVectorsCorrupted = np.zeros((numExamples, inputSize), dtype=uintType)
  outputColumnsCorrupted = np.zeros((numExamples, columnNumber), dtype=uintType)

  x = []
  y = []
  # We will repeat the experiment in the last section for only one input vector
  # in the set of input vectors
  for noiseLevel in np.arange(0, 1.1, 0.1):
    resetVector(inputVectors[0][:], inputVectorsCorrupted[0][:])
    corruptVector(inputVectorsCorrupted[0][:], noiseLevel)

    sp.compute(inputVectors[0][:], False, outputColumns[0][:])
    sp.compute(inputVectorsCorrupted[0][:], False, outputColumnsCorrupted[0][:])

    x.append(pctOverlap(inputVectors[0][:], inputVectorsCorrupted[0][:], inputSize))
    y.append(pctOverlap(outputColumns[0][:], outputColumnsCorrupted[0][:], columnNumber))

  print ""
  print "---------------------------------"
  print "How robust is the SP to noise after learning?"
  print "Figure 4 shows again the output overlap between two binary vectors in function"
  print "of their input overlap. After training, the SP exhibits more robustness to noise"
  print "in its input, resulting in a -almost- sigmoid curve. This implies that even if a"
  print "previous input is presented again with a certain amount of noise its sparse"
  print "representation still resembles somehow its original version."
  print "---------------------------------"
  print ""

  plt.plot(x, y)
  plt.xlabel("Input overlap")
  plt.ylabel("Output overlap")
  plt.title("Figure 4")
  plt.show()



if __name__ == "__main__":
  run()
