# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016 - 2017, Numenta, Inc.  Unless you have an agreement
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
Test the noise tolerance of Layer 2 in isolation.

Perform an experiment to see if L2 eventually recognizes an object.
Test with various noise levels and with various column counts and synapse
sample sizes.
"""

from collections import defaultdict
import json
import math
import random
import os
import time

import matplotlib.pyplot as plt

from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.frameworks.layers.sensor_placement import greedySensorPositions

L4_CELL_COUNT = 8*1024


def createRandomObjectDescriptions(numObjects,
                                   numLocationsPerObject,
                                   featurePool=("A", "B", "C")):
  """
  Returns {"Object 1": [(0, "C"), (1, "B"), (2, "C"), ...],
           "Object 2": [(0, "C"), (1, "A"), (2, "B"), ...]}
  """
  return dict(("Object %d" % i,
               zip(xrange(numLocationsPerObject),
                   [random.choice(featurePool)
                    for _ in xrange(numLocationsPerObject)]))
              for i in xrange(1, numObjects + 1))


def noisy(pattern, noiseLevel, totalNumCells):
  """
  Generate a noisy copy of a pattern.

  Given number of active bits w = len(pattern),
  deactivate noiseLevel*w cells, and activate noiseLevel*w other cells.

  @param pattern (set)
  A set of active indices

  @param noiseLevel (float)
  The percentage of the bits to shuffle

  @param totalNumCells (int)
  The number of cells in the SDR, active and inactive

  @return (set)
  A noisy set of active indices
  """
  n = int(noiseLevel * len(pattern))

  noised = set(pattern)

  noised.difference_update(random.sample(noised, n))

  for _ in xrange(n):
    while True:
      v = random.randint(0, totalNumCells - 1)
      if v not in pattern and v not in noised:
        noised.add(v)
        break

  return noised


def doExperiment(numColumns, l2Overrides, objectDescriptions, noiseMu,
                 noiseSigma, numInitialTraversals, noiseEverywhere):
  """
  Touch every point on an object 'numInitialTraversals' times, then evaluate
  whether it has inferred the object by touching every point once more and
  checking the number of correctly active and incorrectly active cells.

  @param numColumns (int)
  The number of sensors to use

  @param l2Overrides (dict)
  Parameters for the ColumnPooler

  @param objectDescriptions (dict)
  A mapping of object names to their feature-locations.
  See 'createRandomObjectDescriptions'.

  @param noiseMu (float)
  The average amount of noise in a feedforward input. The noise level for each
  column's input is determined once per touch. It is a gaussian distribution
  with mean 'noiseMu' and sigma 'noiseSigma'.

  @param noiseSigma (float)
  The sigma for the gaussian distribution of noise levels. If the noiseSigma is
  0, then the noise level will always be 'noiseMu'.

  @param numInitialTraversals (int)
  The number of times to traverse the object before testing whether the object
  has been inferred.

  @param noiseEverywhere (bool)
  If true, add noise to every column's input, and record accuracy of every
  column. If false, add noise to one column's input, and only record accuracy
  of that column.
  """

  # For each column, keep a mapping from feature-location names to their SDRs
  layer4sdr = lambda : set(random.sample(xrange(L4_CELL_COUNT), 40))
  featureLocationSDRs = [defaultdict(layer4sdr) for _ in xrange(numColumns)]

  params = {"inputWidth": L4_CELL_COUNT,
            "lateralInputWidths": [4096]*(numColumns-1),
            "seed": random.randint(0, 1024)}
  params.update(l2Overrides)

  l2Columns = [ColumnPooler(**params)
               for _ in xrange(numColumns)]

  # Learn the objects
  objectL2Representations = {}
  for objectName, featureLocations in  objectDescriptions.iteritems():
    for featureLocationName in featureLocations:
      # Touch it enough times for the distal synapses to reach the
      # connected permanence, and then once more.
      for _ in xrange(4):
        allLateralInputs = [l2.getActiveCells() for l2 in l2Columns]
        for columnNumber, l2 in enumerate(l2Columns):
          feedforwardInput = featureLocationSDRs[columnNumber][featureLocationName]
          lateralInputs = [lateralInput
                           for i, lateralInput in enumerate(allLateralInputs)
                           if i != columnNumber]
          l2.compute(feedforwardInput, lateralInputs, learn=True)
    objectL2Representations[objectName] = [set(l2.getActiveCells())
                                           for l2 in l2Columns]
    for l2 in l2Columns:
      l2.reset()

  results = []

  # Try to infer the objects
  for objectName, featureLocations in objectDescriptions.iteritems():
    for l2 in l2Columns:
      l2.reset()

    sensorPositionsIterator = greedySensorPositions(numColumns, len(featureLocations))

    # Touch each location at least numInitialTouches times, and then touch it
    # once more, testing it. For each traversal, touch each point on the object
    # ~once. Not once per sensor -- just once. So we translate the "number of
    # traversals" into a "number of touches" according to the number of sensors.
    numTouchesPerTraversal = len(featureLocations) / float(numColumns)
    numInitialTouches = int(math.ceil(numInitialTraversals * numTouchesPerTraversal))

    if noiseEverywhere:
      numTestTouches = int(math.ceil(1 * numTouchesPerTraversal))
    else:
      numTestTouches = len(featureLocations)

    for touch in xrange(numInitialTouches + numTestTouches):
      sensorPositions = next(sensorPositionsIterator)

      # Give the system a few timesteps to settle, allowing lateral connections
      # to cause cells to be inhibited.
      for _ in xrange(3):
        allLateralInputs = [l2.getActiveCells() for l2 in l2Columns]
        for columnNumber, l2 in enumerate(l2Columns):
          position = sensorPositions[columnNumber]
          featureLocationName = featureLocations[position]
          feedforwardInput = featureLocationSDRs[columnNumber][featureLocationName]

          if noiseEverywhere or columnNumber == 0:
            noiseLevel = random.gauss(noiseMu, noiseSigma)
            noiseLevel = max(0.0, min(1.0, noiseLevel))
            feedforwardInput = noisy(feedforwardInput, noiseLevel, L4_CELL_COUNT)

          lateralInputs = [lateralInput
                           for i, lateralInput in enumerate(allLateralInputs)
                           if i != columnNumber]

          l2.compute(feedforwardInput, lateralInputs, learn=False)

      if touch >= numInitialTouches:
        if noiseEverywhere:
          for columnNumber, l2 in enumerate(l2Columns):
            activeCells = set(l2.getActiveCells())
            correctCells = objectL2Representations[objectName][columnNumber]
            results.append((len(activeCells & correctCells),
                            len(activeCells - correctCells)))
        else:
          activeCells = set(l2Columns[0].getActiveCells())
          correctCells = objectL2Representations[objectName][0]
          results.append((len(activeCells & correctCells),
                          len(activeCells - correctCells)))

  return results


def plotSuccessRate_varyNumColumns(noiseSigma, noiseEverywhere):
  """
  Run and plot the experiment, varying the number of cortical columns.
  """

  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  l2Overrides = {"sampleSizeDistal": 20}
  columnCounts = [1, 2, 3, 4]

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for numColumns in columnCounts:
      print "numColumns", numColumns
      for noiseLevel in noiseLevels:
        r = doExperiment(numColumns, l2Overrides, objectDescriptions,
                         noiseLevel, noiseSigma, numInitialTraversals=6,
                         noiseEverywhere=noiseEverywhere)
        results[(numColumns, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colors = dict(zip(columnCounts,
                    ('r', 'k', 'g', 'b')))
  markers = dict(zip(columnCounts,
                     ('o', '*', 'D', 'x')))

  for numColumns in columnCounts:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(numColumns, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colors[numColumns],
             marker=markers[numColumns])

  lgnd = plt.legend(["%d columns" % numColumns
                     for numColumns in columnCounts],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Mean feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with normally distributed noise (stdev=%.2f)" % noiseSigma)

  plotPath = os.path.join("plots",
                          "successRate_varyColumnCount_sigma%.2f_%s.pdf"
                          % (noiseSigma, time.strftime("%Y%m%d-%H%M%S")))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def plotSuccessRate_varyDistalSampleSize(noiseSigma, noiseEverywhere):
  """
  Run and plot the experiment, varying the distal sample size.
  """

  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  noiseSigma = 0.1
  sampleSizes = [13, 20, 30, 40]
  numColumns = 3

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for sampleSizeDistal in sampleSizes:
      print "sampleSizeDistal", sampleSizeDistal
      l2Overrides = {"sampleSizeDistal": sampleSizeDistal}
      for noiseLevel in noiseLevels:
        r = doExperiment(numColumns, l2Overrides, objectDescriptions,
                         noiseLevel, noiseSigma, numInitialTraversals=6,
                         noiseEverywhere=noiseEverywhere)
        results[(sampleSizeDistal, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colorList = dict(zip(sampleSizes,
                       ('r', 'k', 'g', 'b')))
  markerList = dict(zip(sampleSizes,
                        ('o', '*', 'D', 'x')))

  for sampleSizeDistal in sampleSizes:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(sampleSizeDistal, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colorList[sampleSizeDistal],
             marker=markerList[sampleSizeDistal])

  lgnd = plt.legend(["Distal sample size %d" % sampleSizeDistal
                     for sampleSizeDistal in sampleSizes],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Mean feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with normally distributed noise (stdev=0.1)")

  plotPath = os.path.join("plots",
                          "successRate_varyDistalSampleSize_sigma%.2f_%s.pdf"
                          % (noiseSigma, time.strftime("%Y%m%d-%H%M%S")))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def plotSuccessRate_varyProximalSampleSize(noiseSigma, noiseEverywhere):
  """
  Run and plot the experiment, varying the proximal sample size.
  """

  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  noiseSigma = 0.1
  sampleSizes = [13, 20, 30, 40]
  numColumns = 3

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for sampleSizeProximal in sampleSizes:
      print "sampleSizeProximal", sampleSizeProximal
      l2Overrides = {"sampleSizeProximal": sampleSizeProximal}
      for noiseLevel in noiseLevels:
        r = doExperiment(numColumns, l2Overrides, objectDescriptions,
                         noiseLevel, noiseSigma, numInitialTraversals=6,
                         noiseEverywhere=noiseEverywhere)
        results[(sampleSizeProximal, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colorList = dict(zip(sampleSizes,
                       ('r', 'k', 'g', 'b')))
  markerList = dict(zip(sampleSizes,
                        ('o', '*', 'D', 'x')))

  for sampleSizeProximal in sampleSizes:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(sampleSizeProximal, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colorList[sampleSizeProximal],
             marker=markerList[sampleSizeProximal])

  lgnd = plt.legend(["Proximal sample size %d" % sampleSizeProximal
                     for sampleSizeProximal in sampleSizes],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Mean feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with normally distributed noise (stdev=0.1)")

  plotPath = os.path.join("plots",
                          "successRate_varyProximalSampleSize_sigma%.2f_%s.pdf"
                          % (noiseSigma, time.strftime("%Y%m%d-%H%M%S")))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def logCellActivity_varyNumColumns(noiseSigma, noiseEverywhere):
  """
  Run the experiment, varying the column counts, and save each
    [# correctly active cells, # incorrectly active cells]
  pair to a JSON file that can be visualized.
  """
  noiseLevels = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  l2Overrides = {"sampleSizeDistal": 20}
  columnCounts = [1, 2, 3, 4, 5]

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for numColumns in columnCounts:
      print "numColumns", numColumns
      for noiseLevel in noiseLevels:
        r = doExperiment(numColumns, l2Overrides, objectDescriptions,
                         noiseLevel, noiseSigma, numInitialTraversals=6,
                         noiseEverywhere=noiseEverywhere)
        results[(numColumns, noiseLevel)].extend(r)

  d = []
  for (numColumns, noiseLevel), cellCounts in results.iteritems():
    d.append({"numColumns": numColumns,
              "noiseLevel": noiseLevel,
              "results": cellCounts})

  filename = os.path.join("plots",
                          "varyColumns_sigma%.2f_%s.json"
                          % (noiseSigma, time.strftime("%Y%m%d-%H%M%S")))
  with open(filename, "w") as fout:
    json.dump(d, fout)

  print "Wrote to", filename
  print "Visualize this file at: http://numenta.github.io/nupic.research/visualizations/grid-of-scatterplots/L2-columns-with-noise.html"



if __name__ == "__main__":

  # Plot the accuracy of inference when noise is added, varying the number of
  # cortical columns. We find that when noise is applied at a constant equal
  # rate for each column, the accuracy only improves slightly with more cortical
  # columns.
  plotSuccessRate_varyNumColumns(noiseSigma=0.0, noiseEverywhere=True)

  # Change noise to a Gaussian random variable that is independently applied to
  # different columns. We find that the accuracy now improves with more cortical
  # columns. This means that noisy sensors benefit from having lateral input
  # from non-noisy sensors. The sensors that happen to have high noise levels
  # take advantage of the sensors that happen to have low noise levels, so the
  # array as a whole can partially guard itself from noise.
  plotSuccessRate_varyNumColumns(noiseSigma=0.1, noiseEverywhere=True)
  plotSuccessRate_varyNumColumns(noiseSigma=0.2, noiseEverywhere=True)

  # Plot the accuracy of inference when noise is added, varying the ratio of the
  # proximal threshold to the proximal synapse sample size. We find that this
  # ratio does more than any other parameter to determine at what noise level
  # the accuracy drop-off occurs.
  plotSuccessRate_varyProximalSampleSize(noiseSigma=0.1, noiseEverywhere=True)

  # Plot the accuracy of inference when noise is added, varying the ratio of the
  # distal segment activation threshold to the distal synapse sample size. We
  # find that increasing this ratio provides additional noise tolerance on top
  # of the noise tolerance provided by proximal connections.
  plotSuccessRate_varyDistalSampleSize(noiseSigma=0.1, noiseEverywhere=True)

  # Observe the impact of columns without noisy input on columns with noisy
  # input. Add constant noise to one column's input, and don't add noise for the
  # other columns. Observe what happens as more non-noisy columns are added. We
  # find that the lateral input from other columns can help correctly active
  # cells inhibit cells that shouldn't be active, but it doesn't help increase
  # the number of correctly active cells. So the accuracy of inference is
  # improved, but the confidence of the inference isn't.
  logCellActivity_varyNumColumns(noiseSigma=0.0, noiseEverywhere=False)
