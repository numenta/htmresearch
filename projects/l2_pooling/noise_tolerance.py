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

from collections import defaultdict
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
  n = int(noiseLevel * 40)

  noised = set(pattern)

  noised.difference_update(random.sample(noised, n))

  for _ in xrange(n):
    while True:
      v = random.randint(0, totalNumCells - 1)
      if v not in pattern and v not in noised:
        noised.add(v)
        break

  return noised


def doExperiment(numColumns, l2Overrides, objectDescriptions, noiseFn,
                 numInitialTraversals):

  # For each column, keep a mapping from feature-location names to their SDRs
  layer4sdr = lambda : set(random.sample(xrange(L4_CELL_COUNT), 40))
  featureLocationSDRs = [defaultdict(layer4sdr) for _ in xrange(numColumns)]

  poolers = [ColumnPooler(inputWidth=L4_CELL_COUNT,
                          lateralInputWidths=[4096]*(numColumns-1),
                          **l2Overrides)
             for _ in xrange(numColumns)]

  # Learn the objects
  objectL2Representations = {}
  for objectName, featureLocations in  objectDescriptions.iteritems():
    for featureLocationName in featureLocations:
      for _ in xrange(10):
        allLateralInputs = [pooler.getActiveCells() for pooler in poolers]
        for columnNumber, pooler in enumerate(poolers):
          feedforwardInput = featureLocationSDRs[columnNumber][featureLocationName]
          lateralInputs = [lateralInput
                           for i, lateralInput in enumerate(allLateralInputs)
                           if i != columnNumber]
          pooler.compute(feedforwardInput, lateralInputs, learn=True)
    objectL2Representations[objectName] = [set(pooler.getActiveCells())
                                           for pooler in poolers]
    for pooler in poolers:
      pooler.reset()

  results = []

  # Try to infer the objects
  for objectName, featureLocations in objectDescriptions.iteritems():
    for pooler in poolers:
      pooler.reset()

    sensorPositionsIterator = greedySensorPositions(numColumns, len(featureLocations))

    # Touch each location at least numInitialTouches times, and then touch it
    # once more, testing it.
    numTouchesPerTraversal = len(featureLocations) / float(numColumns)
    numInitialTouches = int(math.ceil(numInitialTraversals * numTouchesPerTraversal))
    numTestTouches = int(math.ceil(1 * numTouchesPerTraversal))
    for touch in xrange(numInitialTouches + numTestTouches):
      sensorPositions = next(sensorPositionsIterator)
      for _ in xrange(3):
        allLateralInputs = [pooler.getActiveCells() for pooler in poolers]
        for columnNumber, pooler in enumerate(poolers):
          noiseLevel = max(0.0, min(1.0, noiseFn()))

          position = sensorPositions[columnNumber]
          featureLocationName = featureLocations[position]
          feedforwardInput = featureLocationSDRs[columnNumber][featureLocationName]
          feedforwardInput = noisy(feedforwardInput, noiseLevel, L4_CELL_COUNT)

          lateralInputs = [lateralInput
                           for i, lateralInput in enumerate(allLateralInputs)
                           if i != columnNumber]

          pooler.compute(feedforwardInput, lateralInputs, learn=False)

      if touch >= numInitialTouches:
        for columnNumber, pooler in enumerate(poolers):
          activeCells = set(pooler.getActiveCells())
          inferredCells = objectL2Representations[objectName][columnNumber]

          results.append((len(activeCells & inferredCells),
                          len(activeCells - inferredCells)))

  return results


def doConstantNoiseExperiment(numColumns, l2Overrides, objectDescriptions,
                              noiseLevel, numInitialTraversals):

  noiseFn = lambda: noiseLevel

  return doExperiment(numColumns, l2Overrides, objectDescriptions, noiseFn,
                      numInitialTraversals)


def doVaryingNoiseExperiment(numColumns, l2Overrides, objectDescriptions,
                             noiseMu, noiseSigma, numInitialTraversals):

  noiseFn = lambda: random.gauss(noiseMu, noiseSigma)

  return doExperiment(numColumns, l2Overrides, objectDescriptions, noiseFn,
                      numInitialTraversals)


def constantNoise_varyColumns():
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
        r = doConstantNoiseExperiment(numColumns, l2Overrides,
                                      objectDescriptions, noiseLevel,
                                      numInitialTraversals=6)
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
  plt.xlabel("Feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with constant noise")

  plotPath = os.path.join("plots",
                          "constantNoise_successRate_varyColumnCount_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def constantNoise_varyDistalSampleSize():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
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
        r = doConstantNoiseExperiment(numColumns, l2Overrides,
                                      objectDescriptions, noiseLevel,
                                      numInitialTraversals=6)
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
  plt.xlabel("Feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with constant noise")

  plotPath = os.path.join("plots",
                          "constantNoise_successRate_varyDistalSampleSize_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def constantNoise_varyDistalThreshold():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  thresholds = [2, 5, 13, 17, 20]
  numColumns = 3

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for activationThresholdDistal in thresholds:
      print "activationThresholdDistal", activationThresholdDistal
      l2Overrides = {"activationThresholdDistal": activationThresholdDistal}
      for noiseLevel in noiseLevels:
        r = doConstantNoiseExperiment(numColumns, l2Overrides,
                                      objectDescriptions, noiseLevel,
                                      numInitialTraversals=6)
        results[(activationThresholdDistal, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colorList = dict(zip(thresholds,
                       ('r', 'k', 'g', 'b', 'y')))
  markerList = dict(zip(thresholds,
                        ('o', '*', 'D', 'x', 'o')))

  for activationThresholdDistal in thresholds:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(activationThresholdDistal, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colorList[activationThresholdDistal],
             marker=markerList[activationThresholdDistal])

  lgnd = plt.legend(["Distal threshold %d" % activationThresholdDistal
                     for activationThresholdDistal in thresholds],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with constant noise")

  plotPath = os.path.join("plots",
                          "constantNoise_successRate_varyDistalThreshold_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def constantNoise_varyProximalSampleSize():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
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
        r = doConstantNoiseExperiment(numColumns, l2Overrides,
                                      objectDescriptions, noiseLevel,
                                      numInitialTraversals=6)
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

  lgnd = plt.legend(["Proximal threshold %d" % sampleSizeProximal
                     for sampleSizeProximal in sampleSizes],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with constant noise")

  plotPath = os.path.join("plots",
                          "constantNoise_successRate_varyProximalSampleSize_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def constantNoise_varyProximalThreshold():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  thresholds = [2, 5, 10, 15, 20]
  numColumns = 3

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for minThresholdProximal in thresholds:
      print "minThresholdProximal", minThresholdProximal
      l2Overrides = {"minThresholdProximal": minThresholdProximal}
      for noiseLevel in noiseLevels:
        r = doConstantNoiseExperiment(numColumns, l2Overrides,
                                      objectDescriptions, noiseLevel,
                                      numInitialTraversals=6)
        results[(minThresholdProximal, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colorList = dict(zip(thresholds,
                       ('r', 'k', 'g', 'b', 'y')))
  markerList = dict(zip(thresholds,
                        ('o', '*', 'D', 'x', 'o')))

  for minThresholdProximal in thresholds:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(minThresholdProximal, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colorList[minThresholdProximal],
             marker=markerList[minThresholdProximal])

  lgnd = plt.legend(["Proximal threshold %d" % minThresholdProximal
                     for minThresholdProximal in thresholds],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with constant noise")

  plotPath = os.path.join("plots",
                          "constantNoise_successRate_varyProximalThreshold_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def varyingNoise_varyColumns():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  noiseSigma = 0.1
  l2Overrides = {"sampleSizeDistal": 20}
  columnCounts = [1, 2, 3, 4]

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for numColumns in columnCounts:
      print "numColumns", numColumns
      for noiseLevel in noiseLevels:
        r = doVaryingNoiseExperiment(numColumns, l2Overrides,
                                     objectDescriptions, noiseLevel, noiseSigma,
                                     numInitialTraversals=6)
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
  plt.title("Inference with normally distributed noise (stdev=0.1)")

  plotPath = os.path.join("plots",
                          "varyingNoise_successRate_varyColumnCount_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def varyingNoise_varyDistalSampleSize():
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
        r = doVaryingNoiseExperiment(numColumns, l2Overrides,
                                     objectDescriptions, noiseLevel, noiseSigma,
                                     numInitialTraversals=6)
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
                          "varyingNoise_successRate_varyDistalSampleSize_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def varyingNoise_varyDistalThreshold():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  noiseSigma = 0.1
  thresholds = [2, 5, 13, 17, 20]
  numColumns = 3

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for activationThresholdDistal in thresholds:
      print "activationThresholdDistal", activationThresholdDistal
      l2Overrides = {"activationThresholdDistal": activationThresholdDistal}
      for noiseLevel in noiseLevels:
        r = doVaryingNoiseExperiment(numColumns, l2Overrides,
                                     objectDescriptions, noiseLevel, noiseSigma,
                                     numInitialTraversals=6)
        results[(activationThresholdDistal, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colorList = dict(zip(thresholds,
                       ('r', 'k', 'g', 'b', 'y')))
  markerList = dict(zip(thresholds,
                        ('o', '*', 'D', 'x', 'o')))

  for activationThresholdDistal in thresholds:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(activationThresholdDistal, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colorList[activationThresholdDistal],
             marker=markerList[activationThresholdDistal])

  lgnd = plt.legend(["Distal threshold %d" % activationThresholdDistal
                     for activationThresholdDistal in thresholds],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Mean feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with normally distributed noise (stdev=0.1)")

  plotPath = os.path.join("plots",
                          "varyingNoise_successRate_varyDistalThreshold_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def varyingNoise_varyProximalSampleSize():
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
        r = doVaryingNoiseExperiment(numColumns, l2Overrides,
                                     objectDescriptions, noiseLevel, noiseSigma,
                                     numInitialTraversals=6)
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
                          "varyingNoise_successRate_varyProximalSampleSize_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


def varyingNoise_varyProximalThreshold():
  #
  # Run the experiment
  #
  noiseLevels = [x * 0.01 for x in xrange(0, 101, 5)]
  noiseSigma = 0.1
  thresholds = [2, 8, 10, 15, 20]
  numColumns = 3

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial
    objectDescriptions = createRandomObjectDescriptions(10, 10)

    for minThresholdProximal in thresholds:
      print "minThresholdProximal", minThresholdProximal
      l2Overrides = {"minThresholdProximal": minThresholdProximal}
      for noiseLevel in noiseLevels:
        r = doVaryingNoiseExperiment(numColumns, l2Overrides,
                                     objectDescriptions, noiseLevel, noiseSigma,
                                     numInitialTraversals=6)
        results[(minThresholdProximal, noiseLevel)].extend(r)

  #
  # Plot it
  #
  numCorrectActiveThreshold = 30
  numIncorrectActiveThreshold = 10

  plt.figure()
  colorList = dict(zip(thresholds,
                       ('r', 'k', 'g', 'b', 'y')))
  markerList = dict(zip(thresholds,
                        ('o', '*', 'D', 'x', 'o')))

  for minThresholdProximal in thresholds:
    y = []
    for noiseLevel in noiseLevels:
      trials = results[(minThresholdProximal, noiseLevel)]
      numPassed = len([True for numCorrect, numIncorrect in trials
                       if numCorrect >= numCorrectActiveThreshold
                       and numIncorrect <= numIncorrectActiveThreshold])
      y.append(numPassed / float(len(trials)))

    plt.plot(noiseLevels, y,
             color=colorList[minThresholdProximal],
             marker=markerList[minThresholdProximal])

  lgnd = plt.legend(["Proximal threshold %d" % minThresholdProximal
                     for minThresholdProximal in thresholds],
                    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
  plt.xlabel("Mean feedforward noise level")
  plt.xticks([0.01 * n for n in xrange(0, 101, 10)])
  plt.ylabel("Success rate")
  plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.title("Inference with normally distributed noise (stdev=0.1)")

  plotPath = os.path.join("plots",
                          "varyingNoise_successRate_varyProximalThreshold_%s.pdf"
                          % time.strftime("%Y%m%d-%H%M%S"))
  plt.savefig(plotPath, bbox_extra_artists=(lgnd,), bbox_inches="tight")
  print "Saved file %s" % plotPath


if __name__ == "__main__":
  constantNoise_varyColumns()
  constantNoise_varyDistalSampleSize()
  constantNoise_varyDistalThreshold()
  constantNoise_varyProximalSampleSize()
  constantNoise_varyProximalThreshold()
  varyingNoise_varyColumns()
  varyingNoise_varyDistalSampleSize()
  varyingNoise_varyDistalThreshold()
  varyingNoise_varyProximalSampleSize()
  varyingNoise_varyProximalThreshold()
