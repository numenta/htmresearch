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

"""
Test the noise tolerance of Layer 2 + Layer 4.

Perform an experiment to see if L2 eventually recognizes an object.
Test with various noise levels and with various column counts.
"""

from collections import defaultdict
import json
import math
import random
import os
import time

import numpy as np

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.sensor_placement import greedySensorPositions


TIMESTEPS_PER_SENSATION = 3
NUM_L4_COLUMNS = 2048


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


def createRandomObjects(numObjects, locationsPerObject, featurePoolSize):
  """
  Generate random objects.

  @param numObjects (int)
  The number of objects to generate

  @param locationsPerObject (int)
  The number of points on each object

  @param featurePoolSize (int)
  The number of possible features

  @return
  For example, { 0: [0, 1, 2],
                 1: [0, 2, 1],
                 2: [2, 0, 1], }
  is 3 objects. The first object has Feature 0 and Location 0, Feature 1 at
  Location 1, Feature 2 at location 2, etc.
  """

  allFeatures = range(featurePoolSize)
  allLocations = range(locationsPerObject)
  objects = dict((name,
                     [random.choice(allFeatures) for _ in xrange(locationsPerObject)])
                    for name in xrange(numObjects))

  return objects



def doExperiment(numColumns, objects, l2Overrides, noiseLevels, numInitialTraversals,
                 noisyFeature, noisyLocation):
  """
  Touch every point on an object 'numInitialTraversals' times, then evaluate
  whether it has inferred the object by touching every point once more and
  checking the number of correctly active and incorrectly active cells.

  @param numColumns (int)
  The number of sensors to use

  @param l2Overrides (dict)
  Parameters for the ColumnPooler

  @param objects (dict)
  A mapping of object names to their features.
  See 'createRandomObjects'.

  @param noiseLevels (list of floats)
  The noise levels to experiment with. The experiment is run once per noise
  level. Noise is applied at a constant rate to exactly one cortical column.
  It's applied to the same cortical column every time, and this is the cortical
  column that is measured.

  @param noisyFeature (bool)
  Whether to use a noisy feature

  @param noisyLocation (bool)
  Whether to use a noisy location
  """

  featureSDR = lambda : set(random.sample(xrange(NUM_L4_COLUMNS), 40))
  locationSDR = lambda : set(random.sample(xrange(1024), 40))

  featureSDRsByColumn = [defaultdict(featureSDR) for _ in xrange(numColumns)]
  locationSDRsByColumn = [defaultdict(locationSDR) for _ in xrange(numColumns)]

  exp = L4L2Experiment(
    "Experiment",
    numCorticalColumns=numColumns,
    inputSize=NUM_L4_COLUMNS,
    externalInputSize=1024,
    seed=random.randint(2048, 4096)
  )

  exp.learnObjects(
    dict((objectName,
          [dict((column,
                 (locationSDRsByColumn[column][location],
                  featureSDRsByColumn[column][features[location]]))
                for column in xrange(numColumns))
           for location in xrange(len(features))])
         for objectName, features in objects.iteritems()))

  results = defaultdict(list)

  for noiseLevel in noiseLevels:
    # Try to infer the objects
    for objectName, features in objects.iteritems():
      exp.sendReset()

      inferredL2 = exp.objectL2Representations[objectName]

      sensorPositionsIterator = greedySensorPositions(numColumns, len(features))

      # Touch each location at least numInitialTouches times, and then touch it
      # once more, testing it. For each traversal, touch each point on the object
      # ~once. Not once per sensor -- just once. So we translate the "number of
      # traversals" into a "number of touches" according to the number of sensors.
      numTouchesPerTraversal = len(features) / float(numColumns)
      numInitialTouches = int(math.ceil(numInitialTraversals * numTouchesPerTraversal))
      numTestTouches = len(features)

      for touch in xrange(numInitialTouches + numTestTouches):
        sensorPositions = next(sensorPositionsIterator)

        sensation = dict(
          (column,
           (locationSDRsByColumn[column][sensorPositions[column]],
            featureSDRsByColumn[column][features[sensorPositions[column]]]))
          for column in xrange(1, numColumns))

        # Add noise to the first column.
        featureSDR = featureSDRsByColumn[0][features[sensorPositions[0]]]
        if noisyFeature:
          featureSDR = noisy(featureSDR, noiseLevel, NUM_L4_COLUMNS)

        locationSDR = locationSDRsByColumn[0][sensorPositions[0]]
        if noisyLocation:
          locationSDR = noisy(locationSDR, noiseLevel, 1024)

        sensation[0] = (locationSDR, featureSDR)

        exp.infer([sensation]*TIMESTEPS_PER_SENSATION, reset=False,
                  objectName=objectName)

        if touch >= numInitialTouches:
          activeCells = exp.getL2Representations()[0]
          correctCells = inferredL2[0]
          results[noiseLevel].append((len(activeCells & correctCells),
                                      len(activeCells - correctCells)))

  return results


def logCellActivity_noisyFeature_varyNumColumns(name="cellActivity"):
  """
  Run the experiment, varying the column counts, and save each
    [# correctly active cells, # incorrectly active cells]
  pair to a JSON file that can be visualized.
  """
  noiseLevels = [0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                 0.80, 0.85, 0.90]
  l2Overrides = {"sampleSizeDistal": 20}
  columnCounts = [1, 2, 3, 4]

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial

    objects = createRandomObjects(10, 10, 10)

    for numColumns in columnCounts:
      print "numColumns", numColumns
      r = doExperiment(numColumns, objects, l2Overrides, noiseLevels,
                       numInitialTraversals=6, noisyFeature=True,
                       noisyLocation=False)

      for noiseLevel, counts in r.iteritems():
        results[(numColumns, noiseLevel)].extend(counts)

  d = []
  for (numColumns, noiseLevel), cellCounts in results.iteritems():
    d.append({"numColumns": numColumns,
              "noiseLevel": noiseLevel,
              "results": cellCounts})

  filename = os.path.join("plots",
                          "%s_%s.json"
                          % (name, time.strftime("%Y%m%d-%H%M%S")))
  with open(filename, "w") as fout:
    json.dump(d, fout)

  print "Wrote to", filename
  print "Visualize this file at: http://numenta.github.io/htmresearch/visualizations/grid-of-scatterplots/L2-columns-with-noise.html"


def logCellActivity_noisyLocation_varyNumColumns(name, objects):
  """
  Run the experiment, varying the column counts, and save each
    [# correctly active cells, # incorrectly active cells]
  pair to a JSON file that can be visualized.
  """
  noiseLevels = [0.0, 0.5, 1.0]
  l2Overrides = {"sampleSizeDistal": 20}
  columnCounts = [1]

  results = defaultdict(list)

  for trial in xrange(1):
    print "trial", trial

    for numColumns in columnCounts:
      print "numColumns", numColumns
      r = doExperiment(numColumns, objects, l2Overrides, noiseLevels, numInitialTraversals=6,
                       noisyFeature=False, noisyLocation=True)

      for noiseLevel, counts in r.iteritems():
        results[(numColumns, noiseLevel)].extend(counts)

  d = []
  for (numColumns, noiseLevel), cellCounts in results.iteritems():
    d.append({"numColumns": numColumns,
              "noiseLevel": noiseLevel,
              "results": cellCounts})

  filename = os.path.join("plots",
                          "%s_%s.json"
                          % (name, time.strftime("%Y%m%d-%H%M%S")))
  with open(filename, "w") as fout:
    json.dump(d, fout)

  print "Wrote to", filename
  print "Visualize this file at: http://numenta.github.io/htmresearch/visualizations/grid-of-scatterplots/L2-columns-with-noise.html"



if __name__ == "__main__":
  # Add noise to the L4 minicolumns SDR, observing the impact on L2. Add a
  # constant amount of noise to one cortical column, but leave the other
  # cortical columns untouched. Observe what happens as more untouched columns
  # are added.
  #
  # We find that the lateral input from other columns can help correctly active
  # cells inhibit cells that shouldn't be active, but it doesn't help increase
  # the number of correctly active cells. So the accuracy of inference is
  # improved, but the confidence of the inference isn't.
  print "Test: Noisy L4 minicolumn SDR"
  logCellActivity_noisyFeature_varyNumColumns(name="noisyFeatureSDR")


  # Add noise to L4's location input, observing the impact on L2. Add a constant
  # amount of noise to the location input of one cortical column, but leave the
  # other cortical columns untouched.
  #
  # We find that:
  # - After the L4 loses a sufficient amount of its location input, the L2 + L4
  #   becomes a classifier of bags-of-features. So any object that has a unique
  #   set of features (irrespective of location) can still be classified by the
  #   L2. L2 will infer a union of all objects that have the observed set of
  #   features.
  # - As we add cortical columns, the noisy column quickly becomes able to infer
  #   any object. The lateral input causes the union of objects in L2 to narrow
  #   down to the correct object. So it's not totally necessary for every
  #   cortical column to receive location input.

  # Random objects with a small feature pool. Once noisy, a single column never
  # infers an object, since most objects contain the exact same 3 features.

  print ("Test: Noisy L4 location SDR, "
         "using random objects with a feature pool size of 3")
  logCellActivity_noisyLocation_varyNumColumns(
    "smallFeaturePool", createRandomObjects(numObjects=10,
                                            locationsPerObject=10,
                                            featurePoolSize=3))


  # A larger feature pool. Once noisy, a single column infers an object > 75% of
  # the time, since many objects are unique in their set of features.

  print ("Test: Noisy L4 location SDR, "
         "using random objects with a feature pool size of 10")
  logCellActivity_noisyLocation_varyNumColumns(
    "mediumFeaturePool", createRandomObjects(numObjects=10,
                                             locationsPerObject=10,
                                             featurePoolSize=10))


  # Hand-crafted objects that use only the same features. With a noisy location,
  # the object is never inferred with a single column. L2 always infers the
  # union of the 3 objects.

  print ("Test: Noisy L4 location SDR, "
         "using hand-crafted objects made of the same features")
  logCellActivity_noisyLocation_varyNumColumns(
    name="sameBagsOfFeatures",
    objects={
    0: [0, 1, 2],
    1: [0, 2, 1],
    2: [2, 0, 1],
  })


  # Hand-crafted objects that are each a unique combination of features. The
  # object is always inferred with a single column.

  print ("Test: Noisy L4 location SDR, "
         "using hand-crafted objects made of the unique sets of features")
  logCellActivity_noisyLocation_varyNumColumns(
    name="uniqueBagsOfFeatures",
    objects={
    0: [0, 1, 2],
    1: [1, 2, 3],
    2: [2, 3, 0],
  })
