# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
Convergence simulations for abstract objects.
"""

import argparse
import collections
import io
import os
import random

import numpy as np

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment)
from two_layer_tracing import PIUNVisualizer as trace


def generateFeatures(numFeatures):
  """Return string features.

  If <=62 features are requested, output will be single character
  alphanumeric strings. Otherwise, output will be ["F1", "F2", ...]
  """
  # Capital letters, lowercase letters, numbers
  candidates = ([chr(i+65) for i in xrange(26)] +
                [chr(i+97) for i in xrange(26)] +
                [chr(i+48) for i in xrange(10)])

  if numFeatures > len(candidates):
    candidates = ["F{}".format(i) for i in xrange(numFeatures)]
    return candidates

  return candidates[:numFeatures]


def generateObjects(numObjects, featuresPerObject, objectWidth, featurePool):
  assert featuresPerObject <= (objectWidth ** 2)
  featureScale = 20

  locations = []
  for x in xrange(objectWidth):
    for y in xrange(objectWidth):
      locations.append((x, y))
  np.random.shuffle(locations)

  objects = []
  for o in xrange(numObjects):
    features = []
    for i in xrange(featuresPerObject):
      x, y = locations[i]
      featureName = random.choice(featurePool)
      features.append({"top": y*featureScale, "left": x*featureScale,
                       "width": featureScale, "height": featureScale,
                       "name": featureName})
    objects.append({"name": "Object {}".format(o), "features": features})
  return objects


def doExperiment(cellDimensions, cellCoordinateOffsets, numObjects,
                 featuresPerObject, objectWidth, numFeatures):
  """
  Learn a set of objects. Then try to recognize each object. Output an
  interactive visualization.

  @param cellDimensions (pair)
  The cell dimensions of each module

  @param cellCoordinateOffsets (sequence)
  The "cellCoordinateOffsets" parameter for each module
  """
  if not os.path.exists("traces"):
    os.makedirs("traces")

  features = generateFeatures(numFeatures)
  objects = generateObjects(numObjects, featuresPerObject, objectWidth,
                            features)

  locationConfigs = []
  scale = 40.0

  numModules = 20
  thresholds = 16
  perModRange = float(90.0 / float(numModules))
  for i in xrange(numModules):
    orientation = float(i) * perModRange

    locationConfigs.append({
      "cellDimensions": cellDimensions,
      "moduleMapDimensions": (scale, scale),
      "orientation": orientation,
      "cellCoordinateOffsets": cellCoordinateOffsets,
      "activationThreshold": 8,
      "initialPermanence": 1.0,
      "connectedPermanence": 0.5,
      "learningThreshold": 8,
      "sampleSize": 10,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.0,
    })
  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": thresholds,
    "reducedBasalThreshold": thresholds,
    "minThreshold": thresholds,
    "sampleSize": numModules,
    "cellsPerColumn": 16,
  }

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides)
  exp = PIUNExperiment(column, featureNames=features, numActiveMinicolumns=10)

  for objectDescription in objects:
    exp.learnObject(objectDescription)

  filename = "traces/{}-points-{}-cells-{}-objects-{}-feats.html".format(
    len(cellCoordinateOffsets)**2, np.prod(cellDimensions), numObjects, numFeatures)

  convergence = collections.defaultdict(int)
  with io.open(filename, "w", encoding="utf8") as fileOut:
    with trace(fileOut, exp, includeSynapses=False):
      print "Logging to", filename
      for objectDescription in objects:
        steps = exp.inferObjectWithRandomMovements(objectDescription)
        convergence[steps] += 1
        if steps is None:
          print 'Failed to infer object "{}"'.format(objectDescription["name"])

  for step, num in sorted(convergence.iteritems()):
    print "{}: {}".format(step, num)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--numObjects", type=int, required=True)
  parser.add_argument("--numUniqueFeatures", type=int, required=True)
  parser.add_argument("--locationModuleWidth", type=int, required=True)

  parser.add_argument("--coordinateOffsetWidth", type=int, default=7)

  args = parser.parse_args()

  numOffsets = args.coordinateOffsetWidth
  cellCoordinateOffsets = [i * (0.998 / (numOffsets-1)) + 0.001
                           for i in xrange(numOffsets)]
  doExperiment(
    cellDimensions=(args.locationModuleWidth, args.locationModuleWidth),
    cellCoordinateOffsets=cellCoordinateOffsets,
    numObjects=args.numObjects,
    featuresPerObject=10,
    objectWidth=4,
    numFeatures=args.numUniqueFeatures,
  )
