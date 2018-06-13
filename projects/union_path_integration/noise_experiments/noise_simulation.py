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
Noise simulations for convergence and capacity, using the L4-L6 location
network.
"""

import argparse
import collections
import io
import os
import random
from multiprocessing import cpu_count, Pool
from copy import copy
import time
import json

import numpy as np

np.random.seed(865486387)
random.seed(357627)

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment)
from two_layer_tracing import PIUNVisualizer as trace

# Argparse hack for handling boolean inputs, from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

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

  objects = []
  for o in xrange(numObjects):
    np.random.shuffle(locations)
    features = []
    for i in xrange(featuresPerObject):
      x, y = locations[i]
      featureName = random.choice(featurePool)
      features.append({"top": y*featureScale, "left": x*featureScale,
                       "width": featureScale, "height": featureScale,
                       "name": featureName})
    objects.append({"name": "Object {}".format(o), "features": features})
  return objects


def doExperiment(cellDimensions,
                 cellCoordinateOffsets,
                 numObjects,
                 featuresPerObject,
                 objectWidth,
                 numFeatures,
                 useTrace,
                 noiseFactor,
                 moduleNoiseFactor,
                 anchoringMethod="narrowing",
                 randomLocation=False,
                 threshold=16):
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
  scale = 5*cellDimensions[0] # One cell is about a quarter of a feature

  numModules = 20
  perModRange = float(90.0 / float(numModules))

  if anchoringMethod == "corners":
    cellCoordinateOffsets = (.0001, .5, .9999)

  if anchoringMethod == "discrete":
    cellCoordinateOffsets = (.5,)

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
      "sampleSize": 20,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.0,
      "anchoringMethod": anchoringMethod,
    })
  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": threshold,
    "reducedBasalThreshold": threshold,
    "minThreshold": threshold,
    "sampleSize": numModules,
    "cellsPerColumn": 16,
  }

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides)
  exp = PIUNExperiment(column, featureNames=features,
                       numActiveMinicolumns=10,
                       noiseFactor=noiseFactor,
                       moduleNoiseFactor=moduleNoiseFactor)

  for objectDescription in objects:
    exp.learnObject(objectDescription, randomLocation=randomLocation, useNoise = False)
    print 'Learned object {}'.format(objectDescription["name"])

  filename = "traces/{}-points-{}-cells-{}-objects-{}-feats-{}-random.html".format(
    len(cellCoordinateOffsets)**2, np.prod(cellDimensions), numObjects, numFeatures, randomLocation)

  convergence = collections.defaultdict(int)
  if useTrace:
    with io.open(filename, "w", encoding="utf8") as fileOut:
      with trace(fileOut, exp, includeSynapses=False):
        print "Logging to", filename
        for objectDescription in objects:
          steps = exp.inferObjectWithRandomMovements(objectDescription, randomLocation=randomLocation)
          convergence[steps] += 1
          if steps is None:
            print 'Failed to infer object "{}"'.format(objectDescription["name"])
          else:
            print 'Inferred object {} after {} steps'.format(objectDescription["name"], steps)
  else:
    for objectDescription in objects:
      steps = exp.inferObjectWithRandomMovements(objectDescription, randomLocation=randomLocation)
      convergence[steps] += 1
      if steps is None:
        print 'Failed to infer object "{}"'.format(objectDescription["name"])
      else:
        print 'Inferred object {} after {} steps'.format(objectDescription["name"], steps)

  for step, num in sorted(convergence.iteritems()):
    print "{}: {}".format(step, num)

  return(convergence)


def experimentWrapper(args):
  return doExperiment(**args)

def runMultiprocessNoiseExperiment(resultName=None, numWorkers = 0, **kwargs):
  """
  :param kwargs: Pass lists to distribute as lists, lists that should be passed intact as tuples.
  :return: results, in the format [(arguments, results)].  Also saved to json at resultName, in the same format.
  """

  if resultName is None:
    resultName = str(kwargs) + ".json"

  experiments = [{}]
  for key, values in kwargs.items():
    if type(values) is list:
      newExperiments = []
      for experiment in experiments:
        for val in values:
          newExperiment = copy(experiment)
          newExperiment[key] = val
          newExperiments.append(newExperiment)
      experiments = newExperiments
    else:
      [a.__setitem__(key, values) for a in experiments]

  if numWorkers == 0:
    numWorkers = cpu_count()
  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    rs = pool.map_async(experimentWrapper, experiments, chunksize=1)
    while not rs.ready():
      remaining = rs._number_left
      pctDone = 100.0 - (100.0*remaining) / len(experiments)
      print "    => {} experiments remaining, percent complete={}".format(\
        remaining, pctDone)
      time.sleep(5)
    pool.close()  # No more work
    pool.join()
    result = rs.get()
  else:
    result = []
    for arg in experiments:
      result.append(doExperiment(**arg))

  # Save results for later use
  results = [(arg,res) for arg, res in zip(experiments, result)]
  with open(resultName,"wb") as f:
    json.dump(results,f)

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--numObjects", type=int, nargs="+", required=True)
  parser.add_argument("--numUniqueFeatures", type=int, required=True)
  parser.add_argument("--locationModuleWidth", type=int, required=True)
  parser.add_argument("--coordinateOffsetWidth", type=int, default=7)
  parser.add_argument("--noiseFactor", type=float, nargs="+", required=False, default = 0)
  parser.add_argument("--moduleNoiseFactor", type=float, nargs="+", required=False, default = 0)
  parser.add_argument("--useTrace", action="store_true")
  parser.add_argument("--resultName", type = str, default = "results.json")
  parser.add_argument("--anchoringMethod", type = str, nargs = "+", default = "narrowing")
  parser.add_argument("--randomLocation", type = str2bool, nargs = "+", default = False)
  parser.add_argument("--numWorkers", type = int, default = 0)
  parser.add_argument("--threshold", nargs="+", type = int, default = 16)

  args = parser.parse_args()

  numOffsets = args.coordinateOffsetWidth
  cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001 for i in xrange(numOffsets)])

  if "all" in args.anchoringMethod or "both" in args.anchoringMethod:
    args.anchoringMethod = ["narrowing", "corners"]


  runMultiprocessNoiseExperiment(args.resultName,
    cellDimensions=(args.locationModuleWidth, args.locationModuleWidth),
    cellCoordinateOffsets=cellCoordinateOffsets,
    numObjects=args.numObjects,
    featuresPerObject=10,
    objectWidth=4,
    numFeatures=args.numUniqueFeatures,
    useTrace=args.useTrace,
    noiseFactor=args.noiseFactor,
    moduleNoiseFactor=args.moduleNoiseFactor,
    anchoringMethod=args.anchoringMethod,
    numWorkers=args.numWorkers,
    randomLocation=args.randomLocation,
    threshold=args.threshold,
  )
