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
Measure how many objects the model can learn and recognize with sufficient
accuracy.
"""

import argparse
import collections
import io
import math
import os
import random
from multiprocessing import cpu_count, Pool
from copy import copy
import time
import json

import numpy as np

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment, PIUNExperimentMonitor)
from two_layer_tracing import PIUNVisualizer as trace

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def generateObjects(numObjects, featuresPerObject, objectWidth, numFeatures):
  assert featuresPerObject <= (objectWidth ** 2)
  featureScale = 20

  objectMap = {}
  for i in xrange(numObjects):
    obj = np.zeros((objectWidth ** 2,), dtype=np.int32)
    obj.fill(-1)
    obj[:featuresPerObject] = np.random.randint(numFeatures,
                                                size=featuresPerObject,
                                                dtype=np.int32)
    np.random.shuffle(obj)
    objectMap[i] = obj.reshape((4, 4))

  objects = []
  for o in xrange(numObjects):
    features = []
    for x in xrange(objectWidth):
      for y in xrange(objectWidth):
        feat = objectMap[o][x][y]
        if feat == -1:
          continue
        features.append({"left": y*featureScale, "top": x*featureScale,
                         "width": featureScale, "height": featureScale,
                         "name": str(feat)})
    objects.append({"name": str(o), "features": features})
  return objects


class PIUNCellActivityTracer(PIUNExperimentMonitor):
  def __init__(self, exp):
    self.exp = exp
    self.locationLayerTimelineByObject = {}
    self.inferredStepByObject = {}
    self.currentObjectName = None

  def afterLocationAnchor(self, **kwargs):
    self.locationLayerTimelineByObject[self.currentObjectName].append(
      [module.sensoryAssociatedCells.tolist()
       for module in self.exp.column.L6aModules])

  def beforeInferObject(self, obj):
    self.currentObjectName = obj["name"]
    self.locationLayerTimelineByObject[obj["name"]] = []

  def afterInferObject(self, obj, inferredStep):
    self.inferredStepByObject[obj["name"]] = inferredStep



def doExperiment(locationModuleWidth,
                 cellCoordinateOffsets,
                 initialIncrement,
                 minAccuracy,
                 capacityResolution,
                 featuresPerObject,
                 objectWidth,
                 numFeatures,
                 useTrace,
                 noiseFactor,
                 moduleNoiseFactor,
                 numModules,
                 thresholds,
                 seed1,
                 seed2,
                 anchoringMethod = "narrowing"):
  """
  Finds the capacity of the specified model and object configuration. The
  algorithm has two stages. First it finds an upper bound for the capacity by
  repeatedly incrementing the number of objects by initialIncrement. After it
  finds a number of objects that is above capacity, it begins the second stage:
  performing a binary search over the number of objects to find an exact
  capacity.

  @param initialIncrement (int)
  For example, if this number is 128, this method will test 128 objects, then
  256, and so on, until it finds an upper bound, then it will narrow in on the
  breaking point. This number can't be incorrect, but the simulation will be
  inefficient if it's too low or too high.

  @param capacityResolution (int)
  The resolution of the capacity. If capacityResolution=1, this method will find
  the exact capacity. If the capacityResolution is higher, the method will
  return a capacity that is potentially less than the actual capacity.

  @param minAccuracy (float)
  The recognition success rate that the model must achieve.
  """
  if not os.path.exists("traces"):
    os.makedirs("traces")

  if seed1 != -1:
    np.random.seed(seed1)

  if seed2 != -1:
    random.seed(seed2)

  features = [str(i) for i in xrange(numFeatures)]

  locationConfigs = []
  scale = 40.0

  if thresholds == -1:
    thresholds = int(math.ceil(numModules * 0.8))
  elif thresholds == 0:
    thresholds = numModules
  perModRange = float(90.0 / float(numModules))
  for i in xrange(numModules):
    orientation = (float(i) * perModRange) + (perModRange / 2.0)

    locationConfigs.append({
      "cellsPerAxis": locationModuleWidth,
      "scale": scale,
      "orientation": np.radians(orientation),
      "cellCoordinateOffsets": cellCoordinateOffsets,
      "activationThreshold": 8,
      "initialPermanence": 1.0,
      "connectedPermanence": 0.5,
      "learningThreshold": 8,
      "sampleSize": 10,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.0,
      "anchoringMethod": anchoringMethod,
    })
  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": thresholds,
    "reducedBasalThreshold": thresholds,
    "minThreshold": numModules,
    "sampleSize": numModules,
    "cellsPerColumn": 16,
  }

  increment = initialIncrement
  numObjects = 0
  accuracy = None
  foundUpperBound = False

  while True:
    currentNumObjects = numObjects + increment
    numFailuresAllowed = currentNumObjects * (1 - minAccuracy)
    print "Testing", currentNumObjects

    objects = generateObjects(currentNumObjects, featuresPerObject, objectWidth,
                              numFeatures)

    column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides)
    exp = PIUNExperiment(column, featureNames=features,
                         numActiveMinicolumns=10,
                         noiseFactor=noiseFactor,
                         moduleNoiseFactor=moduleNoiseFactor)

    for objectDescription in objects:
      exp.learnObject(objectDescription)

    numFailures = 0

    try:
      if useTrace:
        filename = os.path.join(
          SCRIPT_DIR,
          "traces/capacity-{}-points-{}-cells-{}-objects-{}-feats.html".format(
            len(cellCoordinateOffsets)**2, exp.column.L6aModules[0].numberOfCells(),
            numObjects, numFeatures)
        )
        traceFileOut = io.open(filename, "w", encoding="utf8")
        traceHandle = trace(traceFileOut, exp, includeSynapses=True)
        print "Logging to", filename

      for objectDescription in objects:
        numSensationsToInference = exp.inferObjectWithRandomMovements(
            objectDescription)
        if numSensationsToInference is None:
          numFailures += 1

          if numFailures > numFailuresAllowed:
            break
    finally:
      if useTrace:
        traceHandle.__exit__()
        traceFileOut.close()

    if numFailures < numFailuresAllowed:
      numObjects = currentNumObjects
      accuracy = float(currentNumObjects - numFailures) / currentNumObjects
    else:
      foundUpperBound = True

    if foundUpperBound:
      increment /= 2
      if increment < capacityResolution:
        break

  result = {
    "numObjects": numObjects,
    "accuracy": accuracy,
  }

  print result
  return result


def experimentWrapper(args):
  return doExperiment(**args)


def runMultiprocessNoiseExperiment(resultName, repeat, numWorkers,
                                   appendResults, **kwargs):
  """
  :param kwargs: Pass lists to distribute as lists, lists that should be passed intact as tuples.
  :return: results, in the format [(arguments, results)].  Also saved to json at resultName, in the same format.
  """
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

  newExperiments = []
  for experiment in experiments:
    for _ in xrange(repeat):
      newExperiments.append(copy(experiment))
  experiments = newExperiments

  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    rs = pool.map_async(experimentWrapper, experiments, chunksize=1)
    while not rs.ready():
      remaining = rs._number_left
      pctDone = 100.0 - (100.0*remaining) / len(experiments)
      print "    =>", remaining, "experiments remaining, percent complete=",pctDone
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

  if appendResults:
    try:
      with open(os.path.join(SCRIPT_DIR, resultName), "r") as f:
        existingResults = json.load(f)
        results = existingResults + results
    except IOError:
      pass

  with open(os.path.join(SCRIPT_DIR, resultName),"wb") as f:
    print "Writing results to {}".format(resultName)
    json.dump(results,f)

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--numUniqueFeatures", type=int, nargs="+", required=True)
  parser.add_argument("--locationModuleWidth", type=int, nargs="+", required=True)
  parser.add_argument("--initialIncrement", type=int, default=128)
  parser.add_argument("--capacityResolution", type=int, default=1)
  parser.add_argument("--minAccuracy", type=float, default=0.9)
  parser.add_argument("--coordinateOffsetWidth", type=int, default=2)
  parser.add_argument("--noiseFactor", type=float, nargs="+", required=False, default = 0)
  parser.add_argument("--moduleNoiseFactor", type=float, nargs="+", required=False, default=0)
  parser.add_argument("--useTrace", action="store_true")
  parser.add_argument("--numModules", type=int, nargs="+", default=[20])
  parser.add_argument("--seed1", type=int, default=-1)
  parser.add_argument("--seed2", type=int, default=-1)
  parser.add_argument(
    "--thresholds", type=int, default=-1, nargs="+",
    help=(
      "The TM prediction threshold. Defaults to ceil(numModules*0.8)."
      "Set to 0 for the threshold to match the number of modules."))
  parser.add_argument("--anchoringMethod", type = str, default="corners")
  parser.add_argument("--resultName", type = str, default="results.json")
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--appendResults", action="store_true")
  parser.add_argument("--numWorkers", type=int, default=cpu_count())

  args = parser.parse_args()

  numOffsets = args.coordinateOffsetWidth
  cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001
                                 for i in xrange(numOffsets)])

  if "both" in args.anchoringMethod:
    args.anchoringMethod = ["narrowing", "corners"]

  runMultiprocessNoiseExperiment(
    args.resultName, args.repeat, args.numWorkers, args.appendResults,
    locationModuleWidth=args.locationModuleWidth,
    cellCoordinateOffsets=cellCoordinateOffsets,
    initialIncrement=args.initialIncrement,
    minAccuracy=args.minAccuracy,
    capacityResolution=args.capacityResolution,
    featuresPerObject=10,
    objectWidth=4,
    numFeatures=args.numUniqueFeatures,
    useTrace=args.useTrace,
    noiseFactor=args.noiseFactor,
    moduleNoiseFactor=args.moduleNoiseFactor,
    numModules=args.numModules,
    thresholds=args.thresholds,
    anchoringMethod=args.anchoringMethod,
    seed1=args.seed1,
    seed2=args.seed2,
  )
