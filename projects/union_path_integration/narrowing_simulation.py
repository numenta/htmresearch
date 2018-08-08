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
from multiprocessing import cpu_count, Pool
from copy import copy
import time
import json
import StringIO

import numpy as np

np.random.seed(865486387)
random.seed(357627)

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment)
from two_layer_tracing import PIUNVisualizer as trace
from two_layer_tracing import PIUNLogger as rawTrace

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def generateObjects(numObjects, featuresPerObject, objectWidth, numFeatures):
  assert featuresPerObject <= (objectWidth ** 2)
  featureScale = 20

  np.random.seed(numObjects)
  objectMap = {}
  for i in xrange(numObjects):
    obj = np.zeros((objectWidth ** 2,), dtype=np.int32)
    obj.fill(-1)
    obj[:featuresPerObject] = np.random.randint(numFeatures, size=featuresPerObject, dtype=np.int32)
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


def doExperiment(cellDimensions,
                 cellCoordinateOffsets,
                 numObjects,
                 featuresPerObject,
                 objectWidth,
                 numFeatures,
                 useTrace,
                 useRawTrace,
                 noiseFactor,
                 moduleNoiseFactor,
                 numModules,
                 thresholds,
                 anchoringMethod = "narrowing"):
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

  features = [str(i) for i in xrange(numFeatures)]
  objects = generateObjects(numObjects, featuresPerObject, objectWidth,
                            numFeatures)

  locationConfigs = []
  scale = 40.0

  if thresholds is None:
    thresholds = int(((numModules + 1)*0.8))
  elif thresholds == 0:
    thresholds = numModules
  perModRange = float(90.0 / float(numModules))
  for i in xrange(numModules):
    orientation = (float(i) * perModRange) + (perModRange / 2.0)

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

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides)
  exp = PIUNExperiment(column, featureNames=features,
                       numActiveMinicolumns=10,
                       noiseFactor=noiseFactor,
                       moduleNoiseFactor=moduleNoiseFactor)

  for objectDescription in objects:
    exp.learnObject(objectDescription)

  filename = os.path.join(
      SCRIPT_DIR,
      "traces/{}-points-{}-cells-{}-objects-{}-feats.html".format(
          len(cellCoordinateOffsets)**2, np.prod(cellDimensions),
          numObjects, numFeatures)
  )
  rawFilename = os.path.join(
      SCRIPT_DIR,
      "traces/{}-points-{}-cells-{}-objects-{}-feats.trace".format(
          len(cellCoordinateOffsets)**2, np.prod(cellDimensions),
          numObjects, numFeatures)
  )

  assert not (useTrace and useRawTrace), "Cannot use both --trace and --rawTrace"

  convergence = collections.defaultdict(int)
  if useTrace:
    with io.open(filename, "w", encoding="utf8") as fileOut:
      with trace(fileOut, exp, includeSynapses=True):
        print "Logging to", filename
        for objectDescription in objects:
          random.seed(int(objectDescription["name"]))
          steps = exp.inferObjectWithRandomMovements(objectDescription)
          convergence[steps] += 1
          if steps is None:
            print 'Failed to infer object "{}"'.format(objectDescription["name"])
  elif useRawTrace:
    with io.open(rawFilename, "w", encoding="utf8") as fileOut:
      strOut = StringIO.StringIO()
      with rawTrace(strOut, exp, includeSynapses=False):
        print "Logging to", filename
        for objectDescription in objects:
          steps = exp.inferObjectWithRandomMovementsNoStopping(objectDescription, steps=10)
          convergence[steps] += 1
          if steps is None:
            print 'Failed to infer object "{}"'.format(objectDescription["name"])
      fileOut.write(unicode(strOut.getvalue()))
  else:
    print "Logging to", filename
    for objectDescription in objects:
      steps = exp.inferObjectWithRandomMovements(objectDescription)
      convergence[steps] += 1
      if steps is None:
        print 'Failed to infer object "{}"'.format(objectDescription["name"])

  for step, num in sorted(convergence.iteritems()):
    print "{}: {}".format(step, num)

  return(convergence)


def experimentWrapper(args):
  return doExperiment(**args)

def runMultiprocessNoiseExperiment(resultName, repeat, **kwargs):
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

  numWorkers = cpu_count()
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
      result.append(doExperiment(arg))

  # Save results for later use
  results = [(arg,res) for arg, res in zip(experiments, result)]
  with open(os.path.join(SCRIPT_DIR, resultName),"wb") as f:
    json.dump(results,f)

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--numObjects", type=int, required=True)
  parser.add_argument("--numUniqueFeatures", type=int, required=True)
  parser.add_argument("--locationModuleWidth", type=int, required=True)
  parser.add_argument("--coordinateOffsetWidth", type=int, default=2)
  parser.add_argument("--noiseFactor", type=float, required=False, default = 0)
  parser.add_argument("--moduleNoiseFactor", type=float, required=False, default=0)
  parser.add_argument("--useTrace", action="store_true")
  parser.add_argument("--useRawTrace", action="store_true")
  parser.add_argument("--numModules", type=int, default=20)
  parser.add_argument(
    "--thresholds", type=int, default=None,
    help=(
      "The TM prediction threshold. Defaults to int((numModules+1)*0.8)."
      "Set to 0 for the threshold to match the number of modules."))
  parser.add_argument("--anchoringMethod", type = str, default="corners")
  parser.add_argument("--resultName", type = str, default="results.json")
  parser.add_argument("--repeat", type=int, default=1)

  args = parser.parse_args()

  numOffsets = args.coordinateOffsetWidth
  cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001 for i in xrange(numOffsets)])

  if "both" in args.anchoringMethod:
    args.anchoringMethod = ["narrowing", "reanchoring"]


  doExperiment(
    cellDimensions=(args.locationModuleWidth, args.locationModuleWidth),
    cellCoordinateOffsets=cellCoordinateOffsets,
    numObjects=args.numObjects,
    featuresPerObject=10,
    objectWidth=4,
    numFeatures=args.numUniqueFeatures,
    useTrace=args.useTrace,
    useRawTrace=args.useRawTrace,
    noiseFactor=args.noiseFactor,
    moduleNoiseFactor=args.moduleNoiseFactor,
    numModules=args.numModules,
    thresholds=args.thresholds,
    anchoringMethod=args.anchoringMethod,
  )
