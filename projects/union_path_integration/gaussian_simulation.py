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
Recognition experiments for abstract objects, using a gaussian bump grid
cell module model.
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

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment)
from two_layer_tracing import PIUNVisualizer as trace
from two_layer_tracing import PIUNLogger as rawTrace

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def generateObjects(numObjects, featuresPerObject, objectWidth, numFeatures):
  assert featuresPerObject <= (objectWidth ** 2)
  featureScale = 20

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


def doExperiment(numObjects,
                 featuresPerObject,
                 objectWidth,
                 numFeatures,
                 useTrace,
                 useRawTrace,
                 noiseFactor,
                 moduleNoiseFactor,
                 numModules,
                 thresholds,
                 inverseReadoutResolution,
                 enlargeModuleFactor,
                 bumpOverlapMethod):
  """
  Learn a set of objects. Then try to recognize each object. Output an
  interactive visualization.
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
        "scale": scale,
        "orientation": orientation,
        "activationThreshold": 8,
        "initialPermanence": 1.0,
        "connectedPermanence": 0.5,
        "learningThreshold": 8,
        "sampleSize": 10,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.0,
        "inverseReadoutResolution": inverseReadoutResolution,
        "enlargeModuleFactor": enlargeModuleFactor,
        "bumpOverlapMethod": bumpOverlapMethod,
    })

  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": thresholds,
    "reducedBasalThreshold": thresholds,
    "minThreshold": numModules,
    "sampleSize": numModules,
    "cellsPerColumn": 16,
  }

  column = PIUNCorticalColumn(locationConfigs, L4Overrides=l4Overrides,
                              useGaussian=True)
  exp = PIUNExperiment(column, featureNames=features,
                       numActiveMinicolumns=10,
                       noiseFactor=noiseFactor,
                       moduleNoiseFactor=moduleNoiseFactor)

  for objectDescription in objects:
    exp.learnObject(objectDescription)

  filename = os.path.join(
      SCRIPT_DIR,
      "traces/{}-resolution-{}-modules-{}-objects-{}-feats.html".format(
          inverseReadoutResolution, numModules, numObjects, numFeatures)
  )
  rawFilename = os.path.join(
      SCRIPT_DIR,
      "traces/{}-resolution-{}-modules-{}-objects-{}-feats.trace".format(
          inverseReadoutResolution, numModules, numObjects, numFeatures)
  )

  assert not (useTrace and useRawTrace), "Cannot use both --trace and --rawTrace"

  convergence = collections.defaultdict(int)
  if useTrace:
    with io.open(filename, "w", encoding="utf8") as fileOut:
      with trace(fileOut, exp, includeSynapses=True):
        print "Logging to", filename
        for objectDescription in objects:
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
          steps = exp.inferObjectWithRandomMovements(objectDescription)
          convergence[steps] += 1
          if steps is None:
            print 'Failed to infer object "{}"'.format(objectDescription["name"])
      fileOut.write(unicode(strOut.getvalue()))
  else:
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
  parser.add_argument("--numObjects", type=int, nargs="+", required=True)
  parser.add_argument("--numUniqueFeatures", type=int, required=True)
  parser.add_argument("--noiseFactor", type=float, nargs="+", required=False, default = 0)
  parser.add_argument("--moduleNoiseFactor", type=float, nargs="+", required=False, default=0)
  parser.add_argument("--inverseReadoutResolution", type=float, default=3)
  parser.add_argument("--enlargeModuleFactor", type=float, nargs="+", required=False, default=1.0)
  parser.add_argument("--useTrace", action="store_true")
  parser.add_argument("--useRawTrace", action="store_true")
  parser.add_argument("--numModules", type=int, nargs="+", default=[20])
  parser.add_argument(
    "--thresholds", type=int, default=None,
    help=(
      "The TM prediction threshold. Defaults to int((numModules+1)*0.8)."
      "Set to 0 for the threshold to match the number of modules."))
  parser.add_argument("--bumpOverlapMethod", type = str, default="probabilistic")
  parser.add_argument("--resultName", type = str, default="results.json")
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--appendResults", action="store_true")
  parser.add_argument("--numWorkers", type=int, default=cpu_count())

  args = parser.parse_args()


  # Use a fixed seed unless we're appending to a file. (In that case we're
  # probably running and rerunning the script to get smoother data, and we want
  # to get different results.)
  if not args.appendResults:
    np.random.seed(865486387)
    random.seed(357627)


  runMultiprocessNoiseExperiment(
    args.resultName, args.repeat, args.numWorkers, args.appendResults,
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
    inverseReadoutResolution=args.inverseReadoutResolution,
    enlargeModuleFactor=args.enlargeModuleFactor,
    bumpOverlapMethod=args.bumpOverlapMethod,
  )
