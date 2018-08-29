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
import math
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
from htmresearch.frameworks.location.two_layer_tracing import (
  PIUNVisualizer as trace)
from htmresearch.frameworks.location.object_generation import generateObjects

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def doExperiment(numObjects,
                 featuresPerObject,
                 objectWidth,
                 numFeatures,
                 featureDistribution,
                 useTrace,
                 noiseFactor,
                 moduleNoiseFactor,
                 numModules,
                 thresholds,
                 inverseReadoutResolution,
                 enlargeModuleFactor,
                 bumpOverlapMethod,
                 seed1,
                 seed2):
  """
  Learn a set of objects. Then try to recognize each object. Output an
  interactive visualization.
  """
  if not os.path.exists("traces"):
    os.makedirs("traces")

  if seed1 != -1:
    np.random.seed(seed1)

  if seed2 != -1:
    random.seed(seed2)

  features = [str(i) for i in xrange(numFeatures)]
  objects = generateObjects(numObjects, featuresPerObject, objectWidth,
                            numFeatures, featureDistribution)

  locationConfigs = []
  scale = 40.0

  if thresholds == -1:
    thresholds = int(math.ceil(numModules*0.8))
  elif thresholds == 0:
    thresholds = numModules
  perModRange = float(60.0 / float(numModules))
  for i in xrange(numModules):
    orientation = (float(i) * perModRange) + (perModRange / 2.0)

    locationConfigs.append({
        "scale": scale,
        "orientation": np.radians(orientation),
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

  convergence = collections.defaultdict(int)
  try:
    if useTrace:
      filename = os.path.join(
        SCRIPT_DIR,
        "traces/{}-resolution-{}-modules-{}-objects-{}-feats.html".format(
          inverseReadoutResolution, numModules, numObjects, numFeatures)
      )
      traceFileOut = io.open(filename, "w", encoding="utf8")
      traceHandle = trace(traceFileOut, exp, includeSynapses=True)
      print "Logging to", filename

    for objectDescription in objects:
      steps = exp.inferObjectWithRandomMovements(objectDescription)
      convergence[steps] += 1
      if steps is None:
        print 'Failed to infer object "{}"'.format(objectDescription["name"])
  finally:
    if useTrace:
      traceHandle.__exit__()
      traceFileOut.close()

  for step, num in sorted(convergence.iteritems()):
    print "{}: {}".format(step, num)

  result = {
    "convergence": convergence,
  }

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
  parser.add_argument("--numObjects", type=int, nargs="+", required=True)
  parser.add_argument("--numUniqueFeatures", type=int, required=True)
  parser.add_argument("--noiseFactor", type=float, nargs="+", required=False, default = 0)
  parser.add_argument("--moduleNoiseFactor", type=float, nargs="+", required=False, default=0)
  parser.add_argument("--inverseReadoutResolution", type=float, default=3)
  parser.add_argument("--enlargeModuleFactor", type=float, nargs="+", required=False, default=1.0)
  parser.add_argument("--useTrace", action="store_true")
  parser.add_argument("--numModules", type=int, nargs="+", default=[20])
  parser.add_argument("--seed1", type=int, default=-1)
  parser.add_argument("--seed2", type=int, default=-1)
  parser.add_argument(
    "--thresholds", type=int, default=-1,
    help=(
      "The TM prediction threshold. Defaults to ceil(numModules*0.8)."
      "Set to 0 for the threshold to match the number of modules."))
  parser.add_argument("--featuresPerObject", type=int, nargs="+", default=10)
  parser.add_argument("--featureDistribution", type = str, nargs="+",
                      default="AllFeaturesEqual_Replacement")
  parser.add_argument("--bumpOverlapMethod", type = str, default="probabilistic")
  parser.add_argument("--resultName", type = str, default="results.json")
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--appendResults", action="store_true")
  parser.add_argument("--numWorkers", type=int, default=cpu_count())

  args = parser.parse_args()


  runMultiprocessNoiseExperiment(
    args.resultName, args.repeat, args.numWorkers, args.appendResults,
    numObjects=args.numObjects,
    featuresPerObject=args.featuresPerObject,
    featureDistribution=args.featureDistribution,
    objectWidth=4,
    numFeatures=args.numUniqueFeatures,
    useTrace=args.useTrace,
    noiseFactor=args.noiseFactor,
    moduleNoiseFactor=args.moduleNoiseFactor,
    numModules=args.numModules,
    thresholds=args.thresholds,
    inverseReadoutResolution=args.inverseReadoutResolution,
    enlargeModuleFactor=args.enlargeModuleFactor,
    bumpOverlapMethod=args.bumpOverlapMethod,
    seed1=args.seed1,
    seed2=args.seed2,
  )
