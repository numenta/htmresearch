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

"""
This file creates simple experiment to test a single column L4-L2 network.
"""

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment



def createThreeObjects():
  """
  Helper function that creates a set of three objects used for basic
  experiments.

  :return:   (list(list(tuple))  List of lists of feature / location pairs.
  """
  objectA = zip(range(10), range(10))
  objectB = [(0, 0), (2, 2), (1, 1), (1, 4), (4, 2), (4, 1)]
  objectC = [(0, 0), (1, 1), (3, 1), (0, 1)]
  return [objectA, objectB, objectC]



def runSharedFeatures(noiseLevel=None, profile=False):
  """
  Runs a simple experiment where three objects share a number of location,
  feature pairs.

  :param noiseLevel: (float) Noise level to add to the locations and features
                             during inference
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.

  """
  exp = L4L2Experiment(
    "shared_features",
  )

  pairs = createThreeObjects()
  objects = {}
  for object in pairs:
    objects = exp.addObject(object, objects=objects)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile()

  inferConfig = {
    "object": 0,
    "numSteps": 10,
    "pairs": {
      0: zip(range(10), range(10))
    }
  }

  exp.infer(inferConfig, noise=noiseLevel)
  if profile:
    exp.printProfile()

  print exp.statistics

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
  )



def runUncertainLocations(missingLoc=None, profile=False):
  """
  Runs the same experiment as above, with missing locations at some timesteps
  during inference (if it was not successfully computed by the rest of the
  network for example).

  :param missingLoc: (dict) A dictionary mapping indices in the object to
                            location index to replace with during inference
                            (-1 means no location, a tuple means an union of
                            locations).
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.
  """
  if missingLoc is None:
    missingLoc = {}

  exp = L4L2Experiment(
    "uncertain_location",
  )

  pairs = createThreeObjects()
  objects = {}
  for object in pairs:
    objects = exp.addObject(object, objects=objects)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile()

  # create pairs with missing locations
  objectA = objects[0]
  for key, val in missingLoc.iteritems():
    objectA[key] = (val, key)

  inferConfig = {
    "object": 0,
    "numSteps": 10,
    "pairs": {
      0: objectA
    }
  }

  exp.infer(inferConfig)
  if profile:
    exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation",
            "L4 Predictive"],
  )



def runStretchExperiment(numObjects=25):
  """
  Generates a lot of random objects to profile the network.
  """
  exp = L4L2Experiment(
    "profiling_experiment",
  )

  objects = exp.createRandomObjects(numObjects=numObjects, numPoints=10)
  exp.learnObjects(objects)
  exp.printProfile()

  inferConfig = {
    "object": 0,
    "numSteps": len(objects[0]),
    "pairs": {
      0: objects[0]
    }
  }

  exp.infer(inferConfig)
  exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"]
  )


if __name__ == "__main__":
  # basic experiment with shared features
  runSharedFeatures()

  # experiment with unions at locations
  missingLoc = {3: (1,2,3), 6: (6,4,2)}
  runUncertainLocations(missingLoc=missingLoc)

  # stretch experiment to profile the regions
  runStretchExperiment()
