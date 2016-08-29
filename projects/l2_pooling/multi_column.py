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

import random

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


def runLateralDisambiguation(noiseLevel=None, profile=False):
  """
  Runs a simple experiment where two objects share a (location, feature) pair.
  At inference, one column sees that ambiguous pair, and the other sees a
  unique one. We should see the first column rapidly converge to a
  unique representation.

  :param noiseLevel: (float) Noise level to add to the locations and features
                             during inference
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.

  """
  exp = L4L2Experiment(
    "lateral_disambiguation",
    numCorticalColumns=2,
  )

  objects = {}
  objects = exp.addObject([(1, 1), (2, 2)], objects=objects)
  objects = exp.addObject([(1, 1), (3, 2)], objects=objects)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile()

  inferConfig = {
    "object": 1,
    "numSteps": 6,
    "pairs": {
      # this should activate 0 and 1
      0: [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
      # this should activate 1
      1: [(3, 2), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2)]
    }
  }

  exp.infer(inferConfig, noise=noiseLevel)
  if profile:
    exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    onePlot=False,
  )



def runDisambiguationByUnions(noiseLevel=None, profile=False):
  """
  Runs a simple experiment where an object is disambiguated as each column
  recognizes a union of two objects, and the real object is the only
  common one.

  :param noiseLevel: (float) Noise level to add to the locations and features
                             during inference
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.

  """
  exp = L4L2Experiment(
    "disambiguation_unions",
    numCorticalColumns=2,
  )

  objects = {}
  objects = exp.addObject([(1, 1), (2, 2)], name=0, objects=objects)
  objects = exp.addObject([(2, 2), (3, 3)], name=1, objects=objects)
  objects = exp.addObject([(3, 3), (4, 4)], name=2, objects=objects)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile()

  inferConfig = {
    "object": 1,
    "numSteps": 6,
    "pairs": {
      # this should activate 1 and 2
      0: [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
      # this should activate 2 and 3
      1: [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    }
  }

  exp.infer(inferConfig, noise=noiseLevel)
  if profile:
    exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    onePlot=False,
  )



def runStretch(noiseLevel=None, profile=False):
  """
  Stretch test that learns a lot of objects.

  :param noiseLevel: (float) Noise level to add to the locations and features
                             during inference
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.

  """
  exp = L4L2Experiment(
    "stretch",
    numCorticalColumns=2,
  )

  objects = exp.createRandomObjects(10, 10)
  print "Objects are:"
  for object, pairs in objects.iteritems():
    print str(object) + ": " + str(pairs)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile()

  objectCopy1 = [pair for pair in objects[0]]
  objectCopy2 = [pair for pair in objects[0]]
  random.shuffle(objectCopy1)
  random.shuffle(objectCopy2)

  # stay multiple steps on each sensation
  object1 = []
  for pair in objectCopy1:
    for _ in xrange(4):
      object1.append(pair)

  # stay multiple steps on each sensation
  object2 = []
  for pair in objectCopy2:
    for _ in xrange(4):
      object2.append(pair)

  inferConfig = {
    "object": 0,
    "numSteps": len(object1),
    "pairs": {
      0: object1,
      1: object2,
    }
  }

  exp.infer(inferConfig, noise=noiseLevel)
  if profile:
    exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    onePlot=False,
  )


def runAmbiguities(noiseLevel=None, profile=False):
  """
  Runs a simple experiment where an object is disambiguated as each column
  recognizes a union of two objects, and the real object is the only
  common one.

  :param noiseLevel: (float) Noise level to add to the locations and features
                             during inference
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.

  """
  exp = L4L2Experiment(
    "ambiguities",
    numCorticalColumns=2,
  )

  objects = {}
  objects = exp.addObject([(1, 1), (2, 1), (3, 3)], name=0, objects=objects)
  objects = exp.addObject([(2, 2), (3, 3), (2, 1)], name=1, objects=objects)
  objects = exp.addObject([(3, 1), (2, 1), (1, 2)], name=2, objects=objects)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile()

  print exp.objectL2Representations

  inferConfig = {
    "object": 1,
    "numSteps": 6,
    "pairs": {
      0: [(2, 1), (2, 1), (3, 3), (2, 2), (2, 2), (2, 2)],
      1: [(3, 3), (3, 3), (3, 3), (2, 2), (2, 1), (2, 1)]
    }
  }

  exp.infer(inferConfig, noise=noiseLevel)
  if profile:
    exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    onePlot=False,
  )

  print exp.getInferenceStats(0)



if __name__ == "__main__":
  # runLateralDisambiguation()
  # runDisambiguationByUnions()
  # runStretch()
  runAmbiguities()
