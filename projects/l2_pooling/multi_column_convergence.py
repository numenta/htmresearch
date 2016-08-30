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
This file checks the convergence of L4-L2 as you increase the number of columns,
or adjust the confusion across objects.
"""

import random

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


def runExperiment(noiseLevel=None,
                  profile=False,
                  numObjects=10,
                  numLocations=10,
                  numFeatures=10,
                  numColumns=3,
                  ):
  """
  Run experiment.

  :param noiseLevel: (float) Noise level to add to the locations and features
                             during inference
  :param profile:    (bool)  If True, the network will be profiled after
                             learning and inference.

  """
  name = "convergencecf_O%03d_L%03d_F%03d_C%03d" % (
    numObjects, numLocations, numFeatures, numColumns
  )
  exp = L4L2Experiment(
    name,
    numCorticalColumns=numColumns,
  )

  objects = exp.createRandomObjects(numObjects, 10, numLocations=numLocations,
                                    numFeatures=numFeatures)
  print "Objects are:"
  for object, pairs in objects.iteritems():
    print str(object) + ": " + str(pairs)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile(reset=True)

  # For inference, we will check and plot convergence for object 0. We create a
  # sequence of random sensations for each column.  We will present each
  # sensation for 4 time steps to let it settle and ensure it converges.
  objectSensations = {}
  for c in range(numColumns):
    objectCopy = [pair for pair in objects[0]]
    random.shuffle(objectCopy)
    # stay multiple steps on each sensation
    sensations = []
    for pair in objectCopy:
      for _ in xrange(4):
        sensations.append(pair)
    objectSensations[c] = sensations

  inferConfig = {
    "object": 0,
    "numSteps": len(objectSensations[0]),
    "pairs": objectSensations
  }

  exp.infer(inferConfig, noise=noiseLevel)
  if profile:
    exp.printProfile(reset=True)

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    onePlot=False,
  )



if __name__ == "__main__":
  runExperiment(numColumns=5, profile=True)
