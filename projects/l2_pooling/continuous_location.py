# ----------------------------------------------------------------------
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
This file creates simple experiment to test an L4-L2 network on physical
objects.
"""

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)
from htmresearch.frameworks.layers.physical_objects import *



def runBasic(noiseLevel=None, profile=False):
  """
  Runs a basic experiment on continuous locations, learning a few locations on
  four basic objects, and inferring one of them.

  This experiment is mostly used for testing the pipeline, as the learned
  locations are too random and sparse to actually perform inference.

  Parameters:
  ----------------------------
  @param    noiseLevel (float)
            Noise level to add to the locations and features during inference

  @param    profile (bool)
            If True, the network will be profiled after learning and inference

  """
  exp = L4L2Experiment(
    "basic_continuous",
    numCorticalColumns=2
  )

  objects = createObjectMachine(
    machineType="continuous",
    numInputBits=21,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=2,
  )

  objects.addObject(Sphere(radius=20), name="sphere")
  objects.addObject(Cylinder(height=50, radius=20), name="cylinder")
  objects.addObject(Box(dimensions=[10, 20, 30,]), name="box")
  objects.addObject(Cube(width=20), name="cube")

  learnConfig = {
    "sphere": [("surface", 10)],
    # the two learning config below will be exactly the same
    "box": [("face", 5), ("edge", 5), ("vertex", 5)],
    "cube": [(feature, 5) for feature in objects["cube"].getLocations()],
    "cylinder": [(feature, 5) for feature in objects["cylinder"].getLocations()]
  }

  exp.learnObjects(
    objects.provideObjectsToLearn(learnConfig, plot=True),
    reset=True
  )
  if profile:
    exp.printProfile()

  inferConfig = {
    "numSteps": 4,
    "noiseLevel": noiseLevel,
    "objectName": "cube",
    "pairs": {
      0: ["face", "face", "edge", "edge"],
      1: ["edge", "face", "face", "edge"]
    }
  }

  exp.infer(
    objects.provideObjectToInfer(inferConfig, plot=True),
    objectName="cube",
    reset=True
  )
  if profile:
    exp.printProfile()

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
  )



if __name__ == "__main__":
  runBasic()
