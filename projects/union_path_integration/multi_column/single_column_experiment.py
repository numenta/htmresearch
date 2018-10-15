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
This file computes the activation density after each sensation, averaged across
all objects and modules using a single column L4-L6a network. This code will
generate figure 6A in the paper "Locations in the Neocortex: A Theory of
Sensorimotor Object Recognition Using Cortical Grid Cells"
Marcus Lewis, Scott Purdy, Subutai Ahmad, Jeff Hawkins
doi: https://doi.org/10.1101/436352
"""
import json
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nupic.engine import Network

from htmresearch.frameworks.layers.sensor_placement import greedySensorPositions
from htmresearch.frameworks.location.location_network_creation import createL4L6aLocationColumn
from htmresearch.frameworks.location.object_generation import generateObjects
from htmresearch.support.expsuite import PyExperimentSuite
from htmresearch.support.register_regions import registerAllResearchRegions



class SingleColumnExperiment(PyExperimentSuite):
  """
  Show the activation density after each sensation, averaged across all objects
  and modules.
  """

  def reset(self, params, repetition):
    """
    Take the steps necessary to reset the experiment before each repetition:
      - Make sure random seed is different for each repetition
      - Create the L4-L6a network
      - Generate objects used by the experiment
      - Learn all objects used by the experiment
    """
    print params["name"], ":", repetition

    L4Params = json.loads('{' + params["l4_params"] + '}')
    L6aParams = json.loads('{' + params["l6a_params"] + '}')

    # Make sure random seed is different for each repetition
    seed = params.get("seed", 42)
    np.random.seed(seed + repetition)
    random.seed(seed + repetition)
    L4Params["seed"] = seed + repetition
    L6aParams["seed"] = seed + repetition

    # Configure L6a params
    numModules = params["num_modules"]
    L6aParams["scale"] = [params["scale"]] * numModules
    angle = params["angle"] / numModules
    orientation = range(angle / 2, angle * numModules, angle)
    L6aParams["orientation"] = np.radians(orientation).tolist()

    # Create L4-L6a network
    network = Network()
    network = createL4L6aLocationColumn(network=network,
                                        L4Params=L4Params,
                                        L6aParams=L6aParams)
    network.initialize()

    self.network = network
    self.sensorInput = network.regions["sensorInput"]
    self.motorInput = network.regions["motorInput"]
    self.L4 = network.regions["L4"]
    self.L6a = network.regions["L6a"]

    # Generate feature SDRs
    numFeatures = params["num_features"]
    numOfMinicolumns = L4Params["columnCount"]
    numOfActiveMinicolumns = params["num_active_minicolumns"]
    self.featureSDR = {
      str(f): sorted(np.random.choice(numOfMinicolumns, numOfActiveMinicolumns))
      for f in xrange(numFeatures)
    }

    # Use the number of iterations as the number of objects. This will allow us
    # to execute one iteration per object and use the "iteration" parameter as
    # the object index
    numObjects = params["iterations"]

    # Generate objects used in the experiment
    self.objects = generateObjects(numObjects=numObjects,
                                   featuresPerObject=params["features_per_object"],
                                   objectWidth=params["object_width"],
                                   numFeatures=numFeatures,
                                   distribution=params["feature_distribution"])

    # Learn objects
    self.numLearningPoints = params["num_learning_points"]
    self.numOfSensations = params["num_sensations"]
    self.learn()

  def iterate(self, params, repetition, iteration):
    """
    Record the number of active cells for each sensation
    """
    objectToInfer = self.objects[iteration]
    activeCells = self.infer(objectToInfer)

    return {'activeCells': activeCells}

  def setLearning(self, learn):
    """
    Set all regions to the given learning mode
    """
    self.L4.setParameter("learn", learn)
    self.L6a.setParameter("learningMode", learn)

  def sendReset(self):
    """
    Sends a reset signal to all regions in the network.
    """
    self.sensorInput.getSelf().addResetToQueue(0)
    self.motorInput.getSelf().addDataToQueue(displacement=[0, 0], reset=True)
    self.network.run(1)

  def learn(self):
    """
    Learn all objects
    """
    self.setLearning(True)

    for obj in self.objects:
      self.sendReset()

      previousLocation = None
      displacement = [0., 0.]
      features = obj["features"]
      numOfFeatures = len(features)

      for sensation in xrange(numOfFeatures):
        feature = features[sensation]
        # Move the sensor to the center of the object
        locationOnObject = np.array([feature["top"] + feature["height"] / 2.,
                                     feature["left"] + feature["width"] / 2.])
        # Calculate displacement from previous location
        if previousLocation is not None:
          displacement = locationOnObject - previousLocation
        previousLocation = locationOnObject

        # learn each pattern multiple times
        activeColumns = self.featureSDR[feature["name"]]
        for _ in xrange(self.numLearningPoints):
          # Sense feature at location
          self.motorInput.getSelf().addDataToQueue(displacement)
          self.sensorInput.getSelf().addDataToQueue(activeColumns, False, 0)
          # Only move to the location on the first sensation.
          displacement = [0, 0]

      self.network.run(numOfFeatures * self.numLearningPoints)

  def infer(self, objectToInfer):
    """
    Move the sensor to random locations on the object and record the number of
    active cells for each sensation
    :param objectToInfer: A previously learned object
    :type objectToInfer: dict
    :return:  number of active cells per sensation
    :rtype list:
    """
    self.setLearning(False)
    self.sendReset()

    previousLocation = None
    displacement = [0., 0.]
    features = objectToInfer["features"]
    numOfFeatures = len(features)

    # Randomize touch sequences
    sensorIterator = greedySensorPositions(1, numOfFeatures)
    touchSequence = [next(sensorIterator) for _ in xrange(self.numOfSensations)]

    results = []
    for sensation in xrange(self.numOfSensations):
      feature = features[touchSequence[sensation][0]]
      # Move the sensor to the center of the object
      locationOnObject = np.array([feature["top"] + feature["height"] / 2.,
                                   feature["left"] + feature["width"] / 2.])
      # Calculate displacement from previous location
      if previousLocation is not None:
        displacement = locationOnObject - previousLocation
      previousLocation = locationOnObject

      # Sense feature at location
      self.motorInput.getSelf().addDataToQueue(displacement)
      self.sensorInput.getSelf().addDataToQueue(self.featureSDR[feature["name"]],
                                                False, 0)
      self.network.run(1)

      # Save activation by sensation
      results.append(self.getL6aNumOfActiveCell())

    return results

  def getL6aNumOfActiveCell(self):
    """
    Return the current number of active cells on L6a
    """
    return len(self.L6a.getOutputData("activeCells").nonzero()[0])



def plotDensitybySensation(suite, name):

  path = suite.cfgparser.get(name, "path")
  path = os.path.join(path, name)

  plt.figure(figsize=(3.25, 2.5), tight_layout={"pad": 0})
  for exp in sorted(suite.get_exps(path=path)):
    params = suite.get_params(exp)
    numModules = params["num_modules"]
    cellsPerAxis = params["cells_per_axis"]
    numCells = cellsPerAxis * cellsPerAxis * numModules

    # Number of objects represented by number of iterations
    numObjects = params["iterations"]
    activeCells = suite.get_history(exp, 0, "activeCells")
    numSteps = params["num_sensations"]
    x = np.arange(1, numSteps + 1)
    y = np.array(activeCells).mean(axis=0) / numCells

    plt.plot(x, y, "-", label="{} learned objects".format(numObjects))

  plt.xlabel("Number of Sensations")
  plt.ylabel("Mean Cell Activation Density")

  plt.ylim((-0.05, 1.05))

  # If there's any opacity, when we export a copy of this from Illustrator, it
  # creates a PDF that isn't compatible with Word.
  framealpha = 1.0
  plt.legend(framealpha=framealpha)

  path = suite.cfgparser.get(name, "path")
  plotPath = os.path.join(path, "{}.pdf".format(name))
  plt.savefig(plotPath)
  plt.close()



if __name__ == "__main__":
  registerAllResearchRegions()

  suite = SingleColumnExperiment()
  suite.start()

  experiments = suite.options.experiments
  if experiments is None:
    experiments = suite.cfgparser.sections()

  for exp in experiments:
    plotDensitybySensation(suite, exp)

