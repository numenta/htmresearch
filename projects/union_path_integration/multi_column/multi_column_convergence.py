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
This file computes number of observations needed to unambiguously recognize an
object with multi-column L2-L4-L6a networks as the number of columns increases.
"""
import collections
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from nupic.engine import Network

from htmresearch.frameworks.location.location_network_creation import createMultipleL246aLocationColumn
from htmresearch.frameworks.location.object_generation import generateObjects
from htmresearch.support.expsuite import PyExperimentSuite
from htmresearch.support.register_regions import registerAllResearchRegions



class MultiColumnExperiment(PyExperimentSuite):
  """
  Number of observations needed to unambiguously recognize an object with
  multi-column networks as the set of columns increases. We train each network
  with a set number of objects and plot the average number of sensations
  required to unambiguously recognize an object.
  """

  def reset(self, params, repetition):
    """
    Take the steps necessary to reset the experiment before each repetition:
      - Make sure random seed is different for each repetition
      - Create the L2-L4-L6a network
      - Generate objects used by the experiment
      - Learn all objects used by the experiment
    """
    print params["name"], ":", repetition

    self.debug = params.get("debug", False)

    L2Params = json.loads('{' + params["l2_params"] + '}')
    L4Params = json.loads('{' + params["l4_params"] + '}')
    L6aParams = json.loads('{' + params["l6a_params"] + '}')

    # Make sure random seed is different for each repetition
    seed = params.get("seed", 42)
    np.random.seed(seed + repetition)
    random.seed(seed + repetition)
    L2Params["seed"] = seed + repetition
    L4Params["seed"] = seed + repetition
    L6aParams["seed"] = seed + repetition

    # Configure L6a params
    numModules = params["num_modules"]
    L6aParams["scale"] = [params["scale"]] * numModules
    angle = params["angle"] / numModules
    orientation = range(angle / 2, angle * numModules, angle)
    L6aParams["orientation"] = np.radians(orientation).tolist()

    # Create multi-column L2-L4-L6a network
    self.numColumns = params["num_cortical_columns"]
    network = Network()
    network = createMultipleL246aLocationColumn(network=network,
                                                numberOfColumns=self.numColumns,
                                                L2Params=L2Params,
                                                L4Params=L4Params,
                                                L6aParams=L6aParams)
    network.initialize()

    self.network = network
    self.sensorInput = []
    self.motorInput = []
    self.L2Regions = []
    self.L4Regions = []
    self.L6aRegions = []
    for i in xrange(self.numColumns):
      col = str(i)
      self.sensorInput.append(network.regions["sensorInput_" + col].getSelf())
      self.motorInput.append(network.regions["motorInput_" + col].getSelf())
      self.L2Regions.append(network.regions["L2_" + col])
      self.L4Regions.append(network.regions["L4_" + col])
      self.L6aRegions.append(network.regions["L6a_" + col])

    # Use the number of iterations as the number of objects. This will allow us
    # to execute one iteration per object and use the "iteration" parameter as
    # the object index
    numObjects = params["iterations"]

    # Generate feature SDRs
    numFeatures = params["num_features"]
    numOfMinicolumns = L4Params["columnCount"]
    numOfActiveMinicolumns = params["num_active_minicolumns"]
    self.featureSDR = {
      str(f): sorted(np.random.choice(numOfMinicolumns, numOfActiveMinicolumns))
      for f in xrange(numFeatures)
    }

    # Generate objects used in the experiment
    self.objects = generateObjects(numObjects=numObjects,
                                   featuresPerObject=params["features_per_object"],
                                   objectWidth=params["object_width"],
                                   numFeatures=numFeatures,
                                   distribution=params["feature_distribution"])

    self.sdrSize = L2Params["sdrSize"]

    # Learn objects
    self.numLearningPoints = params["num_learning_points"]
    self.numOfSensations = params["num_sensations"]
    self.learnedObjects = {}
    self.learn()

  def iterate(self, params, repetition, iteration):
    """
    For each iteration try to infer the object represented by the 'iteration'
    parameter returning the number of touches required to unambiguously
    classify the object.
    :param params: Specific parameters for this iteration. See 'experiments.cfg'
                   for list of parameters
    :param repetition: Current repetition
    :param iteration: Use the iteration to select the object to infer
    :return: number of touches required to unambiguously classify the object
    """
    objectToInfer = self.objects[iteration]
    stats = collections.defaultdict(list)
    touches = self.infer(objectToInfer, stats)
    results = {'touches': touches}
    results.update(stats)

    return results

  def setLearning(self, learn):
    """
    Set all regions in every column into the given learning mode
    """
    for col in xrange(self.numColumns):
      self.L2Regions[col].getSelf().setParameter("learningMode", 0, learn)
      self.L4Regions[col].getSelf().setParameter("learn", 0, learn)
      self.L6aRegions[col].getSelf().setParameter("learningMode", 0, learn)

  def sendReset(self):
    """
    Sends a reset signal to all regions in the network.
    It should be called before changing objects.
    """
    for col in xrange(self.numColumns):
      self.sensorInput[col].addResetToQueue(0)
      self.motorInput[col].addDataToQueue(displacement=[0, 0], reset=True)

    self.network.run(1)

  def learn(self):
    """
    Learn all objects on every column. Each column will learn all the features
    of every object and store the the object's L2 representation to be later
    used in the inference stage
    """
    self.setLearning(True)

    for obj in self.objects:
      self.sendReset()

      previousLocation = [None] * self.numColumns
      displacement = [0., 0.]
      features = obj["features"]
      numOfFeatures = len(features)

      # Randomize touch sequences
      touchSequence = np.random.permutation(numOfFeatures)

      for sensation in xrange(numOfFeatures):
        for col in xrange(self.numColumns):
          # Shift the touch sequence for each column making
          colSequence = np.roll(touchSequence, col)
          feature = features[colSequence[sensation]]
          # Move the sensor to the center of the object
          locationOnObject = np.array([feature["top"] + feature["height"] / 2.,
                                       feature["left"] + feature["width"] / 2.])
          # Calculate displacement from previous location
          if previousLocation[col] is not None:
            displacement = locationOnObject - previousLocation[col]
          previousLocation[col] = locationOnObject

          # learn each pattern multiple times
          activeColumns = self.featureSDR[feature["name"]]
          for _ in xrange(self.numLearningPoints):
            # Sense feature at location
            self.motorInput[col].addDataToQueue(displacement)
            self.sensorInput[col].addDataToQueue(activeColumns, False, 0)
            # Only move to the location on the first sensation.
            displacement = [0, 0]

      self.network.run(numOfFeatures * self.numLearningPoints)

      # update L2 representations for the object
      self.learnedObjects[obj["name"]] = self.getL2Representations()

  def infer(self, objectToInfer, stats=None):
    """
    Attempt to recognize the specified object with the network. Randomly move
    the sensor over the object until the object is recognized.
    """
    self.setLearning(False)
    self.sendReset()

    previousLocation = [None] * self.numColumns
    displacement = [0., 0.]
    features = objectToInfer["features"]
    objName = objectToInfer["name"]
    numOfFeatures = len(features)

    # Randomize touch sequences
    touchSequence = np.random.permutation(numOfFeatures)


    for sensation in xrange(self.numOfSensations):
      # Add sensation for all columns at once
      for col in xrange(self.numColumns):
        # Shift the touch sequence for each column making
        colSequence = np.roll(touchSequence, col)
        feature = features[touchSequence[sensation]]
        # Move the sensor to the center of the object
        locationOnObject = np.array([feature["top"] + feature["height"] / 2.,
                                     feature["left"] + feature["width"] / 2.])
        # Calculate displacement from previous location
        if previousLocation[col] is not None:
          displacement = locationOnObject - previousLocation[col]
        previousLocation[col] = locationOnObject

        # Sense feature at location
        self.motorInput[col].addDataToQueue(displacement)
        self.sensorInput[col].addDataToQueue(self.featureSDR[feature["name"]],
                                             False, 0)
      self.network.run(1)
      if self.debug:
        self._updateInferenceStats(statistics=stats, objectName=objName)

      if self.isObjectClassified(objName, minOverlap=30):
        return sensation

    return self.numOfSensations

  def getL2Representations(self):
    """
    Returns the active representation in L2.
    """
    return [set(L2.getSelf()._pooler.getActiveCells()) for L2 in self.L2Regions]

  def getL4Representations(self):
    """
    Returns the active representation in L4.
    """
    return [set(L4.getOutputData("activeCells").nonzero()[0])
            for L4 in self.L4Regions]

  def getL4PredictedCells(self):
    """
    Returns the cells in L4 that were predicted by the location input.
    """
    return [set(L4.getOutputData("predictedCells").nonzero()[0])
            for L4 in self.L4Regions]

  def getL4PredictedActiveCells(self):
    """
    Returns the cells in L4 that were predicted by the location signal
    and are currently active.  Does not consider apical input.
    """
    return [set(L4.getOutputData("predictedActiveCells").nonzero()[0])
            for L4 in self.L4Regions]

  def isObjectClassified(self, objectName, minOverlap=None, maxL2Size=None):
    """
    Return True if objectName is currently unambiguously classified by every L2
    column. Classification is correct and unambiguous if the current L2 overlap
    with the true object is greater than minOverlap and if the size of the L2
    representation is no more than maxL2Size

    :param minOverlap: min overlap to consider the object as recognized.
                       Defaults to half of the SDR size

    :param maxL2Size: max size for the L2 representation
                       Defaults to 1.5 * SDR size

    :return: True/False
    """
    L2Representation = self.getL2Representations()
    objectRepresentation = self.learnedObjects[objectName]
    if minOverlap is None:
      minOverlap = self.sdrSize / 2
    if maxL2Size is None:
      maxL2Size = 1.5 * self.sdrSize

    numCorrectClassifications = 0
    for col in xrange(self.numColumns):
      overlapWithObject = len(objectRepresentation[col] & L2Representation[col])

      if (overlapWithObject >= minOverlap and
          len(L2Representation[col]) <= maxL2Size):
        numCorrectClassifications += 1

    return numCorrectClassifications == self.numColumns

  def _updateInferenceStats(self, statistics, objectName=None):
    """
    Updates the inference statistics.

    Parameters:
    ----------------------------
    @param  statistics (dict)
            Dictionary in which to write the statistics

    @param  objectName (str)
            Name of the inferred object, if known. Otherwise, set to None.

    """
    L4Representations = self.getL4Representations()
    L4PredictedCells = self.getL4PredictedCells()
    L2Representation = self.getL2Representations()

    for i in xrange(self.numColumns):
      statistics["L4 Representation C" + str(i)].append(
        len(L4Representations[i]))
      statistics["L4 Predicted C" + str(i)].append(len(L4PredictedCells[i]))
      statistics["L2 Representation C" + str(i)].append(
        len(L2Representation[i]))
      statistics["Full L2 SDR C" + str(i)].append(sorted(
        [int(c) for c in L2Representation[i]]))
      statistics["L4 Apical Segments C" + str(i)].append(len(
        self.L4Regions[i].getSelf()._tm.getActiveApicalSegments()))

      # add true overlap and classification result if objectName was learned
      if objectName in self.learnedObjects:
        objectRepresentation = self.learnedObjects[objectName]
        statistics["Overlap L2 with object C" + str(i)].append(
          len(objectRepresentation[i] & L2Representation[i]))

    if objectName in self.learnedObjects:
      if self.isObjectClassified(objectName):
        statistics["Correct classification"].append(1.0)
      else:
        statistics["Correct classification"].append(0.0)



def plotSensationByColumn(suite, name):
  """
  Plots the convergence graph: touches by columns.
  """
  path = suite.cfgparser.get(name, "path")
  path = os.path.join(path, name)

  touches = {}
  plt.figure(tight_layout={"pad": 0})
  for exp in suite.get_exps(path=path):
    params = suite.get_params(exp)
    cols = params["num_cortical_columns"]
    features = params["num_features"]
    if not features in touches:
      touches[features] = {}

    touches[features][cols] = np.mean(
      suite.get_histories_over_repetitions(exp, "touches", np.mean))

  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  for i, features in enumerate(sorted(touches)):
    cols = touches[features]
    plt.plot(cols.keys(), cols.values(), "-",
             label="Unique features={}".format(features),
             color=colorList[i])

  # format
  plt.xlabel("Number of columns")
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (multiple columns)")
  plt.legend(framealpha=1.0)

  # save
  path = suite.cfgparser.get(name, "path")
  plotPath = os.path.join(path, "{}.pdf".format(name))
  plt.savefig(plotPath)
  plt.close()



if __name__ == "__main__":
  registerAllResearchRegions()

  suite = MultiColumnExperiment()
  suite.start()

  experiments = suite.options.experiments
  if experiments is None:
    experiments = suite.cfgparser.sections()

  for exp in experiments:
    plotSensationByColumn(suite, exp)
