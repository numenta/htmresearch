# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017-2018, Numenta, Inc.  Unless you have an agreement
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
A shared experiment class for recognizing 2D objects by using path integration
of unions of locations that are specific to objects.
"""

import abc
import random

import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakPairMemory)
from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.algorithms.location_modules import Superficial2DLocationModule


class PIUNCorticalColumn(object):
  """
  A L4 + L6a network. Sensory input causes minicolumns in L4 to activate,
  which drives activity in L6a. Motor input causes L6a to perform path
  integration, updating its activity, which then depolarizes cells in L4.

  Whenever the sensor moves, call movementCompute. Whenever a sensory input
  arrives, call sensoryCompute.
  """

  def __init__(self, locationConfigs, L4Overrides=None):
    """
    @param L4Overrides (dict)
    Custom parameters for L4

    @param locationConfigs (sequence of dicts)
    Parameters for the location modules
    """
    L4Params = {
      "columnCount": 150,
      "cellsPerColumn": 16,
      "basalInputSize": (len(locationConfigs) *
                         sum(np.prod(config["cellDimensions"])
                             for config in locationConfigs))
    }
    if L4Overrides is not None:
      L4Params.update(L4Overrides)

    self.L4 = ApicalTiebreakPairMemory(**L4Params)

    self.L6aModules = [
      Superficial2DLocationModule(
        anchorInputSize=self.L4.numberOfCells(),
        **config)
      for config in locationConfigs]


  def movementCompute(self, displacement, noiseFactor = 0, moduleNoiseFactor = 0):
    """
    @param displacement (dict)
    The change in location. Example: {"top": 10, "left", 10}

    @return (dict)
    Data for logging/tracing.
    """

    if noiseFactor != 0:
      xdisp = np.random.normal(0, noiseFactor)
      ydisp = np.random.normal(0, noiseFactor)
    else:
      xdisp = 0
      ydisp = 0

    locationParams = {
      "displacement": [displacement["top"] + ydisp,
                       displacement["left"] + xdisp],
      "noiseFactor": moduleNoiseFactor
    }

    for module in self.L6aModules:
      module.movementCompute(**locationParams)

    return locationParams


  def sensoryCompute(self, activeMinicolumns, learn):
    """
    @param activeMinicolumns (numpy array)
    List of indices of minicolumns to activate.

    @param learn (bool)
    If True, the two layers should learn this association.

    @return (tuple of dicts)
    Data for logging/tracing.
    """
    inputParams = {
      "activeColumns": activeMinicolumns,
      "basalInput": self.getLocationRepresentation(),
      "learn": learn
    }
    self.L4.compute(**inputParams)

    locationParams = {
      "anchorInput": self.L4.getActiveCells(),
      "anchorGrowthCandidates": self.L4.getWinnerCells(),
      "learn": learn,
    }
    for module in self.L6aModules:
      module.sensoryCompute(**locationParams)

    return (inputParams, locationParams)


  def reset(self):
    """
    Clear all cell activity.
    """
    self.L4.reset()
    for module in self.L6aModules:
      module.reset()


  def activateRandomLocation(self):
    """
    Activate a random location in the location layer.
    """
    for module in self.L6aModules:
      module.activateRandomLocation()


  def getSensoryRepresentation(self):
    """
    Gets the active cells in the sensory layer.
    """
    return self.L4.getActiveCells()


  def getLocationRepresentation(self):
    """
    Get the full population representation of the location layer.
    """
    activeCells = np.array([], dtype="uint32")

    totalPrevCells = 0
    for module in self.L6aModules:
      activeCells = np.append(activeCells,
                              module.getActiveCells() + totalPrevCells)
      totalPrevCells += module.numberOfCells()

    return activeCells



class PIUNExperiment(object):
  """
  An experiment class which passes sensory and motor inputs into a special two
  layer network, tracks the location of a sensor on an object, and provides
  hooks for tracing.

  The network learns 2D "objects" which consist of arrangements of
  "features". Whenever this experiment moves the sensor to a feature, it always
  places it at the center of the feature.

  The network's location layer represents "the location of the sensor in the
  space of the object". Because it's a touch sensor, and because it always
  senses the center of each feature, this is synonymous with "the location of
  the feature in the space of the object" (but in other situations these
  wouldn't be equivalent).
  """

  def __init__(self, column,
               featureNames,
               numActiveMinicolumns=15,
               noiseFactor = 0,
               moduleNoiseFactor = 0):
    """
    @param column (PIUNColumn)
    A two-layer network.

    @param featureNames (list)
    A list of the features that will ever occur in an object.
    """
    self.column = column
    self.numActiveMinicolumns = numActiveMinicolumns

    # Use these for classifying SDRs and for testing whether they're correct.
    self.locationRepresentations = {
      # Example:
      # (objectName, featureIndex): [0, 26, 54, 77, 101, ...]
    }
    self.inputRepresentations = {
      # Example:
      # (objectName, featureIndex, featureName): [0, 26, 54, 77, 101, ...]
    }

    # Generate a set of feature SDRs.
    self.features = dict(
      (k, np.array(sorted(random.sample(xrange(self.column.L4.numberOfColumns()),
                                        self.numActiveMinicolumns)), dtype="uint32"))
      for k in featureNames)

    # For example:
    # [{"name": "Object 1",
    #   "features": [
    #       {"top": 40, "left": 40, "width": 10, "height" 10, "name": "A"},
    #       {"top": 80, "left": 80, "width": 10, "height" 10, "name": "B"}]]
    self.learnedObjects = []

    # The location of the sensor. For example: {"top": 20, "left": 20}
    self.locationOnObject = None

    self.maxSettlingTime = 10
    self.maxTraversals = 4

    self.monitors = {}
    self.nextMonitorToken = 1

    self.noiseFactor = noiseFactor
    self.moduleNoiseFactor = moduleNoiseFactor

    self.representationSet = set()

  def reset(self):
    self.column.reset()
    self.locationOnObject = None

    for monitor in self.monitors.values():
      monitor.afterReset()


  def learnObject(self,
                  objectDescription,
                  randomLocation=False,
                  useNoise=False,
                  noisyTrainingTime=1):
    """
    Train the network to recognize the specified object. Move the sensor to one of
    its features and activate a random location representation in the location
    layer. Move the sensor over the object, updating the location representation
    through path integration. At each point on the object, form reciprocal
    connections between the represention of the location and the representation
    of the sensory input.

    @param objectDescription (dict)
    For example:
    {"name": "Object 1",
     "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                  {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}
    """
    self.reset()
    self.column.activateRandomLocation()

    if randomLocation or useNoise:
      numIters = noisyTrainingTime
    else:
      numIters = 1
    for i in xrange(numIters):
      for iFeature, feature in enumerate(objectDescription["features"]):
        self._move(feature, randomLocation=randomLocation, useNoise=useNoise)
        featureSDR = self.features[feature["name"]]
        for _ in xrange(1):
          self._sense(featureSDR, learn=True, waitForSettle=False)

        if (objectDescription["name"], iFeature) not in self.locationRepresentations:
          self.locationRepresentations[(objectDescription["name"], iFeature)] = []

        self.locationRepresentations[(objectDescription["name"],
                                      iFeature)].append(
                                        self.column.getLocationRepresentation())
        self.inputRepresentations[(objectDescription["name"],
                                   iFeature, feature["name"])] = (
                                     self.column.getSensoryRepresentation())

    self.learnedObjects.append(objectDescription)

    self.representationSet.add(tuple(self.column.getLocationRepresentation()))


  def inferObjectWithRandomMovements(self,
                                     objectDescription,
                                     randomLocation=False,
                                     checkFalseConvergence=True):
    """
    Attempt to recognize the specified object with the network. Randomly move
    the sensor over the object until the object is recognized.

    @param objectDescription (dict)
    For example:
    {"name": "Object 1",
     "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                  {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}

    @return (bool)
    True if inference succeeded
    """
    self.reset()

    for monitor in self.monitors.values():
      monitor.beforeInferObject(objectDescription)

    currentStep = 0
    inferred = False
    prevTouchSequence = None

    for _ in xrange(self.maxTraversals):

      # Choose touch sequence.
      while True:
        touchSequence = range(len(objectDescription["features"]))
        random.shuffle(touchSequence)

        # Make sure the first touch will cause a movement.
        if (prevTouchSequence is not None and
            touchSequence[0] == prevTouchSequence[-1]):
          continue

        break

      for iFeature in touchSequence:
        currentStep += 1
        feature = objectDescription["features"][iFeature]
        self._move(feature, randomLocation=randomLocation)

        featureSDR = self.features[feature["name"]]
        self._sense(featureSDR, learn=False, waitForSettle=False)

        representation = self.column.getLocationRepresentation()

        target_representations = set(np.concatenate(
          self.locationRepresentations[
            (objectDescription["name"], iFeature)]))
        inferred = (set(representation) <=
          target_representations)

        if not inferred and tuple(representation) in self.representationSet:
          # We have converged to an incorrect representation - declare failure.
          print("Converged to an incorrect representation!")
          return None

        if inferred:
          break

      prevTouchSequence = touchSequence

      if inferred:
        break

    return currentStep if inferred else None


  def _move(self, feature, randomLocation = False, useNoise = True):
    """
    Move the sensor to the center of the specified feature. If the sensor is
    currently at another location, send the displacement into the cortical
    column so that it can perform path integration.
    """

    if randomLocation:
      locationOnObject = {
        "top": feature["top"] + np.random.rand()*feature["height"],
        "left": feature["left"] + np.random.rand()*feature["width"],
      }
    else:
      locationOnObject = {
        "top": feature["top"] + feature["height"]/2.,
        "left": feature["left"] + feature["width"]/2.
      }

    if self.locationOnObject is not None:
      displacement = {"top": locationOnObject["top"] -
                             self.locationOnObject["top"],
                      "left": locationOnObject["left"] -
                              self.locationOnObject["left"]}
      if useNoise:
        params = self.column.movementCompute(displacement,
                                             self.noiseFactor,
                                             self.moduleNoiseFactor)
      else:
        params = self.column.movementCompute(displacement, 0, 0)

      for monitor in self.monitors.values():
        monitor.afterLocationShift(**params)
    else:
      for monitor in self.monitors.values():
        monitor.afterLocationInitialize()

    self.locationOnObject = locationOnObject
    for monitor in self.monitors.values():
      monitor.afterLocationChanged(locationOnObject)


  def _sense(self, featureSDR, learn, waitForSettle):
    """
    Send the sensory input into the network. Optionally, send it multiple times
    until the network settles.
    """

    for monitor in self.monitors.values():
      monitor.beforeSense(featureSDR)

    iteration = 0
    prevCellActivity = None
    while True:
      (inputParams,
       locationParams) = self.column.sensoryCompute(featureSDR, learn)

      if waitForSettle:
        cellActivity = (set(self.column.getSensoryRepresentation()),
                        set(self.column.getLocationRepresentation()))
        if cellActivity == prevCellActivity:
          # It settled. Don't even log this timestep.
          break

        prevCellActivity = cellActivity

      for monitor in self.monitors.values():
        if iteration > 0:
          monitor.beforeSensoryRepetition()
        monitor.afterInputCompute(**inputParams)
        monitor.afterLocationAnchor(**locationParams)

      iteration += 1

      if not waitForSettle or iteration >= self.maxSettlingTime:
        break


  def addMonitor(self, monitor):
    """
    Subscribe to PIUNExperimentMonitor events.

    @param monitor (PIUNExperimentMonitor)
    An object that implements a set of monitor methods

    @return (object)
    An opaque object that can be used to refer to this monitor.
    """
    token = self.nextMonitorToken
    self.nextMonitorToken += 1

    self.monitors[token] = monitor

    return token


  def removeMonitor(self, monitorToken):
    """
    Unsubscribe from PIUNExperimentMonitor events.

    @param monitorToken (object)
    The return value of addMonitor() from when this monitor was added
    """
    del self.monitors[monitorToken]



class PIUNExperimentMonitor(object):
  """
  Abstract base class for a PIUNExperiment monitor.
  """

  __metaclass__ = abc.ABCMeta

  def beforeSense(self, featureSDR): pass
  def beforeSensoryRepetition(self): pass
  def beforeInferObject(self, obj): pass
  def afterReset(self): pass
  def afterLocationChanged(self, locationOnObject): pass
  def afterLocationInitialize(self): pass
  def afterLocationShift(self, **kwargs): pass
  def afterLocationAnchor(self, **kwargs): pass
  def afterInputCompute(self, **kwargs): pass
