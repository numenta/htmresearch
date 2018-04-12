# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
A shared experiment class for inferring 2D objects with location modules.
"""

import abc
import random

import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakPairMemory)
from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.algorithms.location_modules import Superficial2DLocationModule


class Grid2DLocationExperiment(object):
  """
  The experiment code organized into a class.
  """

  def __init__(self, objects, objectPlacements, featureNames, locationConfigs,
               worldDimensions):

    self.objects = objects
    self.objectPlacements = objectPlacements
    self.worldDimensions = worldDimensions

    self.features = dict(
      (k, np.array(sorted(random.sample(xrange(150), 15)), dtype="uint32"))
      for k in featureNames)

    self.locationModules = [Superficial2DLocationModule(anchorInputSize=150*32,
                                                        **config)
                            for config in locationConfigs]

    self.inputLayer = ApicalTiebreakPairMemory(**{
      "columnCount": 150,
      "cellsPerColumn": 32,
      "basalInputSize": 18 * sum(np.prod(config["cellDimensions"])
                                 for config in locationConfigs),
      "apicalInputSize": 4096
    })

    self.objectLayer = ColumnPooler(**{
      "inputWidth": 150 * 32
    })

    # Use these for classifying SDRs and for testing whether they're correct.
    self.locationRepresentations = {
      # Example:
      # (objectName, (top, left)): [0, 26, 54, 77, 101, ...]
    }
    self.inputRepresentations = {
      # Example:
      # (objectName, (top, left), featureName): [0, 26, 54, 77, 101, ...]
    }
    self.objectRepresentations = {
      # Example:
      # objectName: [14, 19, 54, 107, 201, ...]
    }

    self.locationInWorld = None

    self.maxSettlingTime = 10

    self.monitors = {}
    self.nextMonitorToken = 1


  def addMonitor(self, monitor):
    """
    Subscribe to Grid2DLocationExperimentMonitor events.

    @param monitor (Grid2DLocationExperimentMonitor)
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
    Unsubscribe from LocationExperiment events.

    @param monitorToken (object)
    The return value of addMonitor() from when this monitor was added
    """
    del self.monitors[monitorToken]


  def getActiveLocationCells(self):
    activeCells = np.array([], dtype="uint32")

    totalPrevCells = 0
    for i, module in enumerate(self.locationModules):
      activeCells = np.append(activeCells,
                              module.getActiveCells() + totalPrevCells)
      totalPrevCells += module.numberOfCells()

    return activeCells


  def move(self, objectName, locationOnObject):
    objectPlacement = self.objectPlacements[objectName]
    locationInWorld = (objectPlacement[0] + locationOnObject[0],
                       objectPlacement[1] + locationOnObject[1])

    if self.locationInWorld is not None:
      deltaLocation = (locationInWorld[0] - self.locationInWorld[0],
                       locationInWorld[1] - self.locationInWorld[1])

      for monitor in self.monitors.values():
        monitor.beforeMove(deltaLocation)

      params = {
        "displacement": deltaLocation
      }
      for module in self.locationModules:
        module.movementCompute(**params)

      for monitor in self.monitors.values():
        monitor.afterLocationShift(**params)

    self.locationInWorld = locationInWorld
    for monitor in self.monitors.values():
      monitor.afterWorldLocationChanged(locationInWorld)


  def _senseInferenceMode(self, featureSDR):
    prevCellActivity = None
    for i in xrange(self.maxSettlingTime):
      inputParams = {
        "activeColumns": featureSDR,
        "basalInput": self.getActiveLocationCells(),
        "apicalInput": self.objectLayer.getActiveCells(),
        "learn": False
      }
      self.inputLayer.compute(**inputParams)

      objectParams = {
        "feedforwardInput": self.inputLayer.getActiveCells(),
        "feedforwardGrowthCandidates": self.inputLayer.getPredictedActiveCells(),
        "learn": False,
      }
      self.objectLayer.compute(**objectParams)

      locationParams = {
        "anchorInput": self.inputLayer.getActiveCells(),
        "anchorGrowthCandidates": self.inputLayer.getWinnerCells(),
        "learn": False
      }
      for module in self.locationModules:
        module.sensoryCompute(**locationParams)

      cellActivity = (set(self.objectLayer.getActiveCells()),
                      set(self.inputLayer.getActiveCells()),
                      set(self.getActiveLocationCells()))

      if cellActivity == prevCellActivity:
        # It settled. Don't even log this timestep.
        break
      else:
        prevCellActivity = cellActivity
        for monitor in self.monitors.values():
          if i > 0:
            monitor.markSensoryRepetition()

          monitor.afterInputCompute(**inputParams)
          monitor.afterObjectCompute(**objectParams)
          monitor.afterLocationAnchor(**locationParams)


  def _senseLearningMode(self, featureSDR):
    inputParams = {
      "activeColumns": featureSDR,
      "basalInput": self.getActiveLocationCells(),
      "apicalInput": self.objectLayer.getActiveCells(),
      "learn": True
    }
    self.inputLayer.compute(**inputParams)

    objectParams = {
      "feedforwardInput": self.inputLayer.getActiveCells(),
      "feedforwardGrowthCandidates": self.inputLayer.getPredictedActiveCells(),
      "learn": True,
    }
    self.objectLayer.compute(**objectParams)

    locationParams = {
      "anchorInput": self.inputLayer.getActiveCells(),
      "anchorGrowthCandidates": self.inputLayer.getWinnerCells(),
      "learn": True,
    }
    for module in self.locationModules:
      module.sensoryCompute(**locationParams)

    for monitor in self.monitors.values():
      monitor.afterInputCompute(**inputParams)
      monitor.afterObjectCompute(**objectParams)


  def sense(self, featureSDR, learn):
    for monitor in self.monitors.values():
      monitor.beforeSense(featureSDR)

    if learn:
      self._senseLearningMode(featureSDR)
    else:
      self._senseInferenceMode(featureSDR)


  def learnObjects(self):
    """
    Learn each provided object.

    This method simultaneously learns 4 sets of synapses:
    - location -> input
    - input -> location
    - input -> object
    - object -> input
    """
    for objectName, objectFeatures in self.objects.iteritems():
      self.reset()

      for module in self.locationModules:
        module.activateRandomLocation()

      for feature in objectFeatures:
        locationOnObject = (feature["top"] + feature["height"]/2,
                            feature["left"] + feature["width"]/2)
        self.move(objectName, locationOnObject)

        featureName = feature["name"]
        featureSDR = self.features[featureName]
        for _ in xrange(10):
          self.sense(featureSDR, learn=True)

        self.locationRepresentations[(objectName, locationOnObject)] = (
          self.getActiveLocationCells())
        self.inputRepresentations[(objectName, locationOnObject, featureName)] = (
          self.inputLayer.getActiveCells())

      self.objectRepresentations[objectName] = self.objectLayer.getActiveCells()


  def inferObjectsWithRandomMovements(self):
    """
    Infer each object without any location input.
    """
    for objectName, objectFeatures in self.objects.iteritems():
      self.reset()

      inferred = False
      prevTouchSequence = None

      for _ in xrange(4):

        while True:
          touchSequence = list(objectFeatures)
          random.shuffle(touchSequence)

          if prevTouchSequence is not None:
            if touchSequence[0] == prevTouchSequence[-1]:
              continue

          break

        for i, feature in enumerate(touchSequence):
          locationOnObject = (feature["top"] + feature["height"]/2,
                              feature["left"] + feature["width"]/2)
          self.move(objectName, locationOnObject)

          featureName = feature["name"]
          featureSDR = self.features[featureName]
          self.sense(featureSDR, learn=False)

          inferred = (
            set(self.objectLayer.getActiveCells()) ==
            set(self.objectRepresentations[objectName]) and

            set(self.inputLayer.getActiveCells()) ==
            set(self.inputRepresentations[(objectName,
                                           locationOnObject,
                                           featureName)]) and

            set(self.getActiveLocationCells()) ==
            set(self.locationRepresentations[(objectName, locationOnObject)]))

          if inferred:
            break

        prevTouchSequence = touchSequence

        if inferred:
          break


  def reset(self):
    for module in self.locationModules:
      module.reset()
    self.objectLayer.reset()
    self.inputLayer.reset()

    self.locationInWorld = None

    for monitor in self.monitors.values():
      monitor.afterReset()



class Grid2DLocationExperimentMonitor(object):
  """
  Abstract base class for a Grid2DLocationExperiment monitor.
  """

  __metaclass__ = abc.ABCMeta

  def beforeSense(self, featureSDR): pass
  def beforeMove(self, deltaLocation): pass
  def markSensoryRepetition(self): pass
  def afterReset(self): pass
  def afterWorldLocationChanged(self, locationInWorld): pass
  def afterLocationShift(self, **kwargs): pass
  def afterLocationAnchor(self, **kwargs): pass
  def afterInputCompute(self, **kwargs): pass
  def afterObjectCompute(self, **kwargs): pass
