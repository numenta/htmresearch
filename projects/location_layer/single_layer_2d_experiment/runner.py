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

import abc
from collections import defaultdict
import random

import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakPairMemory)
from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.algorithms.single_layer_location_memory import (
  SingleLayerLocationMemory)


class SingleLayerLocation2DExperiment(object):
  """
  The experiment code organized into a class.
  """

  def __init__(self, diameter, objects, featureNames):
    self.diameter = diameter

    self.objects = objects

    # A grid of location SDRs.
    self.locations = dict(
      ((i, j), np.array(sorted(random.sample(xrange(1000), 30)), dtype="uint32"))
      for i in xrange(diameter)
      for j in xrange(diameter))

    # 8 transition SDRs -- one for each straight and diagonal direction.
    self.transitions = dict(
      ((i, j), np.array(sorted(random.sample(xrange(1000), 30)), dtype="uint32"))
      for i in xrange(-1, 2)
      for j in xrange(-1, 2)
      if i != 0 or j != 0)

    self.features = dict(
      (k, np.array(sorted(random.sample(xrange(150), 15)), dtype="uint32"))
      for k in featureNames)

    self.locationLayer = SingleLayerLocationMemory(**{
      "cellCount": 1000,
      "deltaLocationInputSize": 1000,
      "featureLocationInputSize": 150*32,
      "sampleSize": 15,
      "activationThreshold": 10,
      "learningThreshold": 8,
    })

    self.inputLayer = ApicalTiebreakPairMemory(**{
      "columnCount": 150,
      "cellsPerColumn": 32,
      "basalInputSize": 1000,
      "apicalInputSize": 4096,
    })

    self.objectLayer = ColumnPooler(**{
      "inputWidth": 150 * 32
    })

    # Use these for classifying SDRs and for testing whether they're correct.
    self.inputRepresentations = {}
    self.objectRepresentations = {}
    self.learnedObjectPlacements = {}

    self.monitors = {}
    self.nextMonitorToken = 1


  def addMonitor(self, monitor):
    """
    Subscribe to SingleLayer2DExperiment events.

    @param monitor (SingleLayer2DExperimentMonitor)
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


  def doTimestep(self, locationSDR, transitionSDR, featureSDR,
                 egocentricLocation, learn):
    """
    Run one timestep.
    """

    for monitor in self.monitors.values():
      monitor.beforeTimestep(locationSDR, transitionSDR, featureSDR,
                             egocentricLocation, learn)

    params = {
      "newLocation": locationSDR,
      "deltaLocation": transitionSDR,
      "featureLocationInput": self.inputLayer.getActiveCells(),
      "featureLocationGrowthCandidates": self.inputLayer.getPredictedActiveCells(),
      "learn": learn,
    }
    self.locationLayer.compute(**params)
    for monitor in self.monitors.values():
      monitor.afterLocationCompute(**params)

    params = {
      "activeColumns": featureSDR,
      "basalInput": self.locationLayer.getActiveCells(),
      "apicalInput": self.objectLayer.getActiveCells(),
    }
    self.inputLayer.compute(**params)
    for monitor in self.monitors.values():
      monitor.afterInputCompute(**params)

    params = {
      "feedforwardInput": self.inputLayer.getActiveCells(),
      "feedforwardGrowthCandidates": self.inputLayer.getPredictedActiveCells(),
      "learn": learn,
    }
    self.objectLayer.compute(**params)
    for monitor in self.monitors.values():
      monitor.afterObjectCompute(**params)


  def learnTransitions(self):
    """
    Train the location layer to do path integration. For every location, teach
    it each previous-location + motor command pair.
    """

    print "Learning transitions"
    for (i, j), locationSDR in self.locations.iteritems():
      print "i, j", (i, j)
      for (di, dj), transitionSDR in self.transitions.iteritems():
        i2 = i + di
        j2 = j + dj
        if (0 <= i2 < self.diameter and
            0 <= j2 < self.diameter):
          for _ in xrange(5):
            self.locationLayer.reset()
            self.locationLayer.compute(newLocation=self.locations[(i,j)])
            self.locationLayer.compute(deltaLocation=transitionSDR,
                                       newLocation=self.locations[(i2, j2)])

    self.locationLayer.reset()


  def learnObjects(self, objectPlacements):
    """
    Learn each provided object in egocentric space. Touch every location on each
    object.

    This method doesn't try move the sensor along a path. Instead it just leaps
    the sensor to each object location, resetting the location layer with each
    leap.

    This method simultaneously learns 4 sets of synapses:
    - location -> input
    - input -> location
    - input -> object
    - object -> input
    """
    for monitor in self.monitors.values():
      monitor.afterPlaceObjects(objectPlacements)

    for objectName, objectDict in self.objects.iteritems():
      self.reset()

      objectPlacement = objectPlacements[objectName]

      for locationName, featureName in objectDict.iteritems():
        egocentricLocation = (locationName[0] + objectPlacement[0],
                              locationName[1] + objectPlacement[1])

        locationSDR = self.locations[egocentricLocation]
        featureSDR = self.features[featureName]
        transitionSDR = np.empty(0)

        self.locationLayer.reset()
        self.inputLayer.reset()

        for _ in xrange(10):
          self.doTimestep(locationSDR, transitionSDR, featureSDR,
                          egocentricLocation, learn=True)

        self.inputRepresentations[(featureName, egocentricLocation)] = (
          self.inputLayer.getActiveCells())

      self.objectRepresentations[objectName] = self.objectLayer.getActiveCells()
      self.learnedObjectPlacements[objectName] = objectPlacement


  def _selectTransition(self, allocentricLocation, objectDict, visitCounts):
    """
    Choose the transition that lands us in the location we've touched the least
    often. Break ties randomly, i.e. choose the first candidate in a shuffled
    list.
    """

    candidates = list(transition
                      for transition in self.transitions.keys()
                      if (allocentricLocation[0] + transition[0],
                          allocentricLocation[1] + transition[1]) in objectDict)
    random.shuffle(candidates)

    selectedVisitCount = None
    selectedTransition = None
    selectedAllocentricLocation = None

    for transition in candidates:
      candidateLocation = (allocentricLocation[0] + transition[0],
                           allocentricLocation[1] + transition[1])

      if (selectedVisitCount is None or
          visitCounts[candidateLocation] < selectedVisitCount):
        selectedVisitCount = visitCounts[candidateLocation]
        selectedTransition = transition
        selectedAllocentricLocation = candidateLocation

    return selectedAllocentricLocation, selectedTransition


  def inferObject(self, objectPlacements, objectName, startPoint,
                  transitionSequence, settlingTime=2):
    for monitor in self.monitors.values():
      monitor.afterPlaceObjects(objectPlacements)

    objectDict = self.objects[objectName]

    self.reset()

    allocentricLocation = startPoint
    nextTransitionSDR = np.empty(0, dtype="uint32")

    transitionIterator = iter(transitionSequence)

    try:
      while True:
        featureName = objectDict[allocentricLocation]
        egocentricLocation = (allocentricLocation[0] +
                              objectPlacements[objectName][0],
                              allocentricLocation[1] +
                              objectPlacements[objectName][1])
        featureSDR = self.features[featureName]

        steps = ([nextTransitionSDR] +
                 [np.empty(0)]*settlingTime)
        for transitionSDR in steps:
          self.doTimestep(np.empty(0), transitionSDR, featureSDR,
                          egocentricLocation, learn=False)

        transitionName = transitionIterator.next()
        allocentricLocation = (allocentricLocation[0] + transitionName[0],
                               allocentricLocation[1] + transitionName[1])
        nextTransitionSDR = self.transitions[transitionName]
    except StopIteration:
      pass


  def inferObjectsWithRandomMovements(self, objectPlacements, maxTouches=20,
                                      settlingTime=2):
    """
    Infer each object without any location input.
    """

    for monitor in self.monitors.values():
      monitor.afterPlaceObjects(objectPlacements)

    for objectName, objectDict in self.objects.iteritems():
      self.reset()

      visitCounts = defaultdict(int)

      learnedObjectPlacement = self.learnedObjectPlacements[objectName]

      allocentricLocation = random.choice(objectDict.keys())
      nextTransitionSDR = np.empty(0, dtype="uint32")

      # Traverse the object until it is inferred.
      success = False

      for _ in xrange(maxTouches):
        featureName = objectDict[allocentricLocation]
        egocentricLocation = (allocentricLocation[0] +
                              objectPlacements[objectName][0],
                              allocentricLocation[1] +
                              objectPlacements[objectName][1])
        featureSDR = self.features[featureName]

        steps = ([nextTransitionSDR] +
                 [np.empty(0)]*settlingTime)
        for transitionSDR in steps:
          self.doTimestep(np.empty(0), transitionSDR, featureSDR,
                          egocentricLocation, learn=False)

        visitCounts[allocentricLocation] += 1

        # We should eventually infer the egocentric location where we originally
        # learned this location on the object.
        learnedEgocentricLocation = (
          allocentricLocation[0] + learnedObjectPlacement[0],
          allocentricLocation[1] + learnedObjectPlacement[1])

        if (set(self.objectLayer.getActiveCells()) ==
            set(self.objectRepresentations[objectName]) and

            set(self.inputLayer.getActiveCells()) ==
            set(self.inputRepresentations[(featureName,
                                           learnedEgocentricLocation)]) and

            set(self.locationLayer.getActiveCells()) ==
            set(self.locations[learnedEgocentricLocation])):
          success = True
          break
        else:
          allocentricLocation, transitionName = self._selectTransition(
            allocentricLocation, objectDict, visitCounts)
          nextTransitionSDR = self.transitions[transitionName]


  def reset(self):
    self.locationLayer.reset()
    self.objectLayer.reset()
    self.inputLayer.reset()

    for monitor in self.monitors.values():
      monitor.afterReset()



class SingleLayer2DExperimentMonitor(object):
  """
  Abstract base class for a SingleLayer2DExperiment monitor.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def beforeTimestep(self, locationSDR, transitionSDR, featureSDR,
                     egocentricLocation, learn):
    pass

  @abc.abstractmethod
  def afterReset(self):
    pass

  @abc.abstractmethod
  def afterPlaceObjects(self, objectPlacements):
    pass
