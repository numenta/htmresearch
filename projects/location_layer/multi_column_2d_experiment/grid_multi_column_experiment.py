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
A shared experiment class for inferring 2D objects with multiple cortical
columns
"""

import abc
import random

import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakPairMemory)
from htmresearch.algorithms.location_modules import (
  BodyToSpecificObjectModule2D, SensorToBodyModule2D,
  SensorToSpecificObjectModule)
from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.frameworks.layers.sensor_placement import greedySensorPositions


class CorticalColumn(object):
  """
  Data structure that holds each layer in the cortical column and stores
  classification info for each column.
  """
  def __init__(self, inputLayer, objectLayer, sensorToBodyModules,
               sensorToSpecificObjectModules):

    self.inputLayer = inputLayer
    self.objectLayer = objectLayer
    self.sensorToBodyModules = sensorToBodyModules
    self.sensorToSpecificObjectModules = sensorToSpecificObjectModules

    # Use these for classifying SDRs and for testing whether they're correct.
    self.sensorLocationRepresentations = {
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


  def getSensorToSpecificObjectSDR(self):
    activeCells = np.array([], dtype="uint32")

    totalPrevCells = 0
    for module in self.sensorToSpecificObjectModules:
      activeCells = np.append(activeCells,
                              module.activeCells + totalPrevCells)
      totalPrevCells += module.cellCount

    return activeCells


  def getAllCellActivity(self):
    return (set(self.objectLayer.getActiveCells()),
            set(self.inputLayer.getActiveCells()),
            tuple(set(module.activeCells)
                  for module in self.sensorToSpecificObjectModules))


class MultiColumn2DExperiment(object):
  """
  The experiment code organized into a class.
  """

  def __init__(self, objects, objectPlacements, featureNames, locationConfigs,
               numCorticalColumns, worldDimensions):

    self.objects = objects
    self.objectPlacements = objectPlacements
    self.numCorticalColumns = numCorticalColumns
    self.worldDimensions = worldDimensions
    self.locationConfigs = locationConfigs

    self.features = dict(
      ((iCol, k), np.array(sorted(random.sample(xrange(150), 15)), dtype="uint32"))
      for k in featureNames
      for iCol in xrange(numCorticalColumns))

    self.corticalColumns = []
    for _ in xrange(numCorticalColumns):
      inputLayer = ApicalTiebreakPairMemory(**{
        "columnCount": 150,
        "cellsPerColumn": 32,
        "initialPermanence": 1.0,
        "basalInputSize": sum(
          np.prod(config["cellDimensions"])
          for config in locationConfigs),
        "apicalInputSize": 4096,
        "seed": random.randint(0,2048)})

      objectLayer = ColumnPooler(**{
        "inputWidth": 150 * 32,
        "initialProximalPermanence": 1.0,
        "initialProximalPermanence": 1.0,
        "lateralInputWidths": [4096] * (numCorticalColumns - 1),
        "seed": random.randint(0,2048)})

      sensorToBodyModules = [SensorToBodyModule2D(**config)
                             for config in locationConfigs]

      sensorToSpecificObjectModules = [
        SensorToSpecificObjectModule(**{
          "cellDimensions": config["cellDimensions"],
          "anchorInputSize": inputLayer.numberOfCells(),
          "initialPermanence": 1.0,
          "seed": random.randint(0,2048)})
        for config in locationConfigs]

      self.corticalColumns.append(
        CorticalColumn(inputLayer, objectLayer, sensorToBodyModules,
                       sensorToSpecificObjectModules))

    self.bodyToSpecificObjectModules = []
    for iModule, config in enumerate(locationConfigs):
      module = BodyToSpecificObjectModule2D(config["cellDimensions"])
      pairedSensorModules = [c.sensorToSpecificObjectModules[iModule]
                             for c in self.corticalColumns]
      module.formReciprocalSynapses(pairedSensorModules)
      self.bodyToSpecificObjectModules.append(module)

    self.maxSettlingTime = 10

    self.monitors = {}
    self.nextMonitorToken = 1


  def addMonitor(self, monitor):
    """
    Subscribe to MultiColumn2DExperimentMonitor events.

    @param monitor (MultiColumn2DExperimentMonitor)
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


  def compute(self, egocentricLocationByColumn, featureSDRByColumn, learn):
    # Layer 5B
    paramsByModuleByColumn = [
      [{"egocentricLocation": egocentricLocation}
       for _ in c.sensorToBodyModules]
      for c, egocentricLocation in zip(self.corticalColumns,
                                       egocentricLocationByColumn)]

    for c, paramsByModule in zip(self.corticalColumns, paramsByModuleByColumn):
      for module, params in zip(c.sensorToBodyModules, paramsByModule):
        module.compute(**params)

    for monitor in self.monitors.itervalues():
      monitor.afterSensorToBodyCompute(paramsByModuleByColumn)


    # Layer 6A
    paramsByModuleByColumn = [
      [{"sensorToBody": sensorToBody.activeCells,
        "bodyToSpecificObject": bodyToSpecificObject.activeCells}
       for sensorToBody, sensorToSpecificObject, bodyToSpecificObject
       in zip(c.sensorToBodyModules,
              c.sensorToSpecificObjectModules,
              self.bodyToSpecificObjectModules)]
      for c in self.corticalColumns]

    for c, paramsByModule in zip(self.corticalColumns, paramsByModuleByColumn):
      for module, params in zip(c.sensorToSpecificObjectModules,
                                paramsByModule):
        module.metricCompute(**params)

    for monitor in self.monitors.itervalues():
      monitor.afterSensorMetricCompute(paramsByModuleByColumn)


    # Layer 4
    paramsByColumn = [
      {"activeColumns": featureSDR,
       "basalInput": c.getSensorToSpecificObjectSDR(),
       "apicalInput": c.objectLayer.getActiveCells(),
       "learn": learn}
      for c, featureSDR in zip(self.corticalColumns, featureSDRByColumn)
    ]

    for c, params in zip(self.corticalColumns, paramsByColumn):
      c.inputLayer.compute(**params)

    for monitor in self.monitors.itervalues():
      monitor.afterInputCompute(paramsByColumn)


    # Layer 2
    paramsByColumn = [
      {"feedforwardInput": c.inputLayer.getActiveCells(),
       "feedforwardGrowthCandidates": c.inputLayer.getWinnerCells(),
       "lateralInputs": [c2.objectLayer.getActiveCells()
                         for i2, c2 in enumerate(self.corticalColumns)
                         if i2 != i],
       "learn": learn}
      for i, c in enumerate(self.corticalColumns)]

    for c, params in zip(self.corticalColumns, paramsByColumn):
      c.objectLayer.compute(**params)

    for monitor in self.monitors.itervalues():
      monitor.afterObjectCompute(paramsByColumn)


    # Layer 6A
    paramsByColumn = [
      {"anchorInput": c.inputLayer.getActiveCells(),
       "learn": learn}
      for c in self.corticalColumns]

    for c, params in zip(self.corticalColumns, paramsByColumn):
      for module in c.sensorToSpecificObjectModules:
        module.anchorCompute(**params)

    for monitor in self.monitors.itervalues():
      monitor.afterSensorLocationAnchor(paramsByColumn)


    # Subcortical
    paramsByModule = [
      {"sensorToBodyByColumn": [
         c.sensorToBodyModules[iModule].activeCells
         for c in self.corticalColumns],
       "sensorToSpecificObjectByColumn": [
         c.sensorToSpecificObjectModules[iModule].activeCells
         for c in self.corticalColumns]}
      for iModule, module in enumerate(self.bodyToSpecificObjectModules)]

    for module, params in zip(self.bodyToSpecificObjectModules,
                              paramsByModule):
      module.compute(**params)

    for monitor in self.monitors.itervalues():
      monitor.afterBodyLocationAnchor(paramsByModule)


  def learnObjects(self, bodyPlacement):
    """
    Learn each provided object.

    This method simultaneously learns 4 sets of synapses:
    - sensor-to-specific-object -> input
    - input -> sensor-to-specific-object
    - input -> object
    - object -> input
    """
    for monitor in self.monitors.itervalues():
      monitor.afterBodyWorldLocationChanged(bodyPlacement)

    for objectName, objectFeatures in self.objects.iteritems():
      self.reset()

      objectPlacement = self.objectPlacements[objectName]

      for module in self.bodyToSpecificObjectModules:
        module.activateRandomLocation()

      for iFeatureStart in xrange(len(objectFeatures)):
        featureIndexByColumn = np.mod(np.arange(iFeatureStart,
                                                iFeatureStart +
                                                self.numCorticalColumns),
                                      len(objectFeatures))

        featureByColumn = [objectFeatures[iFeature]
                           for iFeature in featureIndexByColumn]
        featureSDRByColumn = [self.features[(iCol, feature["name"])]
                              for iCol, feature in enumerate(featureByColumn)]
        locationOnObjectByColumn = np.array(
          [[feature["top"] + feature["height"]/2,
            feature["left"] + feature["width"]/2]
           for feature in featureByColumn])

        worldLocationByColumn = objectPlacement + locationOnObjectByColumn
        egocentricLocationByColumn = worldLocationByColumn - bodyPlacement

        for monitor in self.monitors.itervalues():
          monitor.afterSensorWorldLocationChanged(worldLocationByColumn)

        for t in xrange(2):
          for monitor in self.monitors.itervalues():
            monitor.beforeCompute(egocentricLocationByColumn, featureSDRByColumn,
                                  isRepeat=(t > 0))
          self.compute(egocentricLocationByColumn, featureSDRByColumn,
                       learn=True)

        for iFeature, locationOnObject, c in zip(featureIndexByColumn,
                                                 locationOnObjectByColumn,
                                                 self.corticalColumns):
          locationOnObject = tuple(locationOnObject.tolist())
          c.sensorLocationRepresentations[(objectName, locationOnObject)] = (
            c.getSensorToSpecificObjectSDR())
          c.inputRepresentations[(objectName, locationOnObject,
                                  objectFeatures[iFeature]["name"])] = (
                                    c.inputLayer.getActiveCells())
          c.objectRepresentations[objectName] = c.objectLayer.getActiveCells()


  def inferObjectsWithTwoTouches(self, bodyPlacement):
    """
    Touch each object with multiple sensors twice.
    """
    for monitor in self.monitors.itervalues():
      monitor.afterBodyWorldLocationChanged(bodyPlacement)

    for objectName, objectFeatures in self.objects.iteritems():
      self.reset()

      objectPlacement = self.objectPlacements[objectName]

      featureIndexByColumnIterator = (
        greedySensorPositions(self.numCorticalColumns, len(objectFeatures)))

      for _ in xrange(2):
        # Choose where to place each sensor.
        featureIndexByColumn = featureIndexByColumnIterator.next()
        sensedFeatures = [objectFeatures[i] for i in featureIndexByColumn]
        featureSDRByColumn = [self.features[(iCol, feature["name"])]
                              for iCol, feature in enumerate(sensedFeatures)]
        worldLocationByColumn = np.array([
          [objectPlacement[0] + feature["top"] + feature["height"]/2,
           objectPlacement[1] + feature["left"] + feature["width"]/2]
          for feature in sensedFeatures])

        for monitor in self.monitors.itervalues():
          monitor.afterSensorWorldLocationChanged(worldLocationByColumn)

        egocentricLocationByColumn = worldLocationByColumn - bodyPlacement

        prevCellActivity = None

        for t in xrange(self.maxSettlingTime):
          for monitor in self.monitors.itervalues():
            monitor.beforeCompute(egocentricLocationByColumn, featureSDRByColumn,
                                  isRepeat=(t > 0))
          self.compute(egocentricLocationByColumn, featureSDRByColumn, learn=False)

          cellActivity = (
            tuple(c.getAllCellActivity()
                  for c in self.corticalColumns),
            tuple(set(module.activeCells)
                  for module in self.bodyToSpecificObjectModules))

          if cellActivity == prevCellActivity:
            # It settled. Cancel logging this timestep.
            for monitor in self.monitors.itervalues():
              monitor.clearUnflushedData()
            break
          else:
            prevCellActivity = cellActivity
            for monitor in self.monitors.itervalues():
              monitor.flush()


  def reset(self):
    for c in self.corticalColumns:
      for module in c.sensorToSpecificObjectModules:
        module.reset()
      c.inputLayer.reset()
      c.objectLayer.reset()

    for module in self.bodyToSpecificObjectModules:
      module.reset()

    for monitor in self.monitors.itervalues():
      monitor.afterReset()



class MultiColumn2DExperimentMonitor(object):
  """
  Abstract base class for a MultiColumn2DExperiment monitor.
  """

  __metaclass__ = abc.ABCMeta

  def beforeCompute(self, egocentricLocationByColumn, featureSDRByColumn,
                    isRepeat): pass
  def afterReset(self): pass
  def afterBodyWorldLocationChanged(self, bodyWorldLocation): pass
  def afterSensorWorldLocationChanged(self, worldLocationByColumn): pass
  def afterSensorToBodyCompute(self, paramsByColumn): pass
  def afterSensorMetricCompute(self, paramsByColumn): pass
  def afterSensorLocationAnchor(self, paramsByColumn): pass
  def afterBodyLocationAnchor(self, params): pass
  def afterInputCompute(self, paramsByColumn): pass
  def afterObjectCompute(self, paramsByColumn): pass
  def flush(self): pass
  def clearUnflushedData(self): pass
