#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
The methods here are a factory to create a classification network:
  encoder -> SP -> TM -> (UP) -> classifier
"""

try:
  import simplejson as json
except ImportError:
  import json

from nupic.encoders import MultiEncoder
from nupic.engine import Network
from nupic.engine import pyRegions

from regions.SequenceClassifierRegion import SequenceClassifierRegion
from settings import (SENSOR_REGION_NAME,
                      SP_REGION_NAME,
                      TM_REGION_NAME,
                      UP_REGION_NAME,
                      CLASSIFIER_REGION_NAME)

_PY_REGIONS = [r[1] for r in pyRegions]



def createEncoder(encoders):
  """
  Creates and returns a MultiEncoder.

  @param encoders    (dict)          Keys are the encoders' names, values are
      dicts of the params; an example is shown below.

  @return encoder       (MultiEncoder)  See nupic.encoders.multi.py.

  Example input:
    {"energy": {"fieldname": u"energy",
                "type": "ScalarEncoder",
                "name": u"consumption",
                "minval": 0.0,
                "maxval": 100.0,
                "w": 21,
                "n": 500},
     "timestamp": {"fieldname": u"timestamp",
                   "type": "DateEncoder",
                   "name": u"timestamp_timeOfDay",
                   "timeOfDay": (21, 9.5)},
    }
  """
  encoder = MultiEncoder()
  encoder.addMultipleEncoders(encoders)
  return encoder



def createSensorRegion(network,
                       regionName,
                       sensorType,
                       sensorParams,
                       dataSource,
                       encoders):
  """
  Initializes the sensor region with an encoder and data source.

  @param network      (Network)
  
  @param regionName   (str) Name for this region

  @param sensorType   (str)           Specific type of region, e.g.
      "py.RecordSensor"; possible options can be found in /nupic/regions/.
      
  @param sensorParams (dict) parameters for the sensor region

  @param dataSource   (RecordStream)  Sensor region reads data from here.
  
  @param encoders     (dict, encoder) If adding multiple encoders, pass a dict
      as specified in createEncoder() docstring. Otherwise an encoder object is
      expected.

  @return             (Region)        Sensor region of the network.
  """
  # Sensor region may be non-standard, so add custom region class to the network
  sensorName = sensorType.split(".")[1]
  sensorModule = sensorName  # conveniently have the same name
  if sensorName not in _PY_REGIONS:
    # Add new region class to the network
    try:
      module = __import__(sensorModule, {}, {}, sensorName)
      sensorClass = getattr(module, sensorName)
      Network.registerRegion(sensorClass)
      # Add region to list of registered PyRegions
      _PY_REGIONS.append(sensorName)
    except ImportError:
      raise RuntimeError("Could not import sensor \'{}\'.".format(sensorName))

  # Add region to network
  network.addRegion(regionName, sensorType, json.dumps(sensorParams))

  # getSelf() returns the actual region, instead of a region wrapper
  sensorRegion = network.regions[regionName].getSelf()

  # Specify how the sensor encodes input values
  if isinstance(encoders, dict):
    # Add encoder(s) from params dict:
    sensorRegion.encoder = createEncoder(encoders)
  else:
    sensorRegion.encoder = encoders

  # Specify the dataSource as a file RecordStream instance
  sensorRegion.dataSource = dataSource

  return sensorRegion



def createSpatialPoolerRegion(network, regionName, spParams):
  """
  Create the spatial pooler region.

  @param network          (Network)   The region will be a node in this network.
  @param regionName       (str) Name for this region
  @param spParams         (dict)      The SP params
  @return                 (Region)    SP region of the network.
  """
  # Add region to network
  spatialPoolerRegion = network.addRegion(
    regionName, "py.SPRegion", json.dumps(spParams))

  # Learning is disabled at initialization. Can be enabled later.
  spatialPoolerRegion.setParameter("learningMode", False)

  # Inference mode outputs the current inference (i.e. active columns).
  # Okay to always leave inference mode on; only there for some corner cases.
  spatialPoolerRegion.setParameter("inferenceMode", True)

  return spatialPoolerRegion



def createTemporalMemoryRegion(network, regionName, tmParams):
  """
  Create the temporal memory region.

  @param network          (Network)   The region will be a node in this network.
  @param regionName       (str) Name for this region
  @param tmParams         (dict)      The params of the TM
  @return                 (Region)    TM region of the network.
  """
  # Add region to network
  tmParams["inputWidth"] = tmParams["columnCount"]
  temporalMemoryRegion = network.addRegion(
    regionName, "py.TPRegion", json.dumps(tmParams))

  # Learning is disabled at initialization. Can be enabled later.
  temporalMemoryRegion.setParameter("learningMode", False)

  # Inference mode outputs the current inference (i.e. active cells).
  # Okay to always leave inference mode on; only there for some corner cases.
  temporalMemoryRegion.setParameter("inferenceMode", True)

  return temporalMemoryRegion



def createClassifierRegion(network,
                           regionName,
                           classifierType,
                           classifierParams):
  """
  Create classifier region.

  @param network (Network) The region will be a node in this network.
  
  @param regionName (str) Name for this region
  
  @param classifierType (str) Specific type of region, e.g. 
    "py.CLAClassifierRegion"; possible options can be found in /nupic/regions/.
   
  @return (Region) Classifier region of the network.

  """
  # Classifier region may be non-standard, so add custom region class to the 
  # network
  if classifierType.split(".")[1] not in _PY_REGIONS:
    # Add new region class to the network
    network.registerRegion(SequenceClassifierRegion)
    _PY_REGIONS.append(classifierType.split(".")[1])

  # Create the classifier region.
  classifierRegion = network.addRegion(
    regionName, classifierType, json.dumps(classifierParams))

  # Disable learning for now (will be enabled in a later training phase)
  classifierRegion.setParameter("learningMode", False)

  # Okay to always leave inference mode on; only there for some corner cases.
  classifierRegion.setParameter("inferenceMode", True)

  return classifierRegion



def createUnionPoolerRegion(network, regionName, upParams):
  """
  Create a Union Pooler region.
  
  @param network (Network) The region will be a node in this network.
  
  @param regionName (str) Name for this region
  
  @param upParams (dict) The UP params
  
  @return (Region) Union Pooler region of the network.
  """
  pass  # TODO: implement UP regions creation. Make sure learning is off. 



def linkRegions(network, sensorRegionName, previousRegion, currentRegion):
  """
  Link the previous region to the current region and propagate the 
  sequence reset from the sensor region.
  @param network (Network) regions to be linked are nodes in this network.
  @param sensorRegionName (str) name of the sensor region
  @param previousRegion (PyRegion) parent node in the network
  @param currentRegion (PyRegion) current node in the network
  """
  network.link(previousRegion, currentRegion, "UniformLink", "")
  network.link(sensorRegionName, currentRegion, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")



def validateRegionWidths(previousRegionWidth, currentRegionWidth):
  """
  Make sure previous and current region have compatible input and output width
  @param previousRegionWidth (int) width of the previous region in the network
  @param currentRegionWidth (int) width of the current region
  """

  if previousRegionWidth != currentRegionWidth:
    raise ValueError("Region widths do not fit. Output width = {}, "
                     "input width = {}.".format(previousRegionWidth,
                                                currentRegionWidth))



def createNetwork(dataSource,
                  encoders,
                  networkConfiguration):
  """
  Create and initialize the network instance with regions for the sensor, SP, 
  TM, and classifier. Before running, be sure to init w/ network.initialize().

  @param dataSource (RecordStream) Sensor region reads data from here.
    
  @param encoders (dict) See createEncoder() docstring for format.
  
  @param networkConfiguration (dict) the configuration of this network. See 
  settings.py for examples.
  
  @return network (Network) Sample network. 
    E.g. SensorRegion -> SP -> TM -> CLAClassifier
  """
  network = Network()

  # Create sensor regions (always enabled)
  sensorParams = networkConfiguration[SENSOR_REGION_NAME]["params"]
  sensorType = networkConfiguration[SENSOR_REGION_NAME]["type"]
  sensorRegion = createSensorRegion(network,
                                    SENSOR_REGION_NAME,
                                    sensorType,
                                    sensorParams,
                                    dataSource,
                                    encoders)

  # Keep track of the previous region name and width to validate and link the 
  # input/output width of two consecutive regions.
  previousRegion = SENSOR_REGION_NAME
  previousRegionWidth = sensorRegion.encoder.getWidth()

  # Create SP region, if enabled.
  if networkConfiguration["spRegion"]["enabled"]:
    spParams = networkConfiguration[SP_REGION_NAME]["params"]
    spParams["inputWidth"] = sensorRegion.encoder.width
    spRegion = createSpatialPoolerRegion(network, SP_REGION_NAME, spParams)
    validateRegionWidths(previousRegionWidth, spRegion.getSelf().inputWidth)
    linkRegions(network,
                SENSOR_REGION_NAME,
                previousRegion,
                SP_REGION_NAME)
    previousRegion = SP_REGION_NAME
    previousRegionWidth = spRegion.getSelf().columnCount

  # Create TM region, if enabled.
  if networkConfiguration[TM_REGION_NAME]["enabled"]:
    tmParams = networkConfiguration[TM_REGION_NAME]["params"]
    tmRegion = createTemporalMemoryRegion(network, TM_REGION_NAME, tmParams)
    validateRegionWidths(previousRegionWidth, tmRegion.getSelf().columnCount)
    linkRegions(network,
                SENSOR_REGION_NAME,
                previousRegion,
                TM_REGION_NAME)
    previousRegion = TM_REGION_NAME
    previousRegionWidth = tmRegion.getSelf().cellsPerColumn

  # Create UP region, if enabled.
  if networkConfiguration[UP_REGION_NAME]["enabled"]:
    upParams = networkConfiguration[UP_REGION_NAME]["params"]
    upRegion = createUnionPoolerRegion(network, upParams)
    # TODO: not sure about the UP region width params. This needs to be updated.
    validateRegionWidths(previousRegionWidth, upRegion.getSelf().cellsPerColumn)
    linkRegions(network,
                SENSOR_REGION_NAME,
                previousRegion,
                UP_REGION_NAME)
    previousRegion = UP_REGION_NAME

  # Create classifier region (always enabled)
  classifierParams = networkConfiguration[CLASSIFIER_REGION_NAME]["params"]
  classifierType = networkConfiguration[CLASSIFIER_REGION_NAME]["type"]
  createClassifierRegion(network,
                         CLASSIFIER_REGION_NAME,
                         classifierType,
                         classifierParams)
  # Link the classifier to previous region and sensor region - to send in 
  # category labels.
  network.link(previousRegion, CLASSIFIER_REGION_NAME, "UniformLink", "")
  network.link(SENSOR_REGION_NAME,
               CLASSIFIER_REGION_NAME,
               "UniformLink",
               "",
               srcOutput="categoryOut",
               destInput="categoryIn")

  return network
