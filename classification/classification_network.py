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

PY_REGIONS = [r[1] for r in pyRegions]



def createEncoder(newEncoders):
  """
  Creates and returns a MultiEncoder.

  @param newEncoders    (dict)          Keys are the encoders' names, values are
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
  encoder.addMultipleEncoders(newEncoders)
  return encoder



def createSensorRegion(network, sensorType, sensorParams, dataSource, encoders):
  """
  Initializes the sensor region with an encoder and data source.

  @param network      (Network)

  @param sensorType   (str)           Specific type of region, e.g.
      "py.RecordSensor"; possible options can be found in /nupic/regions/.
      
  @param sensorParams (int) Maximum number of categories of the input data.

  @param dataSource   (RecordStream)  Sensor region reads data from here.
  
  @param encoders     (dict, encoder) If adding multiple encoders, pass a dict
      as specified in createEncoder() docstring. Otherwise an encoder object is
      expected.

  @return             (Region)        Sensor region of the network.
  """
  # Sensor region may be non-standard, so add custom region class to the network
  sensorName = sensorType.split(".")[1]
  sensorModule = sensorName  # conveniently have the same name
  if sensorName not in PY_REGIONS:
    # Add new region class to the network
    try:
      module = __import__(sensorModule, {}, {}, sensorName)
      sensorClass = getattr(module, sensorName)
      Network.registerRegion(sensorClass)
      # Add region to list of registered PyRegions
      PY_REGIONS.append(sensorName)
    except ImportError:
      raise RuntimeError("Could not import sensor \'{}\'.".format(sensorName))

  # Add region to network
  network.addRegion("sensor", sensorType, sensorParams)

  # getSelf() returns the actual region, instead of a region wrapper
  sensorRegion = network.regions["sensor"].getSelf()

  # Specify how the sensor encodes input values
  if isinstance(encoders, dict):
    # Add encoder(s) from params dict:
    sensorRegion.encoder = createEncoder(encoders)
  else:
    sensorRegion.encoder = encoders

  # Specify the dataSource as a file RecordStream instance
  sensorRegion.dataSource = dataSource

  return sensorRegion



def createSpatialPoolerRegion(network, spParams):
  """
  Create the spatial pooler region.

  @param network          (Network)   The region will be a node in this network.
  @param spParams         (dict)      The SP params
  @return                 (Region)    SP region of the network.
  """
  # Add region to network
  spatialPoolerRegion = network.addRegion(
    "SP", "py.SPRegion", json.dumps(spParams))

  # Make sure learning is ON.
  spatialPoolerRegion.setParameter("learningMode", True)

  # Inference mode outputs the current inference (i.e. active columns).
  # Okay to always leave inference mode on; only there for some corner cases.
  spatialPoolerRegion.setParameter("inferenceMode", True)

  return spatialPoolerRegion



def createTemporalMemoryRegion(network, tmParams):
  """
  Create the temporal memory region.

  @param network          (Network)   The region will be a node in this network.
  @param tmParams         (dict)      The params of the TM
  @return                 (Region)    TM region of the network.
  """
  # Add region to network
  tmParams["inputWidth"] = tmParams["columnCount"]
  temporalMemoryRegion = network.addRegion(
    "TM", "py.TPRegion", json.dumps(tmParams))

  # Make sure learning is enabled (this is the default)
  temporalMemoryRegion.setParameter("learningMode", False)

  # Inference mode outputs the current inference (i.e. active cells).
  # Okay to always leave inference mode on; only there for some corner cases.
  temporalMemoryRegion.setParameter("inferenceMode", True)

  return temporalMemoryRegion



def createClassifierRegion(network, classifierType, classifierParams):
  """
  Create classifier region.

  @param network (Network) The region will be a node in this network.
  
  @param classifierType (str) Specific type of region, e.g. 
    "py.CLAClassifierRegion"; possible options can be found in /nupic/regions/.
   
  @return (Region) Classifier region of the network.

  """
  # Classifier region may be non-standard, so add custom region class to the 
  # network
  if classifierType.split(".")[1] not in PY_REGIONS:
    # Add new region class to the network
    network.registerRegion(SequenceClassifierRegion)
    PY_REGIONS.append(classifierType.split(".")[1])

  # Create the classifier region.
  classifierRegion = network.addRegion(
    "classifier", classifierType, json.dumps(classifierParams))

  # Disable learning for now (will be enabled in a later training phase)
  classifierRegion.setParameter("learningMode", False)

  # Okay to always leave inference mode on; only there for some corner cases.
  classifierRegion.setParameter("inferenceMode", True)

  return classifierRegion



def createUnionPoolerRegion(network, upParams):
  """
  Create a Union Pooler region.
  
  @param network (Network) The region will be a node in this network.
  
  @param upParams (dict) The UP params
  
  @return (Region) Union Pooler region of the network.
  """
  return



def linkRegions(network, previousRegion, currentRegion):
  """Link the regions, as commented below."""

  # Link the previous region to the current region
  network.link(previousRegion, currentRegion, "UniformLink", "")

  # Propagate the sequence reset
  network.link(previousRegion, currentRegion, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")



def validateRegionWidths(previousRegionWidth, currentRegionWidth):
  """
  Make sure previous and current region have compatible input and output width
  """

  if previousRegionWidth != currentRegionWidth:
    raise ValueError("Region widths do not fit. Output width = {}, "
                     "input width = {}.".format(previousRegionWidth,
                                                currentRegionWidth))



def createNetwork(dataSource,
                  encoders,
                  networkConfiguration):
  """
  TODO: update doc string with new params
  
  Create and initialize the network instance with regions for the sensor, SP, 
  TM, and classifier. Before running, be sure to init w/ network.initialize().

  @param dataSource (RecordStream) Sensor region reads data from here.
    
  @param encoders (dict) See createEncoder() docstring for format.
  
  @param networkConfiguration (dict) the configuration of this network. E.g.
    {
      "sensorRegion":
        {
          "type": "py.RecordSensor",
          "params": RECORD_SENSOR_PARAMS
        },
      "spRegion":
        {
          "enabled": True,
          "params": SP_PARAMS,
        },
      "tmRegion":
        {
          "enabled": True,
          "params": TM_PARAMS,
        },
      "upRegion":
        {
          "enabled": False,
          "params": UP_PARAMS,
        },
      "classifierRegion":
        {
          "type": "py.CLAClassifierRegion",
          "params": CLA_CLASSIFIER_PARAMS
            
        }
    }
  
  @return network (Network) Sample network. 
    E.g. SensorRegion -> SP -> TM -> CLAClassifier
  """
  network = Network()

  sensorParams = networkConfiguration["sensorRegion"]["params"]
  sensorType = networkConfiguration["sensorRegion"]["type"]
  sensorRegion = createSensorRegion(network,
                                    sensorType,
                                    sensorParams,
                                    dataSource,
                                    encoders)

  # keep track of the previous region name and width to validate and link the 
  # input/output width of two consecutive regions.
  previousRegion = "sensor"
  previousRegionWidth = sensorRegion.encoder.getWidth()

  if networkConfiguration["spRegion"]["enabled"]:
    spParams = networkConfiguration["spRegion"]["params"]
    spParams["inputWidth"] = sensorRegion.encoder.getWidth
    spRegion = createSpatialPoolerRegion(network, spParams)
    linkRegions(network, previousRegion, "SP")
    validateRegionWidths(previousRegionWidth, spRegion.getSelf().inputWidth)
    previousRegion = "SP"
    previousRegionWidth = spRegion.getSelf().columnCount

  if networkConfiguration["tmRegion"]["enabled"]:
    tmParams = networkConfiguration["tmRegion"]["params"]
    tmRegion = createTemporalMemoryRegion(network, tmParams)
    linkRegions(network, previousRegion, "TM")
    validateRegionWidths(previousRegionWidth, tmRegion.getSelf().columnCount)
    previousRegion = "TM"
    previousRegionWidth = tmRegion.getSelf().cellsPerColumn

  if networkConfiguration["upRegion"]["enabled"]:
    upParams = networkConfiguration["upRegion"]["params"]
    upRegion = createUnionPoolerRegion(network, upParams)
    linkRegions(network, previousRegion, "UP")
    # TODO: not sure about the UP region width params. This needs to be updated.
    validateRegionWidths(previousRegionWidth, upRegion.getSelf().cellsPerColumn)
    previousRegion = "UP"

  classifierParams = networkConfiguration["classifierRegion"]["params"]
  classifierType = networkConfiguration["classifierRegion"]["type"]
  createClassifierRegion(network,
                         classifierType,
                         classifierParams)
  linkRegions(network, previousRegion, "classifier")

  # Link the sensor to the classifier to send in category labels.
  network.link("sensor", "classifier", "UniformLink", "",
               srcOutput="categoryOut", destInput="categoryIn")

  return network
