#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
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
from regions.SequenceClassifierRegion import SequenceClassifierRegion
from nupic.engine import pyRegions

_VERBOSITY = 0

SP_PARAMS = {
  "spVerbosity": _VERBOSITY,
  "spatialImp": "cpp",
  "globalInhibition": 1,
  "columnCount": 2048,
  "numActiveColumnsPerInhArea": 40,
  "seed": 1956,
  "potentialPct": 0.8,
  "synPermConnected": 0.1,
  "synPermActiveInc": 0.0001,
  "synPermInactiveDec": 0.0005,
  "maxBoost": 1.0,
}

TM_PARAMS = {
  "verbosity": _VERBOSITY,
  "columnCount": 2048,
  "cellsPerColumn": 32,
  "seed": 1960,
  "temporalImp": "tm_py",
  "newSynapseCount": 20,
  "maxSynapsesPerSegment": 32,
  "maxSegmentsPerCell": 128,
  "initialPerm": 0.21,
  "permanenceInc": 0.1,
  "permanenceDec": 0.1,
  "globalDecay": 0.0,
  "maxAge": 0,
  "minThreshold": 9,
  "activationThreshold": 12,
  "outputType": "normal",
  "pamLength": 3,
}

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


def createSensorRegion(network, sensorType, encoders, dataSource, numCats):
  """
  Initializes the sensor region with an encoder and data source.

  @param network      (Network)

  @param sensorType   (str)           Specific type of region, e.g.
      "py.RecordSensor"; possible options can be found in /nupic/regions/.

  @param encoders     (dict, encoder) If adding multiple encoders, pass a dict
      as specified in createEncoder() docstring. Otherwise an encoder object is
      expected.

  @param dataSource   (RecordStream)  Sensor region reads data from here.
  
  @param numCats   (int) Maximum number of categories of the input data.

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
      raise RuntimeError("Could not find sensor \'{}\' to import.".
                         format(sensorName))

  try:
    # Add region to network
    regionParams = json.dumps({"verbosity": _VERBOSITY,
                               "numCategories": numCats})
    network.addRegion("sensor", sensorType, regionParams)
  except RuntimeError:
    print ("Custom region not added correctly. Possible issues are the spec is "
           "wrong or the region class is not in the Python path.")
    return

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


def createSpatialPoolerRegion(network, prevRegionWidth):
  """
  Create the spatial pooler region.

  @param network          (Network)   The region will be a node in this network.
  @param prevRegionWidth  (int)       Width of region below.
  @return                 (Region)    SP region of the network.
  """
  # Add region to network
  SP_PARAMS["inputWidth"] = prevRegionWidth
  spatialPoolerRegion = network.addRegion(
    "SP", "py.SPRegion", json.dumps(SP_PARAMS))

  # Make sure learning is ON
  spatialPoolerRegion.setParameter("learningMode", True)

  # Inference mode outputs the current inference (i.e. active columns).
  # Okay to always leave inference mode on; only there for some corner cases.
  spatialPoolerRegion.setParameter("inferenceMode", True)

  return spatialPoolerRegion


def createTemporalMemoryRegion(network):
  """
  Create the temporal memory region.

  @param network          (Network)   The region will be a node in this network.
  @return                 (Region)    TM region of the network.
  """
  # Add region to network
  TM_PARAMS["inputWidth"] = TM_PARAMS["columnCount"]
  temporalMemoryRegion = network.addRegion(
    "TM", "py.TPRegion", json.dumps(TM_PARAMS))

  # Make sure learning is enabled (this is the default)
  temporalMemoryRegion.setParameter("learningMode", False)

  # We want to compute the predictedActiveCells
  #temporalMemoryRegion.setParameter("computePredictedActiveCellIndices", True)

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
  # Classifier region may be non-standard, so add custom region class to the network
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


def validateRegions(sensor, sp, tm, classifier):
  """ Make sure region widths fit"""

  sensorOutputWidth = sensor.encoder.getWidth()
  spInputWidth = sp.getSelf().inputWidth
  spOutputWidth = sp.getSelf().columnCount
  tmInputWidth = tm.getSelf().columnCount
  tmOutputWidth = tmInputWidth * tm.getSelf().cellsPerColumn

  if sensorOutputWidth != spInputWidth:
    raise ValueError("Region widths do not fit. Sensor output width = %s. SP input width = %s"
                     % (sensorOutputWidth, spInputWidth))

  if spOutputWidth != tmInputWidth:
    raise ValueError("Region widths do not fit. SP output width = %s. TM input width = %s"
                     % (spOutputWidth, tmInputWidth))

  # TODO: should we check if TM output width matches classifier input width? Not sure param exists.


def linkRegions(network):
  """Link the regions, as commented below."""

  # Link the SP region to the sensor input
  network.link("sensor", "SP", "UniformLink", "")

  # Forward the sensor region sequence reset to the SP
  network.link("sensor", "SP", "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  # Feed forward link from SP to TM
  network.link("SP", "TM", "UniformLink", "",
               srcOutput="bottomUpOut", destInput="bottomUpIn")

  # Feedback links (unnecessary??)
  network.link("TM", "SP", "UniformLink", "",
               srcOutput="topDownOut", destInput="topDownIn")
  network.link("TM", "sensor", "UniformLink", "",
               srcOutput="topDownOut", destInput="temporalTopDownIn")

  # Forward the sensor region sequence reset to the TM
  network.link("sensor", "TM", "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  # Feed the TM states to the classifier.
  network.link("TM", "classifier", "UniformLink", "",
               srcOutput="bottomUpOut", destInput="bottomUpIn")

  # Feed the predicted active cells to the classifier
  network.link("TM", "classifier", "UniformLink", "",
               srcOutput="predictedActiveCells", destInput="predictedActiveCells")

  # Link the sensor to the classifier to send in category labels.
  network.link("sensor", "classifier", "UniformLink", "",
               srcOutput="categoryOut", destInput="categoryIn")


def createNetwork(dataSource,
                  sensorType,
                  encoders,
                  numCategories,
                  classifierType,
                  classifierParams):
  """
  Create the network instance with regions for the sensor, SP, TM, and
  classifier. Before running, be sure to init w/ network.initialize().

  @param dataSource (RecordStream) Sensor region reads data from here.
  
  @param sensorType (str) Specific type of region, e.g. "py.RecordSensor";
    possible options can be found in nupic/regions/.
    
  @param encoders (dict) See createEncoder() docstring for format.
  
  @param numCategories (int) Max number of categories of the input data.
  
  @param classifierType (str) Specific type of classifier region, 
    e.g. "py.SequenceClassifier"; possible options can be found in nupic/regions/.
    
  @param classifierParams (dict) Parameters for the model. E.g. {'maxCategoryCount': 3} 
  
  @return (Network) Sample network: SensorRegion -> SP -> TM -> CLA classifier
  """
  network = Network()

  sensor = createSensorRegion(network,
                              sensorType,
                              encoders,
                              dataSource,
                              numCategories)

  sp = createSpatialPoolerRegion(network,
                                 sensor.encoder.getWidth())

  tm = createTemporalMemoryRegion(network)

  classifier = createClassifierRegion(network,
                                      classifierType,
                                      classifierParams)

  validateRegions(sensor, sp, tm, classifier)

  linkRegions(network)

  return network
