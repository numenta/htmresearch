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
The methods here are a factory to create a classification network 
of any of sensor, SP, TM, UP, and classifier regions.
"""

try:
  import simplejson as json
except ImportError:
  import json
import numpy

from nupic.encoders import MultiEncoder
from nupic.engine import Network
from nupic.engine import pyRegions

from sensor_data_exp_settings import (TEST_PARTITION_NAME, DEBUG_VERBOSITY)

_PY_REGIONS = [r[1] for r in pyRegions]



def _createEncoder(encoders):
  """
  Creates and returns a MultiEncoder.

  @param encoders: (dict) Keys are the encoders' names, values are dicts of 
  the params; an example is shown below.
  @return encoder: (MultiEncoder) See nupic.encoders.multi.py. Example input:
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



def _createSensorRegion(network, regionConfig, dataSource):
  """
  Register a sensor region and initialize it the sensor region with an encoder 
  and data source.

  @param network: (Network) The network instance.
  @param regionConfig: (dict) configuration of the sensor region
  @param dataSource: (RecordStream) Sensor region reads data from here.
  @return sensorRegion: (PyRegion) Sensor region of the network.
  """
  regionType = regionConfig["regionType"]
  regionName = regionConfig["regionName"]
  regionParams = regionConfig["regionParams"]
  encoders = regionConfig["encoders"]

  _registerRegion(regionType)

  # Add region to network
  network.addRegion(regionName, regionType, json.dumps(regionParams))

  # getSelf() returns the actual region, instead of a region wrapper
  sensorRegion = network.regions[regionName].getSelf()

  # Specify how the sensor encodes input values
  if isinstance(encoders, dict):
    # Add encoder(s) from params dict:
    sensorRegion.encoder = _createEncoder(encoders)
  else:
    sensorRegion.encoder = encoders

  # Specify the dataSource as a file RecordStream instance
  sensorRegion.dataSource = dataSource

  return sensorRegion



def _registerRegion(regionType):
  """
  Sensor region may be non-standard, so add custom region class to the network.
  
  @param regionType: (str) type of the region. E.g py.SensorRegion.
  """
  regionTypeName = regionType.split(".")[1]
  sensorModule = regionTypeName  # conveniently have the same name
  if regionTypeName not in _PY_REGIONS:
    # Add new region class to the network
    try:
      module = __import__(sensorModule, {}, {}, regionTypeName)
      sensorClass = getattr(module, regionTypeName)
      Network.registerRegion(sensorClass)
      # Add region to list of registered PyRegions
      _PY_REGIONS.append(regionTypeName)
    except ImportError:
      raise RuntimeError(
        "Could not import sensor \'{}\'.".format(regionTypeName))



def _createRegion(network, regionConfig):
  """
  Create the SP, TM, UP or classifier region.

  @param network: (Network) The region will be a node in this network.
  @param regionConfig: (dict) The region configuration
  @return region: (PyRegion) region of the network.
  """
  regionName = regionConfig["regionName"]
  regionType = regionConfig["regionType"]
  regionParams = regionConfig["regionParams"]

  _registerRegion(regionType)

  # Add region to network
  region = network.addRegion(regionName, regionType, json.dumps(regionParams))

  # Disable learning at initialization. 
  region.setParameter("learningMode", False)

  # Inference mode outputs the current inference (i.e. active columns).
  # Okay to always leave inference mode on; only there for some corner cases.
  region.setParameter("inferenceMode", True)

  return region



def _linkRegions(network,
                 sensorRegionName,
                 previousRegionName,
                 currentRegionName):
  """
  Link the previous region to the current region and propagate the 
  sequence reset from the sensor region.
  
  @param network: (Network) regions to be linked are nodes in this network.
  @param sensorRegionName: (str) name of the sensor region
  @param previousRegionName: (str) parent node in the network
  @param currentRegionName: (str) current node in the network
  """
  network.link(previousRegionName, currentRegionName, "UniformLink", "")
  network.link(sensorRegionName, currentRegionName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")



def _validateRegionWidths(previousRegionWidth, currentRegionWidth):
  """
  Make sure previous and current region have compatible input and output width
  
  @param previousRegionWidth: (int) width of the previous region in the network
  @param currentRegionWidth: (int) width of the current region
  """

  if previousRegionWidth != currentRegionWidth:
    raise ValueError("Region widths do not fit. Output width = {}, "
                     "input width = {}.".format(previousRegionWidth,
                                                currentRegionWidth))



def createNetwork(dataSource, networkConfiguration):
  """
  Create and initialize the network instance with regions for the sensor, SP, 
  TM, and classifier. Before running, be sure to init w/ network.initialize().

  @param dataSource: (RecordStream) Sensor region reads data from here.
  @param networkConfiguration: (dict) the configuration of this network. 
  @return network: (Network) Sample network. E.g. Sensor -> SP -> TM -> Classif.
  """

  network = Network()

  # Create sensor regions (always enabled)
  sensorRegionConfig = networkConfiguration["sensorRegionConfig"]
  sensorRegionName = sensorRegionConfig["regionName"]
  sensorRegion = _createSensorRegion(network,
                                     sensorRegionConfig,
                                     dataSource)

  # Keep track of the previous region name and width to validate and link the 
  # input/output width of two consecutive regions.
  previousRegion = sensorRegionName
  previousRegionWidth = sensorRegion.encoder.getWidth()

  # Create SP region, if enabled.
  regionConfig = networkConfiguration["spRegionConfig"]
  regionName = regionConfig["regionName"]
  regionParams = regionConfig["regionParams"]
  regionEnabled = regionConfig["regionEnabled"]
  if regionEnabled:
    regionParams["inputWidth"] = sensorRegion.encoder.width
    spRegion = _createRegion(network, regionConfig)
    _validateRegionWidths(previousRegionWidth, spRegion.getSelf().inputWidth)
    _linkRegions(network,
                 sensorRegionName,
                 previousRegion,
                 regionName)
    previousRegion = regionName
    previousRegionWidth = spRegion.getSelf().columnCount

  # Create TM region, if enabled.
  regionConfig = networkConfiguration["tmRegionConfig"]
  regionName = regionConfig["regionName"]
  regionParams = regionConfig["regionParams"]
  regionEnabled = regionConfig["regionEnabled"]
  if regionEnabled:
    regionParams["inputWidth"] = regionParams["columnCount"]
    tmRegion = _createRegion(network, regionConfig)
    _validateRegionWidths(previousRegionWidth, tmRegion.getSelf().columnCount)
    _linkRegions(network,
                 sensorRegionName,
                 previousRegion,
                 regionName)
    previousRegion = regionName
    previousRegionWidth = tmRegion.getSelf().cellsPerColumn

  # Create UP region, if enabled.
  regionConfig = networkConfiguration["upRegionConfig"]
  regionName = regionConfig["regionName"]
  regionParams = regionConfig["regionParams"]
  regionEnabled = regionConfig["regionEnabled"]
  if regionEnabled:
    # TODO: not sure about the UP region width params. This needs to be updated.
    regionParams["inputWidth"] = previousRegionWidth
    upRegion = _createRegion(network, regionConfig)
    _validateRegionWidths(previousRegionWidth,
                          upRegion.getSelf().cellsPerColumn)
    _linkRegions(network,
                 sensorRegionName,
                 previousRegion,
                 regionName)
    previousRegion = regionName

  # Create classifier region (always enabled)
  regionConfig = networkConfiguration["classifierRegionConfig"]
  regionName = regionConfig["regionName"]
  _createRegion(network, regionConfig)
  # Link the classifier to previous region and sensor region - to send in 
  # category labels.
  network.link(previousRegion, regionName, "UniformLink", "")
  network.link(sensorRegionName,
               regionName,
               "UniformLink",
               "",
               srcOutput="categoryOut",
               destInput="categoryIn")

  return network



def _enableRegionLearning(network,
                          trainedRegionNames,
                          regionName,
                          recordNumber):
  """
  Enable learning for a specific region.
  
  @param network: (Network) the network instance 
  @param trainedRegionNames: (list) regions that have been trained on the 
  input data.
  @param regionName: (str) name of the current region  
  @param recordNumber: (int) value of the current record number 
  """

  network.regions[regionName].setParameter("learningMode", True)
  phaseInfo = ("-> Training '%s'. RecordNumber=%s. Learning is ON for %s, "
               "but OFF for the remaining regions." % (regionName,
                                                       recordNumber,
                                                       trainedRegionNames))
  print phaseInfo



def _stopLearning(network, trainedRegionNames, recordNumber):
  """
  Disable learning for all trained regions.
  
  @param network: (Network) the network instance 
  @param trainedRegionNames: (list) regions that have been trained on the 
  input data.
  @param recordNumber: (int) value of the current record number 
  """

  for regionName in trainedRegionNames:
    region = network.regions[regionName]
    region.setParameter("learningMode", False)

  phaseInfo = ("-> Test phase. RecordNumber=%s. "
               "Learning is OFF for all regions: %s" % (recordNumber,
                                                        trainedRegionNames))
  print phaseInfo



def runNetwork(network, networkConfig, partitions, numRecords):
  """
  Run the network and write classification results output.
  
  @param networkConfig: (dict) the config of the network.
  @param partitions: (list of tuples) Region names and associated indices 
  partitioning the input dataset to indicate at which recordNumber it should 
  start learning. The remaining of the data (last partition) is used as a test 
  set. 
  @param numRecords: (int) Number of records of the input dataset.
  @param network: (Network) a Network instance to run.
  """
  sensorRegionName = getRegionConfigParam(networkConfig,
                                          "sensorRegionConfig",
                                          "regionName")
  classifierRegionName = getRegionConfigParam(networkConfig,
                                              "classifierRegionConfig",
                                              "regionName")
  classifierRegionType = getRegionConfigParam(networkConfig,
                                              "classifierRegionConfig",
                                              "regionType")
  sensorRegion = network.regions[sensorRegionName]
  classifierRegion = network.regions[classifierRegionName]
  testIndex = partitions[-1][1]

  # keep track of the regions that have been trained
  trainedRegionNames = []
  numCorrect = 0
  numTestRecords = 0
  for recordNumber in xrange(numRecords):
    # Run the network for a single iteration
    network.run(1)

    actualValue = sensorRegion.getOutputData("categoryOut")[0]

    partitionName = partitions[0][0]
    partitionIndex = partitions[0][1]

    # train all of the regions
    if partitionIndex == recordNumber:

      if partitionName == TEST_PARTITION_NAME:
        _stopLearning(network, trainedRegionNames, recordNumber)

      else:
        partitions.pop(0)
        trainedRegionNames.append(partitionName)
        _enableRegionLearning(network,
                              trainedRegionNames,
                              partitionName,
                              recordNumber)

    # Evaluate the predictions on the test set.
    if recordNumber >= testIndex:
      if classifierRegionType == "py.KNNClassifierRegion":
        # The use of numpy.lexsort() here is to first sort by labelFreq, then 
        # sort by random values; this breaks ties in a random manner.
        inferenceValues = classifierRegion.getOutputData("categoriesOut")
        randomValues = numpy.random.random(inferenceValues.size)
        inferredValue = numpy.lexsort((randomValues, inferenceValues))[-1]
      elif classifierRegionType == "py.CLAClassifierRegion":
        inferredValue = classifierRegion.getOutputData("categoriesOut")[0]
      if actualValue == inferredValue:
        numCorrect += 1
      elif DEBUG_VERBOSITY > 0:
        print ("[DEBUG] recordNum=%s, actualValue=%s, inferredValue=%s"
               % (recordNumber, actualValue, inferredValue))
      numTestRecords += 1

  predictionAccuracy = round(100.0 * numCorrect / numTestRecords, 2)

  results = ("RESULTS: accuracy=%s | %s correctly predicted records out of %s "
             "test records \n" % (predictionAccuracy,
                                  numCorrect,
                                  numTestRecords))
  print results

  return numCorrect, numTestRecords, predictionAccuracy



def configureNetwork(dataSource, networkParams):
  """
  Configure the network for various experiment values.
  
  @param dataSource: (RecordStream) CSV file record stream.
  @param networkParams: (dict) the configuration of this network.
  """

  # if the sensor region has a scalar encoder, then set the min and max values. 
  encoderType = getEncoderParam(networkParams, "scalarEncoder", "type")
  if encoderType is not None:
    _setScalarEncoderMinMax(networkParams, dataSource)

  network = createNetwork(dataSource, networkParams)

  # Need to init the network before it can run.
  network.initialize()
  return network



def setRegionConfigParam(networkConfig,
                         regionConfigKey,
                         regionConfigParamKey,
                         regionConfigParamValue):
  """
  Set the value of a region config parameter. E.g. 'regionName' in:
  
  "spRegionConfig": {
    "regionEnabled": true,
    "regionName": "SP",
    "regionType": "py.SPRegion",
    "regionParams": {
      "spVerbosity": 0,
      "spatialImp": "cpp",
      "globalInhibition": 1,
      "columnCount": 2048,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "maxBoost": 1.0
    }
  }
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  @param regionConfigParamKey: (str) name of the region config param to 
  update. E.g. 'regionType'.
  @param regionConfigParamValue: value of the region config param.
  """

  networkConfig[regionConfigKey][regionConfigParamKey] = regionConfigParamValue



def setRegionParam(networkConfig,
                   regionConfigKey,
                   regionParamKey,
                   regionParamValue):
  """
  Set value of a key in the 'regionParams' dict of a region config. See:
  
  "spRegionConfig": {
    "regionEnabled": true,
    "regionName": "SP",
    "regionType": "py.SPRegion",
    "regionParams": {
      "spVerbosity": 0,
      "spatialImp": "cpp",
      "globalInhibition": 1,
      "columnCount": 2048,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "maxBoost": 1.0
    }
  }
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  @param regionParamKey: (str) name of the region param to update. E.g. 
  'inputWidth'.
  @param regionParamValue: value of the region param.
  """

  (networkConfig[regionConfigKey]['regionParams']
   [regionParamKey]) = regionParamValue



def getRegionConfigParam(networkConfig,
                         regionConfigKey,
                         regionConfigParamKey):
  """
  Get the value of a region config parameter. E.g. 'regionName' in:
  
  "spRegionConfig": {
    "regionEnabled": true,
    "regionName": "SP",
    "regionType": "py.SPRegion",
    "regionParams": {
      "spVerbosity": 0,
      "spatialImp": "cpp",
      "globalInhibition": 1,
      "columnCount": 2048,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "maxBoost": 1.0
    }
  }
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  @param regionConfigParamKey: (str) name of the region config param to 
  update. E.g. 'regionType'.
  @return regionConfigParamValue: value of the region config param.
  """

  return networkConfig[regionConfigKey].get(regionConfigParamKey)



def getRegionParam(networkConfig,
                   regionConfigKey,
                   regionParamKey,
                   regionParamValue):
  """
  Get value of a key in the 'regionParams' dict of a region config. See:
  
  "spRegionConfig": {
    "regionEnabled": true,
    "regionName": "SP",
    "regionType": "py.SPRegion",
    "regionParams": {
      "spVerbosity": 0,
      "spatialImp": "cpp",
      "globalInhibition": 1,
      "columnCount": 2048,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "maxBoost": 1.0
    }
  }
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  @param regionParamKey: (str) name of the region param to update. E.g. 
  'inputWidth'.
  @param regionParamValue: value of the region param.
  """

  (networkConfig[regionConfigKey]['regionParams']
   [regionParamKey]) = regionParamValue



def setEncoderParam(networkConfig,
                    encoderName,
                    paramName,
                    paramValue):
  """
  Set the value of an encoder parameter for the sensor region.
  
  @param networkConfig: (dict) the config of the network
  @param encoderName: (str) name of the encoder. E.g. 'scalar'.
  @param paramName: (str) name of the encoder param to update. 
  E.g. 'minval'.
  @param paramValue: value of the encoder param.
  """

  (networkConfig["sensorRegionConfig"]["encoders"][encoderName]
   [paramName]) = paramValue



def getEncoderParam(networkConfig, encoderName, paramName):
  """
  Get the value of an encoder parameter for the sensor region.
  
  @param networkConfig: (dict) the configuration of the network
  @param encoderName: (str) name of the encoder. E.g. 'ScalarEncoder'.
  @param paramName: (str) name of the param to update. E.g. 'minval'.
  @return paramValue: None if key 'paramName' does not exist. Value otherwise.
  """

  return networkConfig["sensorRegionConfig"]["encoders"][encoderName].get(
    paramName)



def _setScalarEncoderMinMax(networkConfig, dataSource):
  """
  Set the min and max values of a scalar encoder.
  
  @param networkConfig: (dict) configuration of the network.
  @param dataSource: (RecordStream) the input source
  """

  fieldName = getEncoderParam(networkConfig, "scalarEncoder", "fieldname")
  minval = dataSource.getFieldMin(fieldName)
  maxval = dataSource.getFieldMax(fieldName)
  setEncoderParam(networkConfig, "scalarEncoder", "minval", minval)
  setEncoderParam(networkConfig, "scalarEncoder", "maxval", maxval)



def setRegionEnabled(networkConfig,
                     regionConfigKey,
                     regionEnabledValue):
  """
  Set the value of the config param 'regionEnabled'.
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  update. E.g. 'regionType'.
  @param regionEnabledValue: (bool) 1 if region enabled. 0 Otherwise.
  """
  setRegionConfigParam(networkConfig,
                       regionConfigKey,
                       "regionEnabled",
                       regionEnabledValue)



def getRegionEnabled(networkConfig,
                     regionConfigKey):
  """
  Get the value of the config param 'regionEnabled'.
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  update. E.g. 'regionType'.
  @retun regionEnabledValue: (bool) 1 if region enabled. 0 Otherwise.
  """
  return getRegionConfigParam(networkConfig,
                              regionConfigKey,
                              "regionEnabled")



def setRegionType(networkConfig,
                  regionConfigKey,
                  regionTypeValue):
  """
  Set the value of the config param 'regionType'.
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  update. E.g. 'regionType'.
  @param regionTypeValue: (str) Type of the region. E.g. "py.KNNClassifier"
  """
  setRegionConfigParam(networkConfig,
                       regionConfigKey,
                       "regionType",
                       regionTypeValue)



def getRegionType(networkConfig,
                  regionConfigKey):
  """
  Get the value of the config param 'regionType'.
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  update. E.g. 'regionType'.
  @return regionTypeValue: (str) Type of the region. E.g. "py.KNNClassifier"
  """
  return getRegionConfigParam(networkConfig,
                              regionConfigKey,
                              "regionType")



def setRegionParams(networkConfig,
                    regionConfigKey,
                    regionParamsValue):
  """
  Set the value of the config param 'regionParams'.
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  update. E.g. 'regionType'.
  @param regionParamsValue: (dict) params of the region. E.g for a 
  region of type "py.KNNClassifier":
      {
        "k": 1,
        "distThreshold": 0,
        "maxCategoryCount": 10
      }
  """
  setRegionConfigParam(networkConfig,
                       regionConfigKey,
                       "regionParams",
                       regionParamsValue)



def getRegionParams(networkConfig,
                    regionConfigKey):
  """
  Get the value of the config param 'regionParams'.
  
  @param networkConfig: (dict) the config of the network.
  @param regionConfigKey: (str) name of the region config for which we want to 
  modify the param. E.g. 'sensorRegionConfig'.
  update. E.g. 'regionType'.
  @return regionParamsValue: (dict) params of the region. E.g for a 
  region of type "py.KNNClassifier":
      {
        "k": 1,
        "distThreshold": 0,
        "maxCategoryCount": 10
      }
  """
  return getRegionConfigParam(networkConfig,
                              regionConfigKey,
                              "regionParams")
