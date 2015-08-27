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

from nupic.data.file_record_stream import FileRecordStream
from nupic.encoders import MultiEncoder
from nupic.engine import Network
from nupic.engine import pyRegions

from network_params import (CLASSIFIER_REGION_NAME,
                            SENSOR_REGION_NAME,
                            SCALAR_ENCODER_NAME,
                            SP_REGION_NAME,
                            TM_REGION_NAME,
                            UP_REGION_NAME,
                            KNN_CLASSIFIER_TYPE,
                            CLA_CLASSIFIER_TYPE)
from settings import (TEST_PARTITION_NAME,
                      DEBUG_VERBOSITY)

_PY_REGIONS = [r[1] for r in pyRegions]



def _createEncoder(encoders):
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



def _createSensorRegion(network,
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
    sensorRegion.encoder = _createEncoder(encoders)
  else:
    sensorRegion.encoder = encoders

  # Specify the dataSource as a file RecordStream instance
  sensorRegion.dataSource = dataSource

  return sensorRegion



def _createSpatialPoolerRegion(network, regionName, spParams):
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

  # Enable learning at initialization. 
  spatialPoolerRegion.setParameter("learningMode", True)

  # Inference mode outputs the current inference (i.e. active columns).
  # Okay to always leave inference mode on; only there for some corner cases.
  spatialPoolerRegion.setParameter("inferenceMode", True)

  return spatialPoolerRegion



def _createTemporalMemoryRegion(network, regionName, tmParams):
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



def _createClassifierRegion(network,
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
  # Create the classifier region.
  classifierRegion = network.addRegion(
    regionName, classifierType, json.dumps(classifierParams))

  # Disable learning for now (will be enabled in a later training phase)
  classifierRegion.setParameter("learningMode", False)

  # Okay to always leave inference mode on; only there for some corner cases.
  classifierRegion.setParameter("inferenceMode", True)

  return classifierRegion



def _createUnionPoolerRegion(network, regionName, upParams):
  """
  Create a Union Pooler region.
  
  @param network (Network) The region will be a node in this network.
  
  @param regionName (str) Name for this region
  
  @param upParams (dict) The UP params
  
  @return (Region) Union Pooler region of the network.
  """
  pass  # TODO: implement UP regions creation. Make sure learning is off. 



def _linkRegions(network, sensorRegionName, previousRegion, currentRegion):
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



def _validateRegionWidths(previousRegionWidth, currentRegionWidth):
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
  sensorRegion = _createSensorRegion(network,
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
    spRegion = _createSpatialPoolerRegion(network, SP_REGION_NAME, spParams)
    _validateRegionWidths(previousRegionWidth, spRegion.getSelf().inputWidth)
    _linkRegions(network,
                 SENSOR_REGION_NAME,
                 previousRegion,
                 SP_REGION_NAME)
    previousRegion = SP_REGION_NAME
    previousRegionWidth = spRegion.getSelf().columnCount

  # Create TM region, if enabled.
  if networkConfiguration[TM_REGION_NAME]["enabled"]:
    tmParams = networkConfiguration[TM_REGION_NAME]["params"]
    tmRegion = _createTemporalMemoryRegion(network, TM_REGION_NAME, tmParams)
    _validateRegionWidths(previousRegionWidth, tmRegion.getSelf().columnCount)
    _linkRegions(network,
                 SENSOR_REGION_NAME,
                 previousRegion,
                 TM_REGION_NAME)
    previousRegion = TM_REGION_NAME
    previousRegionWidth = tmRegion.getSelf().cellsPerColumn

  # Create UP region, if enabled.
  if networkConfiguration[UP_REGION_NAME]["enabled"]:
    upParams = networkConfiguration[UP_REGION_NAME]["params"]
    upRegion = _createUnionPoolerRegion(network, upParams)
    # TODO: not sure about the UP region width params. This needs to be updated.
    _validateRegionWidths(previousRegionWidth,
                          upRegion.getSelf().cellsPerColumn)
    _linkRegions(network,
                 SENSOR_REGION_NAME,
                 previousRegion,
                 UP_REGION_NAME)
    previousRegion = UP_REGION_NAME

  # Create classifier region (always enabled)
  classifierParams = networkConfiguration[CLASSIFIER_REGION_NAME]["params"]
  classifierType = networkConfiguration[CLASSIFIER_REGION_NAME]["type"]
  _createClassifierRegion(network,
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



def _enableRegionLearning(network,
                          trainedRegionNames,
                          regionName,
                          recordNumber):
  """
  Enable learning for a specific region.
  
  @param network (Network) the network instance 
  @param trainedRegionNames (list) regions that have been trained on the 
  input data.
  @param regionName (str) name of the current region  
  @param recordNumber (int) value of the current record number 
  """

  region = network.regions[regionName]
  region.setParameter("learningMode", True)
  phaseInfo = ("-> Training '%s'. RecordNumber=%s. Learning is ON for %s, "
               "but OFF for the remaining regions." % (regionName,
                                                       recordNumber,
                                                       trainedRegionNames))
  print phaseInfo



def _stopLearning(network, trainedRegionNames, recordNumber):
  """
  Disable learning for all trained regions.
  
  @param network (Network) the network instance 
  @param trainedRegionNames (list) regions that have been trained on the 
  input data.
  @param recordNumber (int) value of the current record number 
  """

  for regionName in trainedRegionNames:
    region = network.regions[regionName]
    region.setParameter("learningMode", False)

  phaseInfo = ("-> Test phase. RecordNumber=%s. "
               "Learning is OFF for all regions: %s" % (recordNumber,
                                                        trainedRegionNames))
  print phaseInfo



def runNetwork(network, networkConfiguration, numRecords):
  """
  Run the network and write classification results output.
  
  @param networkConfiguration (dict) the configuration of this network.
  @param numRecords (int) Number of records of the input dataset.
  @param network (Network) a Network instance to run.
  """

  partitions = _findNumberOfPartitions(networkConfiguration, numRecords)
  sensorRegion = network.regions[SENSOR_REGION_NAME]
  classifierRegion = network.regions[CLASSIFIER_REGION_NAME]
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
      if classifierRegion.type == KNN_CLASSIFIER_TYPE:
        inferredValue = classifierRegion.getOutputData("categoriesOut").argmax()
      elif classifierRegion.type == CLA_CLASSIFIER_TYPE:
        inferredValue = classifierRegion.getOutputData("categoriesOut")[0]
      if actualValue == inferredValue:
        numCorrect += 1
      elif DEBUG_VERBOSITY > 0:
        print ("[DEBUG] recordNum=%s, actualValue=%s, inferredValue=%s"
               % (recordNumber, actualValue, inferredValue))
      numTestRecords += 1

  predictionAccuracy = 100.0 * numCorrect / numTestRecords

  results = ("RESULTS: accuracy=%s | %s correctly predicted records out of %s "
             "test records \n" % (predictionAccuracy,
                                  numCorrect,
                                  numTestRecords))
  print results

  return numCorrect, numTestRecords, predictionAccuracy



def configureNetwork(inputFile,
                     networkConfiguration):
  """
  Configure the network for various experiment values.
  
  @param inputFile (str) file containing the input data that will be fed to the 
  network
  @param networkConfiguration (dict) the configuration of this network.
  """
  # Create and run network on this data.
  #   Input data comes from a CSV file (scalar values, labels). The
  #   RecordSensor region allows us to specify a file record stream as the
  #   input source via the dataSource attribute.

  # create the network encoders for sensor data. 
  dataSource = FileRecordStream(streamID=inputFile)
  _setScalarEncoderMinMax(networkConfiguration, dataSource)
  encoders = networkConfiguration[SENSOR_REGION_NAME]["encoders"]

  network = createNetwork(dataSource,
                          encoders,
                          networkConfiguration)

  # Need to init the network before it can run.
  network.initialize()
  return network



def _findNumberOfPartitions(networkConfiguration, numRecords):
  """
  Find the number of partitions for the input data based on a specific
  networkConfiguration. 
  
  @param networkConfiguration (dict) the configuration of this network.
  @param numRecords (int) Number of records of the input dataset.
  @return partitions (list of tuples) Region names and associated indices 
  partitioning the input dataset to indicate at which recordNumber it should 
  start learning. The remaining of the data (last partition) is used as a test 
  set. 
  """

  spEnabled = networkConfiguration[SP_REGION_NAME]["enabled"]
  tmEnabled = networkConfiguration[TM_REGION_NAME]["enabled"]
  upEnabled = networkConfiguration[UP_REGION_NAME]["enabled"]

  partitionNames = []
  if spEnabled and tmEnabled and upEnabled:
    numPartitions = 5
    partitionNames.extend([SP_REGION_NAME,
                           TM_REGION_NAME,
                           UP_REGION_NAME])
  elif spEnabled and tmEnabled:
    numPartitions = 4
    partitionNames.extend([SP_REGION_NAME,
                           TM_REGION_NAME])
  elif spEnabled:
    numPartitions = 3
    partitionNames.append(SP_REGION_NAME)
  else:
    numPartitions = 2  # only the classifier, so just test and train partitions
  partitionNames.append(CLASSIFIER_REGION_NAME)
  partitionNames.append(TEST_PARTITION_NAME)

  partitionIndices = [numRecords * i / numPartitions
                      for i in range(0, numPartitions)]

  return zip(partitionNames, partitionIndices)



def _setScalarEncoderMinMax(networkConfiguration, dataSource):
  """
  Set the min and max values.
  
  @param networkConfiguration (dict) the configuration of this network.
  @param dataSource (RecordStream) the input source
  """
  scalarEncoderParams = (networkConfiguration
                         [SENSOR_REGION_NAME]
                         ["encoders"]
                         [SCALAR_ENCODER_NAME])
  fieldName = scalarEncoderParams["fieldname"]
  scalarEncoderParams["minval"] = dataSource.getFieldMin(fieldName)
  scalarEncoderParams["maxval"] = dataSource.getFieldMax(fieldName) 

