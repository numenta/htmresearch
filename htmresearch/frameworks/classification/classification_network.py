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
of any of sensor, SP, TM, TP, and classifier regions.
"""
import copy
import simplejson as json
import logging
import numpy
import sys

from nupic.encoders import MultiEncoder
from nupic.engine import Network
from nupic.engine import pyRegions

from htmresearch.support.register_regions import registerResearchRegion

_PY_REGIONS = [r[1] for r in pyRegions]
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                    stream=sys.stdout)
TEST_PARTITION_NAME = "test"



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
  if not isinstance(encoders, dict):
    raise TypeError("Encoders specified in incorrect format.")

  encoder = MultiEncoder()
  encoder.addMultipleEncoders(encoders)

  return encoder



def _setScalarEncoderMinMax(networkConfig, dataSource):
  """
  Set the min and max values of a scalar encoder.

  @param networkConfig: (dict) configuration of the network.
  @param dataSource: (RecordStream) the input source
  """
  fieldName = getEncoderParam(networkConfig, "scalarEncoder", "fieldname")
  minval = dataSource.getFieldMin(fieldName)
  maxval = dataSource.getFieldMax(fieldName)
  networkConfig["sensorRegionConfig"]["encoders"]["scalarEncoder"]["minval"] = (
    minval)
  networkConfig["sensorRegionConfig"]["encoders"]["scalarEncoder"]["maxval"] = (
    maxval)



def _createSensorRegion(network, regionConfig, dataSource, encoder=None,
  moduleName=None):
  """
  Register a sensor region and initialize it the sensor region with an encoder
  and data source.

  @param network: (Network) The network instance.
  @param regionConfig: (dict) configuration of the sensor region
  @param dataSource: (RecordStream) Sensor region reads data from here.
  @param encoder: (Encoder) encoding object to use instead of specifying in
    networkConfig.
  @param moduleName: (str) location of the region class, only needed if 
    registering a region that is outside the expected "regions/" dir.
  @return sensorRegion: (PyRegion) Sensor region of the network.
  """
  regionType = regionConfig["regionType"]
  regionName = regionConfig["regionName"]
  regionParams = regionConfig["regionParams"]
  encoders = regionConfig["encoders"]
  if not encoders:
    encoders = encoder

  _addRegisteredRegion(network, regionConfig, moduleName)

  # getSelf() returns the actual region, instead of a region wrapper
  sensorRegion = network.regions[regionName].getSelf()

  if isinstance(encoders, dict):
    # Add encoder(s) from params dict.
    sensorRegion.encoder = _createEncoder(encoders)
  else:
    sensorRegion.encoder = encoders

  sensorRegion.dataSource = dataSource

  return sensorRegion



def _addRegisteredRegion(network, regionConfig, moduleName=None):
  """
  Add the region to the network, and register it if necessary. Return the
  added region.
  """
  regionName = regionConfig["regionName"]
  regionType = regionConfig["regionType"]
  regionParams = regionConfig["regionParams"]

  regionTypeName = regionType.split(".")[1]
  if regionTypeName not in _PY_REGIONS:
    registerResearchRegion(regionTypeName, moduleName)

  return network.addRegion(regionName, regionType, json.dumps(regionParams))


def _createRegion(network, regionConfig, moduleName=None):
  """
  Create the SP, TM, TP, or classifier region.

  @param network: (Network) The region will be a node in this network.
  @param regionConfig: (dict) The region configuration
  @return region: (PyRegion) region of the network.
  """
  region = _addRegisteredRegion(network, regionConfig, moduleName)

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
  network.link(sensorRegionName, currentRegionName, "UniformLink", "",
               srcOutput="sequenceIdOut", destInput="sequenceIdIn")


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


def configureNetwork(dataSource, networkParams, encoder=None):
  """
  Configure the network for various experiment values.

  @param dataSource: (RecordStream) CSV file record stream.
  @param networkParams: (dict) the configuration of this network.
  @param encoder: (Encoder) encoding object to use instead of specifying in
    networkConfig.
  """
  encoderDict = networkParams["sensorRegionConfig"].get("encoders")
  if not encoderDict and not encoder:
    raise ValueError("No encoder specified; cannot create sensor region.")

  # if the sensor region has a scalar encoder, then set the min and max values.
  scalarEncoder = encoderDict.get("scalarEncoder")
  if scalarEncoder:
    _setScalarEncoderMinMax(networkParams, dataSource)

  network = createNetwork(dataSource, networkParams, encoder)

  # Need to init the network before it can run.
  network.initialize()
  return network



def createNetwork(dataSource, networkConfig, encoder=None):
  """
  Create and initialize the network instance with regions for the sensor, SP,
  TM, and classifier. Before running, be sure to init w/ network.initialize().

  @param dataSource: (RecordStream) Sensor region reads data from here.
  @param networkConfig: (dict) the configuration of this network.
  @param encoder: (Encoder) encoding object to use instead of specifying in
    networkConfig.
  @return network: (Network) Sample network. E.g. Sensor -> SP -> TM -> Classif.
  """
  network = Network()

  # Create sensor region (always enabled)
  sensorRegionConfig = networkConfig["sensorRegionConfig"]
  sensorRegionName = sensorRegionConfig["regionName"]
  sensorRegion = _createSensorRegion(network,
                                     sensorRegionConfig,
                                     dataSource,
                                     encoder)

  # Keep track of the previous region name and width to validate and link the
  # input/output width of two consecutive regions.
  previousRegion = sensorRegionName
  previousRegionWidth = sensorRegion.encoder.getWidth()

  networkRegions = [r for r in networkConfig.keys() 
                    if networkConfig[r]["regionEnabled"]]

  if "spRegionConfig" in networkRegions:
    # create SP region, if enabled
    regionConfig = networkConfig["spRegionConfig"]
    regionName = regionConfig["regionName"]
    regionParams = regionConfig["regionParams"]
    regionParams["inputWidth"] = sensorRegion.encoder.getWidth()
    spRegion = _createRegion(network, regionConfig)
    _validateRegionWidths(previousRegionWidth, spRegion.getSelf().inputWidth)
    _linkRegions(network,
                 sensorRegionName,
                 previousRegion,
                 regionName)
    previousRegion = regionName
    previousRegionWidth = spRegion.getSelf().columnCount

  if "tmRegionConfig" in networkRegions:
    # create TM region, if enabled
    regionConfig = networkConfig["tmRegionConfig"]
    regionName = regionConfig["regionName"]
    regionParams = regionConfig["regionParams"]
    regionParams["inputWidth"] = regionParams["columnCount"]
    tmRegion = _createRegion(network, regionConfig)
    _validateRegionWidths(previousRegionWidth, tmRegion.getSelf().columnCount)
    _linkRegions(network,
                 sensorRegionName,
                 previousRegion,
                 regionName)
    previousRegion = regionName
    previousRegionWidth = (tmRegion.getSelf().columnCount *
                           tmRegion.getSelf().cellsPerColumn)

  if "tpRegionConfig" in networkRegions:
    # create TP region, if enabled
    regionConfig = networkConfig["tpRegionConfig"]
    regionName = regionConfig["regionName"]
    regionParams = regionConfig["regionParams"]
    regionParams["inputWidth"] = previousRegionWidth
    tpRegion = _createRegion(network, regionConfig,
      moduleName="htmresearch.regions.TemporalPoolerRegion")
    _validateRegionWidths(previousRegionWidth,
                          tpRegion.getSelf()._inputWidth)
    _linkRegions(network,
                 sensorRegionName,
                 previousRegion,
                 regionName)
    previousRegion = regionName

  # Create classifier region (always enabled)
  regionConfig = networkConfig["classifierRegionConfig"]
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

  # Link in sequenceId/partitionId if the appropriate input exists
  classifierSpec = network.regions[regionName].getSpec()
  if classifierSpec.inputs.contains('partitionIn'):
    network.link(sensorRegionName, regionName, "UniformLink", "",
                 srcOutput="sequenceIdOut", destInput="partitionIn")
  

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
  _LOGGER.info(phaseInfo)



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
  _LOGGER.info(phaseInfo)



def trainNetwork(network, networkConfig, networkPartitions, numRecords):
  """
  Train the network.

  @param network: (Network) a Network instance to run.
  @param networkConfig: (dict) params for network regions.
  @param networkPartitions: (list of tuples) Region names and index at which the
   region is to begin learning, including a test partition (the last entry).
  @param numRecords: (int) Number of records of the input dataset.
  """
  
  partitions = copy.deepcopy(networkPartitions) # preserve original partitions
  
  sensorRegion = network.regions[
    networkConfig["sensorRegionConfig"].get("regionName")]
  classifierRegion = network.regions[
    networkConfig["classifierRegionConfig"].get("regionName")]

  # Keep track of the regions that have been trained.
  trainedRegionNames = []
  numCorrect = 0
  numTestRecords = 0
  for recordNumber in xrange(numRecords):
    # Run the network for a single iteration.
    network.run(1)

    if recordNumber == partitions[0][1]:
      # end of the current partition
      partitionName = partitions[0][0]

      if partitionName == TEST_PARTITION_NAME:
        _stopLearning(network, trainedRegionNames, recordNumber)

      else:
        partitions.pop(0)
        trainedRegionNames.append(partitionName)
        _enableRegionLearning(network,
                              trainedRegionNames,
                              partitionName,
                              recordNumber)

    if recordNumber >= partitions[-1][1]:
      # evaluate the predictions on the test set
      # classifierConfig = networkConfig["classifierRegionConfig"]
      classifierRegion.setParameter("inferenceMode", True)

      actualValue = sensorRegion.getOutputData("categoryOut")[0]
      inferredValue = _getClassifierInference(classifierRegion)
      if actualValue == inferredValue:
        numCorrect += 1
      _LOGGER.debug("recordNum=%s, actualValue=%s, inferredValue=%s"
               % (recordNumber, actualValue, inferredValue))
      numTestRecords += 1

  classificationAccuracy = round(100.0 * numCorrect / numTestRecords, 2)

  results = ("RESULTS: accuracy=%s | %s correctly classified records out of %s "
             "test records \n" % (classificationAccuracy,
                                  numCorrect,
                                  numTestRecords))
  _LOGGER.info(results)

  return classificationAccuracy



def _getClassifierInference(classifierRegion):
  """Return output categories from the classifier region."""
  if classifierRegion.type == "py.KNNClassifierRegion":
    # The use of numpy.lexsort() here is to first sort by labelFreq, then
    # sort by random values; this breaks ties in a random manner.
    inferenceValues = classifierRegion.getOutputData("categoriesOut")
    randomValues = numpy.random.random(inferenceValues.size)
    return numpy.lexsort((randomValues, inferenceValues))[-1]

  elif classifierRegion.type == "py.CLAClassifierRegion":
    return classifierRegion.getOutputData("categoriesOut")[0]



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
