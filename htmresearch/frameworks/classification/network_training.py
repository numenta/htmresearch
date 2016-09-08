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

import copy
import logging
import numpy
import sys

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG,
                    stream=sys.stdout)

TEST_PARTITION_NAME = "test"



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



def trainNetwork(network, networkConfig, networkPartitions, numRecords,
                 verbosity=0):
  """
  Train the network.

  @param network: (Network) a Network instance to run.
  @param networkConfig: (dict) params for network regions.
  @param networkPartitions: (list of tuples) Region names and index at which the
   region is to begin learning, including a test partition (the last entry).
  @param numRecords: (int) Number of records of the input dataset.
  @param verbosity: (0 or 1) How verbose the log is. (0 is less verbose)
  """

  partitions = copy.deepcopy(networkPartitions)  # preserve original partitions

  sensorRegion = network.regions[
    networkConfig["sensorRegionConfig"].get("regionName")]
  classifierRegion = network.regions[
    networkConfig["classifierRegionConfig"].get("regionName")]
  if networkConfig['tpRegionConfig'].get('regionEnabled'):
    tpRegion = network.regions[
      networkConfig['tpRegionConfig'].get('regionName')]
  else:
    tpRegion = None

  trackTMmetrics = False
  # track TM metrics if monitored_tm_py implementation is being used
  if networkConfig["tmRegionConfig"].get("regionEnabled"):
    tmRegion = network.regions[
      networkConfig["tmRegionConfig"].get("regionName")]

    if tmRegion.getParameter("temporalImp") == "monitored_tm_py":
      trackTMmetrics = True
      tm = tmRegion.getSelf().getAlgorithmInstance()
  else:
    tmRegion = None
    tm = None

  # Keep track of the regions that have been trained.
  trainedRegionNames = []

  # Number of correctly classified records
  numCorrectlyClassifiedRecords = 0
  numCorrectlyClassifiedTestRecords = 0
  numPoints = 0
  numTestPoints = 0

  # Network traces
  sensorValueTrace = []
  classificationAccuracyTrace = []
  testClassificationAccuracyTrace = []
  categoryTrace = []
  tpActiveCellsTrace = []
  tmActiveCellsTrace = []
  tmPredictedActiveCellsTrace = []
  predictedCategoryTrace = []
  for recordNumber in xrange(numRecords):

    # Run the network for a single iteration.
    network.run(1)

    if tpRegion:
      tpActiveCells = tpRegion.getOutputData("mostActiveCells")
      tpActiveCells = tpActiveCells.nonzero()[0]
      tpActiveCellsTrace.append(tpActiveCells)

    if tmRegion:
      tmPredictedActiveCells = tmRegion.getOutputData("predictedActiveCells")
      tmPredictedActiveCells = tmPredictedActiveCells.nonzero()[0]
      tmActiveCells = tmRegion.getOutputData("activeCells")
      tmActiveCells = tmActiveCells.nonzero()[0]
      tmActiveCellsTrace.append(tmActiveCells)
      tmPredictedActiveCellsTrace.append(tmPredictedActiveCells)

    sensorValueTrace.append(sensorRegion.getOutputData("sourceOut")[0])
    inferredCategory = _getClassifierInference(classifierRegion)
    predictedCategoryTrace.append(inferredCategory)
    actualCategory = sensorRegion.getOutputData("categoryOut")[0]
    categoryTrace.append(actualCategory)
    if actualCategory > 0:  # don't evaluate the noise (category = 0)
      numPoints += 1
      if actualCategory == inferredCategory:
        numCorrectlyClassifiedRecords += 1
      else:
        if verbosity > 0:
          _LOGGER.debug("recordNum=%s, actualCategory=%s, inferredCategory=%s"
                        % (recordNumber, actualCategory, inferredCategory))
      clfAccuracy = round(100.0 * numCorrectlyClassifiedRecords / numPoints, 2)
      classificationAccuracyTrace.append(clfAccuracy)
    else:
      classificationAccuracyTrace.append(None)

    if trackTMmetrics:

      activeColsTrace = tm.mmGetTraceActiveColumns()
      predictedActiveColsTrace = tm.mmGetTracePredictedActiveColumns()

      if tmRegion.getParameter("learningMode") and recordNumber % 100 == 0:
        (avgPredictedActiveCols,
         avgPredictedInactiveCols,
         avgUnpredictedActiveCols) = _inspectTMPredictionQuality(
          tm, numRecordsToInspect=100)
        tmStats = ("recordNumber %4d # predicted -> active cols=%4.1f | "
                   "# predicted -> inactive cols=%4.1f | "
                   "# unpredicted -> active cols=%4.1f " % (
                     recordNumber,
                     avgPredictedActiveCols,
                     avgPredictedInactiveCols,
                     avgUnpredictedActiveCols
                   ))
        _LOGGER.info(tmStats)

    if recordNumber == partitions[0][1]:
      # end of the current partition
      partitionName = partitions[0][0]

      # stop learning for all regions
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
      if actualCategory > 0:  # don't evaluate the noise (category = 0)
        numTestPoints += 1
        if actualCategory == inferredCategory:
          numCorrectlyClassifiedTestRecords += 1
        testClassificationAccuracy = round(
          100.0 * numCorrectlyClassifiedTestRecords / numTestPoints, 2)
        testClassificationAccuracyTrace.append(testClassificationAccuracy)
      else:
        testClassificationAccuracyTrace.append(None)
  _LOGGER.info("RESULTS: accuracy=%s | "
               "%s correctly classified records out of %s test records \n" %
               (testClassificationAccuracyTrace[-1],
                numCorrectlyClassifiedTestRecords,
                numTestPoints))

  traces = {
    'predictedCategoryTrace': predictedCategoryTrace,
    'classificationAccuracyTrace': classificationAccuracyTrace,
    'testClassificationAccuracyTrace': testClassificationAccuracyTrace,
    'sensorValueTrace': sensorValueTrace,
    'categoryTrace': categoryTrace,
    'tmActiveCellsTrace': tmActiveCellsTrace,
    'tmPredictiveActiveCellsTrace': tmPredictedActiveCellsTrace,
    'tpActiveCellsTrace': tpActiveCellsTrace
  }

  if trackTMmetrics:
    traces['activeColsTrace'] = activeColsTrace.data
    traces['predictedActiveColsTrace'] = predictedActiveColsTrace.data

  return traces



def _getClassifierInference(classifierRegion):
  """Return output categories from the classifier region."""
  if classifierRegion.type == "py.KNNClassifierRegion":
    # The use of numpy.lexsort() here is to first sort by labelFreq, then
    # sort by random values; this breaks ties in a random manner.
    inferenceValues = classifierRegion.getOutputData("categoriesOut")
    randomValues = numpy.random.random(inferenceValues.size)
    return numpy.lexsort((randomValues, inferenceValues))[-1]
  else:
    return classifierRegion.getOutputData("categoriesOut")[0]



def _inspectTMPredictionQuality(tm, numRecordsToInspect):
  """ Inspect prediction quality of TM over the most recent
  numRecordsToInspect records """
  # correct predictions: predicted -> active columns
  predictedActiveCols = tm.mmGetTracePredictedActiveColumns()
  numPredictedActiveCols = predictedActiveCols.makeCountsTrace().data

  # false/extra predictions: predicted -> inactive column
  predictedInactiveCols = tm.mmGetTracePredictedInactiveColumns()
  numPredictedInactiveCols = predictedInactiveCols.makeCountsTrace().data

  # unpredicted inputs: unpredicted -> active
  unpredictedActiveCols = tm.mmGetTraceUnpredictedActiveColumns()
  numUnpredictedActiveCols = unpredictedActiveCols.makeCountsTrace().data

  avgPredictedActiveCols = numpy.mean(
    numPredictedActiveCols[-numRecordsToInspect:])
  avgPredictedInactiveCols = numpy.mean(
    numPredictedInactiveCols[-numRecordsToInspect:])
  avgUnpredictedActiveCols = numpy.mean(
    numUnpredictedActiveCols[-numRecordsToInspect:])

  return (avgPredictedActiveCols,
          avgPredictedInactiveCols,
          avgUnpredictedActiveCols)
