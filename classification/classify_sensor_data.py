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

import os

from classification_network import createNetwork
from generate_sensor_data import generateData
from generate_model_params import findMinMax
from nupic.data.file_record_stream import FileRecordStream
from settings import (NUM_CATEGORIES,
                      NUM_RECORDS,
                      PARTITIONS,
                      SIGNAL_AMPLITUDE,
                      SIGNAL_MEAN,
                      SIGNAL_PERIOD,
                      WHITE_NOISE_AMPLITUDE_RANGES,
                      DATA_DIR,
                      )

_VERBOSITY = 2

_SCALAR_ENCODER_PARAMS = {
  "name": "white_noise",
  "fieldname": "y",
  "type": "ScalarEncoder",
  "n": 256,
  "w": 21,
  "minval": None,  # needs to be initialized after file introspection
  "maxval": None  # needs to be initialized after file introspection
}

_CATEGORY_ENCODER_PARAMS = {
  "name": 'white_noise',
  "fieldname": 'label',
  "type": "CategoryEncoder",
  "w": 21,
  "categoryList": range(NUM_CATEGORIES)
}

_SEQUENCE_CLASSIFIER_PARAMS = {'numCategories': NUM_CATEGORIES, "clVerbosity": _VERBOSITY}

_KNN_CLASSIFIER_PARAMS = {
  "k": 1,
  'distThreshold': 0,
  'maxCategoryCount': NUM_CATEGORIES,
}

_OUT_FILE = open("results/network.out", 'wb')


def run(net, numRecords, partitions, outFile):
  """
  Run the network and write classification results output.
  @param net: a Network instance to run.
  @param outFile: a writer instance to write output to file.
  @param partitions: list of indices at which training begins for the SP, TM,
      and classifier regions, respectively, e.g. [100, 200, 300].

  TODO: break this into smaller methods.
  """
  sensorRegion = net.regions["sensor"]
  spatialPoolerRegion = net.regions["SP"]
  temporalMemoryRegion = net.regions["TM"]
  classifierRegion = net.regions["classifier"]

  phaseInfo = "-> Training SP. Index=0. LEARNING: SP is ON | TM is OFF | Classifier is OFF \n"
  outFile.write(phaseInfo)
  print phaseInfo

  numCorrect = 0
  numTestRecords = 0
  for i in xrange(numRecords):
    # Run the network for a single iteration
    net.run(1)

    # Various info from the network, useful for debugging & monitoring
    anomalyScore = temporalMemoryRegion.getOutputData("anomalyScore")

    # NOTE: To be able to extract a category, one of the field of the the
    # dataset needs to have the flag C so it can be recognized as a category
    # by the FileRecordStream instance.
    actualValue = sensorRegion.getOutputData("categoryOut")[0]

    outFile.write("=> INDEX=%s |  actualValue=%s | anomalyScore=%s \n" % (i, actualValue, anomalyScore))

    # SP has been trained. Now start training the TM too.
    if i == partitions[0]:
      temporalMemoryRegion.setParameter("learningMode", True)
      phaseInfo = "-> Training TM. Index=%s. LEARNING: SP is ON | TM is ON | Classifier is OFF \n" % i
      outFile.write(phaseInfo)
      print phaseInfo

    # Start training the classifier as well.
    elif i == partitions[1]:
      classifierRegion.setParameter("learningMode", True)
      phaseInfo = "-> Training Classifier. Index=%s. LEARNING: SP is OFF | TM is ON | Classifier is ON \n" % i
      outFile.write(phaseInfo)
      print phaseInfo

    # Stop training.
    elif i == partitions[2]:
      spatialPoolerRegion.setParameter("learningMode", False)
      temporalMemoryRegion.setParameter("learningMode", False)
      classifierRegion.setParameter("learningMode", False)
      phaseInfo = "-> Test. Index=%s. LEARNING: SP is OFF | TM is OFF | Classifier is OFF \n" % i
      outFile.write(phaseInfo)
      print phaseInfo

    # --- BEGIN PREDICTING TEST SET --#
    if i >= partitions[2]:

      # get various outputs
      inferredValue = classifierRegion.getOutputData("categoryOut")[0]

      if actualValue == inferredValue:
        numCorrect += 1
      else:
        print "[DEBUG] INCORRECT_PREDICTION: index=%s | actualValue = %s | inferredValue = %s \n" \
              % (i, actualValue, inferredValue)

      numTestRecords += 1

  predictionAccuracy = 100.0 * numCorrect / numTestRecords

  results = "RESULTS: accuracy=%s | %s correctly predicted records out of %s test records \n" % (
    predictionAccuracy, numCorrect, numTestRecords)
  outFile.write(results)
  print results

  return numCorrect, numTestRecords, predictionAccuracy


def _setupScalarEncoder(minval, maxval):
  # Set min and max for scalar encoder params.
  _SCALAR_ENCODER_PARAMS["minval"] = minval
  _SCALAR_ENCODER_PARAMS["maxval"] = maxval


if __name__ == "__main__":

  for noiseAmplitude in WHITE_NOISE_AMPLITUDE_RANGES:
    expParams = "RUNNING EXPERIMENT WITH PARAMS: " + \
                "numRecords=%s | noiseAmplitude=%s | signalAmplitude=%s | signalMean=%s | signalPeriod=%s \n" \
                % (NUM_RECORDS, noiseAmplitude, SIGNAL_AMPLITUDE, SIGNAL_MEAN, SIGNAL_PERIOD)
    _OUT_FILE.write(expParams)
    print expParams

    # Generate the data, and get the min/max values
    generateData(noise_amplitude=noiseAmplitude)
    inputFile = os.path.join(DATA_DIR, "white_noise_%s.csv" % noiseAmplitude)
    minval, maxval = findMinMax(inputFile)
    # Partition records into training sets for SP, TM, and classifier
    _setupScalarEncoder(minval, maxval)

    # Create and run network on this data.
    #   Input data comes from a CSV file (scalar values, labels). The
    #   RecordSensor region allows us to specify a file record stream as the
    #   input source via the dataSource attribute.
    dataSource = FileRecordStream(streamID=inputFile)
    encoders = encoders = {"y": _SCALAR_ENCODER_PARAMS, "label": _CATEGORY_ENCODER_PARAMS}
    network = createNetwork(
      (dataSource, "py.RecordSensor", encoders, NUM_CATEGORIES, "py.SequenceClassifierRegion",
       _SEQUENCE_CLASSIFIER_PARAMS))

    # Need to init the network before it can run.
    network.initialize()
    run(network, NUM_RECORDS, PARTITIONS, _OUT_FILE)

  _OUT_FILE.close()
