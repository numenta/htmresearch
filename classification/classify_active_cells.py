#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

import numpy
import os

from classification_network import createNetwork
from generate_data import generateData
from generate_model_params import findMinMax
from nupic.data.file_record_stream import FileRecordStream
from settings import (NUMBER_OF_LABELS,
                      NUM_RECORDS,
                      CLASSIFIER_TRAINING_SET_SIZE,
                      TM_TRAINING_SET_SIZE,
                      SP_TRAINING_SET_SIZE,
                      SIGNAL_AMPLITUDE,
                      SIGNAL_MEAN,
                      SIGNAL_PERIOD,
                      WHITE_NOISE_AMPLITUDE_RANGES,
                      DATA_DIR,
                      )


_OUT_FILE = "results/network.out"
_VERBOSITY = 0

SCALAR_ENCODER_PARAMS = {
    "name": "white_noise",
    "fieldname": "y",
    "type": "ScalarEncoder",
    "n": 256,
    "w": 21,
    "minval": None, # needs to be initialized after file introspection
    "maxval": None  # needs to be initialized after file introspection
}

CATEGORY_ENCODER_PARAMS = {
    "name": 'label',
    "w": 21,
    "categoryList": range(NUMBER_OF_LABELS)
}


outFile = open(_OUT_FILE, 'wb')


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
    spOut = spatialPoolerRegion.getOutputData("bottomUpOut")
    tpOut = temporalMemoryRegion.getOutputData("bottomUpOut")
    tmInstance = temporalMemoryRegion.getSelf()._tfdr
    
    # NOTE: To be able to extract a category, one of the field of the the
    # dataset needs to have the flag C so it can be recognized as a category
    # by the encoder.
    actualValue = sensorRegion.getOutputData("categoryOut")[0]

    outFile.write("=> INDEX=%s |  actualValue=%s | anomalyScore=%s \n" %(i, actualValue, anomalyScore))

    # SP has been trained. Now start training the TM too.
    if i == partitions[0]:
      temporalMemoryRegion.setParameter("learningMode", True)
      phaseInfo = "-> Training TM. Index=%s. LEARNING: SP is ON | TM is ON | Classifier is OFF \n" %i
      outFile.write(phaseInfo)
      print phaseInfo

    # Start training the classifier as well.
    elif i == partitions[1]:
      classifierRegion.setParameter("learningMode", True)
      phaseInfo = "-> Training Classifier. Index=%s. LEARNING: SP is OFF | TM is ON | Classifier is ON \n" %i
      outFile.write(phaseInfo)
      print phaseInfo

    # Stop training.
    elif i == partitions[2]:
      spatialPoolerRegion.setParameter("learningMode", False)
      temporalMemoryRegion.setParameter("learningMode", False)
      classifierRegion.setParameter("learningMode", False)
      phaseInfo = "-> Test. Index=%s. LEARNING: SP is OFF | TM is OFF | Classifier is OFF \n" %i
      outFile.write(phaseInfo)
      print phaseInfo

    #--- BEGIN PREDICTING TEST SET --#
    if i >= partitions[1]:
      # Pass this information to the classifier's custom compute method
      # so that it can assign the current classification to possibly
      # multiple patterns from the past and current, and also provide
      # the expected classification for some time step(s) in the future.

      # TODO: this is a hack for int categories... try to get the
      # getBucketIndices() method working instead.
      #bucketIdx = net.sensorRegion.getBucketIndices(actualValue)[0]
      bucketIdx = actualValue

      classificationIn = {"bucketIdx": int(bucketIdx),
                          "actValue": int(actualValue)}

      # List the indices of active cells (non-zero pattern)
      activeCells = temporalMemoryRegion.getOutputData("bottomUpOut")
      patternNZ = activeCells.nonzero()[0]

      # classify predicted active cells
      # TODO: ideally we would want to get the tmInstance.getOutput("predictedActiveCells")
      # TODO[continued] but it is not implemented in the tm_py, so we use this work around for now.
      # predictiveCells = tmInstance.getOutput("predictedActiveCells")
      predictiveCells = tmInstance.predictiveCells
      predictedActiveCells = numpy.intersect1d(activeCells, predictiveCells)
      if len(predictiveCells) >0: #TODO: this line to be removed. for debugging purposes.
        #print "predictiveActiveCells: %s" %predictedActiveCells
        pass
      
      # Call classifier
      clResults = classifierRegion.getSelf().customCompute(
          recordNum=i, patternNZ=patternNZ, classification=classificationIn)

      inferredValue = clResults["actualValues"][clResults[int(classifierRegion.getParameter("steps"))].argmax()]

      outFile.write(" inferredValue=%s | classificationIn=%s | \n clResults=%s \n" %(inferredValue, classificationIn, clResults))

      # Evaluate the predictions in the test set.
      if i > partitions[2]:
        if actualValue == inferredValue:
          numCorrect += 1
        else:  # TODO: remove. debugging.
          #print " INCORRECT_PREDICTION: index=%s | actualValue = %s | inferredValue = %s | \n clResults = %s \n\n" % (i, actualValue, inferredValue, clResults)
          pass

        numTestRecords += 1

  predictionAccuracy =  100.0 * numCorrect / numTestRecords

  results = "RESULTS: accuracy=%s | %s correctly predicted records out of %s test records \n" %(predictionAccuracy, numCorrect, numTestRecords)
  outFile.write(results)
  print results

  return numCorrect, numTestRecords, predictionAccuracy


def _setupScalarEncoder(minval, maxval):
  # Set min and max for scalar encoder params.
  SCALAR_ENCODER_PARAMS["minval"] = minval
  SCALAR_ENCODER_PARAMS["maxval"] = maxval


if __name__ == "__main__":

  for noiseAmplitude in WHITE_NOISE_AMPLITUDE_RANGES:
    
    expParams = "RUNNING EXPERIMENT WITH PARAMS: numRecords=%s | noiseAmplitude=%s | signalAmplitude=%s | signalMean=%s | signalPeriod=%s \n"\
          %(NUM_RECORDS, noiseAmplitude, SIGNAL_AMPLITUDE, SIGNAL_MEAN, SIGNAL_PERIOD)
    outFile.write(expParams)
    print expParams

    # Generate the data, and get the min/max values
    generateData(whiteNoise=True, noise_amplitude=noiseAmplitude)
    inputFile = os.path.join(DATA_DIR, "white_noise_%s.csv" % noiseAmplitude)
    minval, maxval = findMinMax(inputFile)
    # Partition records into training sets for SP, TM, and classifier
    partitions = [
      SP_TRAINING_SET_SIZE, TM_TRAINING_SET_SIZE, CLASSIFIER_TRAINING_SET_SIZE]

    _setupScalarEncoder(minval, maxval)

    # Create and run network on this data.
    #   Input data comes from a CSV file (scalar values, labels). The
    #   RecordSensor region allows us to specify a file record stream as the
    #   input source via the dataSource attribute.
    dataSource = FileRecordStream(streamID=inputFile)
    encoders = {"white_noise": SCALAR_ENCODER_PARAMS}
    network = createNetwork((dataSource, "py.RecordSensor", encoders))

    # Need to init the network before it can run.
    network.initialize()
    run(network, NUM_RECORDS, partitions, outFile)

    print "Results written to: %s" %_OUT_FILE

  outFile.close()
