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

import json
import numpy

from classification_network import ClassificationNetwork
from generate_data import generateData
from generate_model_params import findMinMax

from nupic.data.file_record_stream import FileRecordStream
from nupic.encoders import ScalarEncoder, CategoryEncoder, MultiEncoder
from nupic.engine import Network
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


_OUT_FILE = 'results/network.out'
_VERBOSITY = 0

SCALAR_ENCODER_PARAMS = {
    "name": 'y',
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


def createScalarEncoder():
  scalarEncoder = ScalarEncoder(SCALAR_ENCODER_PARAMS['w'], 
                       SCALAR_ENCODER_PARAMS['minval'], 
                       SCALAR_ENCODER_PARAMS['maxval'], 
                       n=SCALAR_ENCODER_PARAMS['n'], 
                       name=SCALAR_ENCODER_PARAMS['name'])
  
  # NOTE: we don't want to encode the category along with the scalar input. 
  # The category will be fed separately to the classifier later, during the
  # training phase.
  #categoryEncoder = CategoryEncoder(CATEGORY_ENCODER_PARAMS['w'],
  #                                  CATEGORY_ENCODER_PARAMS['categoryList'],
  #                                  name=CATEGORY_ENCODER_PARAMS['name'])
  encoder = MultiEncoder()
  encoder.addEncoder(SCALAR_ENCODER_PARAMS['name'], scalarEncoder)
  
  return encoder


def run(net, outFile):
  """
  Run the network and write classification results output.
  
  @param net: a Network instance to run
  @param outFile: a writer instance to write output to file.
  
  TODO: break this into smaller methods.
  """

  phaseInfo = "\n-> Training SP. Index=0. LEARNING: SP is ON | TM is OFF | Classifier is OFF \n"
  outFile.write(phaseInfo)
  print phaseInfo
  
  numCorrect = 0
  numTestRecords = 0
  for i in xrange(NUM_RECORDS):
    # Run the network for a single iteration
    net.network.run(1)
    
    # Various info from the network, useful for debugging & monitoring
    anomalyScore = net.temporalMemoryRegion.getOutputData("anomalyScore")
    spOut = net.spatialPoolerRegion.getOutputData("bottomUpOut")
    tpOut = net.temporalMemoryRegion.getOutputData("bottomUpOut")
    tmInstance = net.temporalMemoryRegion.getSelf()._tfdr
    predictiveCells = tmInstance.predictiveCells
    #if len(predictiveCells) >0:
    #  print len(predictiveCells)

    # NOTE: To be able to extract a category, one of the field of the the
    # dataset needs to have the flag C so it can be recognized as a category
    # by the encoder.
    actualValue = net.sensorRegion.getOutputData("categoryOut")[0]

    
    outFile.write("=> INDEX=%s |  actualValue=%s | anomalyScore=%s | tpOutNZ=%s\n" %(i, actualValue, anomalyScore, tpOut.nonzero()[0]))
    
    # SP has been trained. Now start training the TM too.
    if i == SP_TRAINING_SET_SIZE:
      net.temporalMemoryRegion.setParameter("learningMode", True)
      phaseInfo = "\n-> Training TM. Index=%s. LEARNING: SP is ON | TM is ON | Classifier is OFF \n" %i
      outFile.write(phaseInfo)
      print phaseInfo
      
    # Start training the classifier as well.
    elif i == TM_TRAINING_SET_SIZE:
      net.classifierRegion.setParameter("learningMode", True)
      phaseInfo = "\n-> Training Classifier. Index=%s. LEARNING: SP is OFF | TM is ON | Classifier is ON \n" %i
      outFile.write(phaseInfo)
      print phaseInfo
    
    # Stop training.
    elif i == CLASSIFIER_TRAINING_SET_SIZE:
      net.spatialPoolerRegion.setParameter("learningMode", False)
      net.temporalMemoryRegion.setParameter("learningMode", False)
      net.classifierRegion.setParameter("learningMode", False)
      phaseInfo = "-> Test. Index=%s. LEARNING: SP is OFF | TM is OFF | Classifier is OFF \n" %i
      outFile.write(phaseInfo)
      print phaseInfo
      
    
    #--- BEGIN PREDICTING TEST SET --#
    if i >= TM_TRAINING_SET_SIZE:
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
      activeCells = net.temporalMemoryRegion.getOutputData("bottomUpOut")
      patternNZ = activeCells.nonzero()[0]
      
      # Call classifier
      clResults = net.classifierRegion.getSelf().customCompute(
          recordNum=i, patternNZ=patternNZ, classification=classificationIn)
      
      inferredValue = clResults["actualValues"][clResults[int(net.classifierParams["steps"])].argmax()]
      
      outFile.write(" inferredValue=%s | classificationIn=%s | \n clResults=%s \n\n" %(inferredValue, classificationIn, clResults))
    
      # Evaluate the predictions in the test set.
      if i > CLASSIFIER_TRAINING_SET_SIZE:

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



if __name__ == "__main__":
  
  for noiseAmplitude in WHITE_NOISE_AMPLITUDE_RANGES:
    
    expParams = "\nRUNNING EXPERIMENT WITH PARAMS: numRecords=%s | noiseAmplitude=%s | signalAmplitude=%s | signalMean=%s | signalPeriod=%s \n\n"\
          %(NUM_RECORDS, noiseAmplitude, SIGNAL_AMPLITUDE, SIGNAL_MEAN, SIGNAL_PERIOD)
    outFile.write(expParams)
    print expParams    
    
    generateData(whiteNoise=True, noise_amplitude=noiseAmplitude)
  
    # Set min and max for scalar encoder params.
    inputFile = "%s/white_noise_%s.csv" % (DATA_DIR, noiseAmplitude)
    minval, maxval = findMinMax(inputFile)
    SCALAR_ENCODER_PARAMS["minval"] = minval
    SCALAR_ENCODER_PARAMS["maxval"] = maxval

    # Setup scalar encoder; not category b/c fed seperately later.
    scalarEncoder = ScalarEncoder(SCALAR_ENCODER_PARAMS["w"],
                                  SCALAR_ENCODER_PARAMS["minval"],
                                  SCALAR_ENCODER_PARAMS["maxval"],
                                  n=SCALAR_ENCODER_PARAMS["n"],
                                  name=SCALAR_ENCODER_PARAMS["name"])

    # Create and run network on this data.
    dataSource = FileRecordStream(streamID=inputFile)
    network = ClassificationNetwork()
    network.createEncoder(SCALAR_ENCODER_PARAMS["name"], scalarEncoder)
    network.createNetwork(dataSource)

    run(network, outFile)
    
    print "results written to: %s" %_OUT_FILE
