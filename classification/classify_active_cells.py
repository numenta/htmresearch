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

from nupic.data.file_record_stream import FileRecordStream
from nupic.encoders import ScalarEncoder, CategoryEncoder, MultiEncoder
from nupic.engine import Network
from settings import \
  NUMBER_OF_LABELS, \
  NUM_RECORDS, \
  CLASSIFIER_TRAINING_SET_SIZE, \
  TM_TRAINING_SET_SIZE, \
  SP_TRAINING_SET_SIZE, \
  SIGNAL_AMPLITUDE, \
  SIGNAL_MEAN, \
  SIGNAL_PERIOD, \
  WHITE_NOISE_AMPLITUDE_RANGES, \
  DATA_DIR
from generate_model_params import findMinMax
from generate_data import generateData

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

SP_PARAMS = {
    "spVerbosity": _VERBOSITY,
    "spatialImp": "cpp",
    "globalInhibition": 1,
    "columnCount": 2048,
    "inputWidth": 0,
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
    "inputWidth": 2048,
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

#KNN_CLASSIFIER_PARAMS = {
#  "k": 1,
#  'distThreshold': 0,
#  'maxCategoryCount': NUMBER_OF_LABELS,
#}

CLA_CLASSIFIER_PARAMS = { 'steps': "1",
                          'implementation': 'py'}


outFile = open(_OUT_FILE, 'wb')


def createScalarEncoder():
  scalarEncoder = ScalarEncoder(SCALAR_ENCODER_PARAMS['w'], 
                       SCALAR_ENCODER_PARAMS['minval'], 
                       SCALAR_ENCODER_PARAMS['maxval'], 
                       n=SCALAR_ENCODER_PARAMS['n'], 
                       name=SCALAR_ENCODER_PARAMS['name'])
  
  # NOTE: we don't want to encode the category along with the scalar input. 
  # The category will be fed separately to the classifier later, during the training phase.
  #categoryEncoder = CategoryEncoder(CATEGORY_ENCODER_PARAMS['w'],
  #                                  CATEGORY_ENCODER_PARAMS['categoryList'],
  #                                  name=CATEGORY_ENCODER_PARAMS['name'])
  encoder = MultiEncoder()
  encoder.addEncoder(SCALAR_ENCODER_PARAMS['name'], scalarEncoder)
  
  return encoder


def createNetwork(dataSource):
  """Create the Network instance.

  :param dataSource: a RecordStream instance to get data from
  :returns: a Network instance ready to run
  """
  
  network = Network()

  #----- SENSOR REGION -----#
  
  # Input data comes from a CSV file (scalar values, labels). The RecordSensor region
  # allows us to specify a file record stream as the input source via the
  # dataSource attribute.
  network.addRegion("sensor", "py.RecordSensor",
                    json.dumps({"verbosity": _VERBOSITY}))
  sensor = network.regions["sensor"].getSelf()

  # The RecordSensor needs to know how to encode the input values
  sensor.encoder = createScalarEncoder()

  # Specify the dataSource as a file record stream instance
  sensor.dataSource = dataSource

  # Region width
  prevRegionWidth = sensor.encoder.getWidth()


  #----- SPATIAL POOLER -----#

  # Create the spatial pooler region
  SP_PARAMS["inputWidth"] = prevRegionWidth
  network.addRegion("SP", "py.SPRegion", json.dumps(SP_PARAMS))

  # Link the SP region to the sensor input
  network.link("sensor", "SP", "UniformLink", "")
  
  # Forward the sensor region sequence reset to the SP
  network.link("sensor", "SP", "UniformLink", "", srcOutput="resetOut", destInput="resetIn")
  
  # Make sure learning is ON
  spatialPoolerRegion = network.regions["SP"]
  spatialPoolerRegion.setParameter("learningMode", True)
  
  # Inference mode outputs the current inference (e.g. active columns). 
  # It's ok to always leave inference mode on - it's only there for some corner cases.
  spatialPoolerRegion.setParameter('inferenceMode', True)

  # Region width
  prevRegionWidth = SP_PARAMS['columnCount']

  
  #----- TEMPORAL MEMORY -----#
  
  # Make sure region widths fit
  assert TM_PARAMS['columnCount'] == prevRegionWidth
  TM_PARAMS['inputWidth'] = TM_PARAMS['columnCount']
  
  # Create the TM region
  network.addRegion("TM", "py.TPRegion", json.dumps(TM_PARAMS))

  # Feed forward link from SP to TM
  network.link("SP", "TM", "UniformLink", "", srcOutput="bottomUpOut", destInput="bottomUpIn")
  
  # Feedback links (unnecessary ?)
  network.link("TM", "SP", "UniformLink", "", srcOutput="topDownOut", destInput="topDownIn")
  network.link("TM", "sensor", "UniformLink", "", srcOutput="topDownOut", destInput="temporalTopDownIn")

  # Forward the sensor region sequence reset to the TM
  network.link("sensor", "TM", "UniformLink", "", srcOutput="resetOut", destInput="resetIn")

  # Make sure learning is not enabled (we want to train the SP first)
  temporalMemoryRegion = network.regions["TM"]
  temporalMemoryRegion.setParameter("learningMode", False)
  
  # Inference mode outputs the current inference (e.g. active cells). 
  # It's ok to always leave inference mode on - it's only there for some corner cases.
  temporalMemoryRegion.setParameter('inferenceMode', True)
  
  # Region width
  prevRegionWidth = TM_PARAMS['inputWidth']


  #----- CLASSIFIER REGION -----#

  # create classifier region
  network.addRegion('classifier', 'py.CLAClassifierRegion', json.dumps(CLA_CLASSIFIER_PARAMS))

  # feed the TM states to the classifier
  network.link("TM", "classifier", "UniformLink", "", srcOutput = "bottomUpOut", destInput = "bottomUpIn")
  
  
  # create a link from the sensor to the classifier to send in category labels.
  # TODO: this link is actually useless right now because the CLAclassifier region compute() function doesn't work
  # and that we are feeding TM states & categories manually to the classifier via the customCompute() function. 
  network.link("sensor", "classifier", "UniformLink", "", srcOutput = "categoryOut", destInput = "categoryIn")

  # disable learning for now (will be enables in a later training phase)
  classifier =  network.regions["classifier"]
  classifier.setParameter('learningMode', False)

  # Inference mode outputs the current inference. We can always leave it on.
  classifier.setParameter('inferenceMode', True)

  

  #------ INITIALIZE -----#  
  
  # The network until you try to run it. Make sure it's initialized right away.
  network.initialize()

  return network


def runNetwork(network):
  """Run the network and write output to writer.

  :param network: a Network instance to run
  :param writer: a csv.writer instance to write output to
  """
  sensorRegion = network.regions["sensor"]
  spatialPoolerRegion = network.regions["SP"]
  temporalMemoryRegion = network.regions["TM"]
  classifier = network.regions["classifier"]

  phaseInfo =  "\n-> Training SP. Index=0. LEARNING: SP is ON | TM is OFF | Classifier is OFF \n"
  outFile.write(phaseInfo)
  print phaseInfo
  
  numCorrect = 0
  numTestRecords = 0
  for i in xrange(NUM_RECORDS):
    # Run the network for a single iteration
    network.run(1)
    
    # Various info from the network, useful for debugging & monitoring
    anomalyScore = temporalMemoryRegion.getOutputData("anomalyScore") 
    spOut = spatialPoolerRegion.getOutputData("bottomUpOut") 
    tpOut = temporalMemoryRegion.getOutputData("bottomUpOut") 
    tmInstance = temporalMemoryRegion.getSelf()._tfdr
    predictiveCells = tmInstance.predictiveCells
    #if len(predictiveCells) >0:
    #  print len(predictiveCells)
    
    # NOTE: To be able to extract a category, one of the field of the the dataset needs to have the flag C 
    # so that it can be recognized as a category by the encoder.
    actualValue = sensorRegion.getOutputData("categoryOut")[0]  

    
    outFile.write("=> INDEX=%s |  actualValue=%s | anomalyScore=%s | tpOutNZ=%s\n" %(i, actualValue, anomalyScore, tpOut.nonzero()[0]))
    
    # SP has been trained. Now start training the TM too.
    if i == SP_TRAINING_SET_SIZE:
      temporalMemoryRegion.setParameter("learningMode", True)
      phaseInfo = "\n-> Training TM. Index=%s. LEARNING: SP is ON | TM is ON | Classifier is OFF \n" %i
      outFile.write(phaseInfo)
      print phaseInfo
      
    # Start training the classifier as well.
    elif i == TM_TRAINING_SET_SIZE:
      classifier.setParameter('learningMode', True)
      phaseInfo = "\n-> Training Classifier. Index=%s. LEARNING: SP is OFF | TM is ON | Classifier is ON \n" %i
      outFile.write(phaseInfo)
      print phaseInfo
      
      
    elif i == CLASSIFIER_TRAINING_SET_SIZE:
      spatialPoolerRegion.setParameter("learningMode", False)
      temporalMemoryRegion.setParameter("learningMode", False)
      classifier.setParameter('learningMode', False)
      phaseInfo = "-> Test. Index=%s. LEARNING: SP is OFF | TM is OFF | Classifier is OFF \n" %i
      outFile.write(phaseInfo)
      print phaseInfo
      
    
    #-- OUTER LOOP TO FEED TM STATES TO CLA CLASS because compute() doesn't work for now --#
    if i >= TM_TRAINING_SET_SIZE:
      # Pass this information to the classifier's custom compute method
      # so that it can assign the current classification to possibly
      # multiple patterns from the past and current, and also provide
      # the expected classification for some time step(s) in the future.

      #bucketIdx = sensorRegion.getBucketIndices(actualValue)[0]  # doesn't work :-(
      bucketIdx = actualValue # TODO: hack for int categories. try to get the getBucketIndices() working instead
      
      classificationIn = {'bucketIdx': int(bucketIdx),
                          'actValue': int(actualValue)}
      
      activeCells = temporalMemoryRegion.getOutputData("bottomUpOut")
      patternNZ = activeCells.nonzero()[0] # list of indices of active cells (non-zero pattern)
      
      clResults = classifier.getSelf().customCompute(recordNum=i,
                                             patternNZ=patternNZ,
                                             classification=classificationIn)
      
      inferredValue = clResults['actualValues'][clResults[int(CLA_CLASSIFIER_PARAMS['steps'])].argmax()]
      
      outFile.write(" inferredValue=%s | classificationIn=%s | \n clResults=%s \n\n" %(inferredValue, classificationIn, clResults))
    
      # prediction evaluation of test set
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
    
    # experience params
    expParams = "\nRUNNING EXPERIMENT WITH PARAMS: numRecords=%s | noiseAmplitude=%s | signalAmplitude=%s | signalMean=%s | signalPeriod=%s \n\n"\
          %(NUM_RECORDS, noiseAmplitude, SIGNAL_AMPLITUDE, SIGNAL_MEAN, SIGNAL_PERIOD)
    
    outFile.write(expParams)
    print expParams    
    
    # generate the data
    generateData(whiteNoise=True, noise_amplitude=noiseAmplitude)
  
    # set min max for scalar encoder params
    inputFile = "%s/white_noise_%s.csv" % (DATA_DIR, noiseAmplitude)
    minval, maxval = findMinMax(inputFile)
    SCALAR_ENCODER_PARAMS["minval"] = minval
    SCALAR_ENCODER_PARAMS["maxval"] = maxval
  
    
    # create and run network on this data
    dataSource = FileRecordStream(streamID=inputFile)
    network = createNetwork(dataSource)
    runNetwork(network)
    
    print "results written to: %s" %_OUT_FILE
    

  