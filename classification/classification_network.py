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

import numpy

try:
  import simplejson as json
except ImportError:
  import json

from generate_data import generateData
from generate_model_params import findMinMax
from nupic.data.file_record_stream import FileRecordStream
from nupic.encoders import CategoryEncoder, MultiEncoder, ScalarEncoder
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
  "maxval": None,  # needs to be initialized after file introspection
}

CATEGORY_ENCODER_PARAMS = {
  "name": 'label',
  "w": 21,
  "categoryList": range(NUMBER_OF_LABELS),
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

CLA_CLASSIFIER_PARAMS = {"steps": "1",
                         "implementation": "py"}


outFile = open(_OUT_FILE, "wb")



class ClassificationNetwork(object):
  """
  Factory to create a classification network:
    encoder -> SP -> TM -> (UP) -> classifier
  """

  def __init__(self):
    """
    """
  
    # Init region variables.
    self.network = Network()
    self.sensorRegion = None
    self.spatialPoolerRegion = None
    self.temporalMemoryRegion = None
    self.classifierRegion = None


  @staticmethod
  def _createEncoder(self):
    """
    TODO: generalize for other encoder types (?).
    """

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


  def _initSensorRegion(self, dataSource):
    """
    Initializes the sensor region with an encoder and data source.
    
    @param dataSource   (RecordStream)  Sensor region reads data from here.
    @return             ()              Region width to input to next region.
    """
    # Input data comes from a CSV file (scalar values, labels). The RecordSensor region
    # allows us to specify a file record stream as the input source via the
    # dataSource attribute.
    self.network.addRegion(
        "sensor", "py.RecordSensor", json.dumps({"verbosity": _VERBOSITY}))
    self.sensorRegion = self.network.regions["sensor"].getSelf()

    # Specify how RecordSensor encodes input values
    sensorRegion.encoder = self._createEncoder()

    # Specify the dataSource as a file record stream instance
    sensorRegion.dataSource = dataSource

    return sensorRegion.encoder.getWidth()


  def _initSpatialPoolerRegion(self, prevRegionWidth):
    """
    Create the spatial pooler region.
    
    @return             ()              Region width to input to next region.
    """
    SP_PARAMS["inputWidth"] = prevRegionWidth
    self.network.addRegion("SP", "py.SPRegion", json.dumps(SP_PARAMS))
    self.spatialPoolerRegion = self.network.regions["SP"]

    # Link the SP region to the sensor input
    network.link("sensor", "SP", "UniformLink", "")
    
    # Forward the sensor region sequence reset to the SP
    network.link("sensor", "SP", "UniformLink", "", srcOutput="resetOut", destInput="resetIn")
    
    # Make sure learning is ON
    self.spatialPoolerRegion.setParameter("learningMode", True)
    
    # Inference mode outputs the current inference (e.g. active columns). 
    # It's ok to always leave inference mode on - it's only there for some corner cases.
    self.spatialPoolerRegion.setParameter("inferenceMode", True)

    return SP_PARAMS["columnCount"]


  def _initTemporalMemoryRegion(self, prevRegionWidth):
    """
    Create the temporal memory region.
    
    @return             ()              Region width to input to next region.
    """
    # Make sure region widths fit
    if TM_PARAMS['columnCount'] != prevRegionWidth:
      raise ValueError("Region widths do not fit.")
    TM_PARAMS['inputWidth'] = TM_PARAMS['columnCount']
    
    # Create the TM region
    self.network.addRegion("TM", "py.TPRegion", json.dumps(TM_PARAMS))
    self.temporalMemoryRegion = self.network.regions["TM"]

    # Feed forward link from SP to TM
    self.network.link("SP", "TM", "UniformLink", "",
        srcOutput="bottomUpOut", destInput="bottomUpIn")
    
    # Feedback links (unnecessary ?)
    self.network.link("TM", "SP", "UniformLink", "",
        srcOutput="topDownOut", destInput="topDownIn")
    self.network.link("TM", "sensor", "UniformLink", "",
        srcOutput="topDownOut", destInput="temporalTopDownIn")

    # Forward the sensor region sequence reset to the TM
    self.network.link("sensor", "TM", "UniformLink", "",
        srcOutput="resetOut", destInput="resetIn")

    # Make sure learning is enabled (this is the default)
    self.temporalMemoryRegion.setParameter("learningMode", False)
    
    # Inference mode outputs the current inference (i.e. active cells).
    # Okay to always leave inference mode on - only there for some corner cases.
    self.temporalMemoryRegion.setParameter("inferenceMode", True)
    
    return TM_PARAMS['inputWidth']


  def _initClassifierRegion(self):
    """
    Create classifier region.
    
    """
    # Create the classifier region.
    self.network.addRegion(
      "classifier", "py.CLAClassifierRegion", json.dumps(CLA_CLASSIFIER_PARAMS))
    self.classifierRegion = self.network.regions["classifier"]

    # Feed the TM states to the classifier.
    network.link("TM", "classifier", "UniformLink", "",
        srcOutput = "bottomUpOut", destInput = "bottomUpIn")
    
    
    # Link the sensor to the classifier to send in category labels.
    # TODO: this link is actually useless right now because the CLAclassifier region compute() function doesn't work
    # and that we are feeding TM states & categories manually to the classifier via the customCompute() function. 
    self.network.link("sensor", "classifier", "UniformLink", "",
        srcOutput = "categoryOut", destInput = "categoryIn")

    # Disable learning for now (will be enabled in a later training phase).
    self.classifierRegion.setParameter("learningMode", False)

    # Inference mode outputs the current inference. We can always leave it on.
    self.classifierRegion.setParameter("inferenceMode", True)


  def createNetwork(self, dataSource):
    """
    Create and return the Network instance.

    @param dataSource   (RecordStream)  Sensor region reads data from here.
    @return             (Network)       Instance ready to run, with regions for
                                        the sensor, SP, TM, and classifier.
    """
    sensorRegionWidth = self._initSensorRegion(dataSource)
    
    SPRegionWidth = self._initSpatialPoolerRegion(sensorRegionWidth)

    self._initTemporalMemoryRegion(SPRegionWidth)
  
    self._initClassifierRegion()
  
    self.network.initialize()


  def runNetwork(self):
    """Run the network and write classification results output.

    @param writer: a csv.writer instance to write output to file.
    
    TODO: break this into smaller methods.
    """
    phaseInfo = "\n-> Training SP. Index=0. LEARNING: SP is ON | TM is OFF | Classifier is OFF \n"
    outFile.write(phaseInfo)
    print phaseInfo
    
    numCorrect = 0
    numTestRecords = 0
    for i in xrange(NUM_RECORDS):
      # Run the network for a single iteration
      self.network.run(1)
      
      # Various info from the network, useful for debugging & monitoring
      anomalyScore = self.temporalMemoryRegion.getOutputData("anomalyScore")
      spOut = self.spatialPoolerRegion.getOutputData("bottomUpOut")
      tpOut = self.temporalMemoryRegion.getOutputData("bottomUpOut")
      tmInstance = self.temporalMemoryRegion.getSelf()._tfdr
      predictiveCells = tmInstance.predictiveCells
      #if len(predictiveCells) >0:
      #  print len(predictiveCells)
      
      # NOTE: To be able to extract a category, one of the field of the the
      # dataset needs to have the flag C so it can be recognized as a category
      # by the encoder.
      actualValue = self.sensorRegion.getOutputData("categoryOut")[0]

      
      outFile.write("=> INDEX=%s |  actualValue=%s | anomalyScore=%s | tpOutNZ=%s\n" %(i, actualValue, anomalyScore, tpOut.nonzero()[0]))
      
      # SP has been trained. Now start training the TM too.
      if i == SP_TRAINING_SET_SIZE:
        self.temporalMemoryRegion.setParameter("learningMode", True)
        phaseInfo = "\n-> Training TM. Index=%s. LEARNING: SP is ON | TM is ON | Classifier is OFF \n" %i
        outFile.write(phaseInfo)
        print phaseInfo
        
      # Start training the classifier as well.
      elif i == TM_TRAINING_SET_SIZE:
        self.classifier.setParameter('learningMode', True)
        phaseInfo = "\n-> Training Classifier. Index=%s. LEARNING: SP is OFF | TM is ON | Classifier is ON \n" %i
        outFile.write(phaseInfo)
        print phaseInfo
      
      # Stop training.
      elif i == CLASSIFIER_TRAINING_SET_SIZE:
        self.spatialPoolerRegion.setParameter("learningMode", False)
        self.temporalMemoryRegion.setParameter("learningMode", False)
        self.classifier.setParameter("learningMode", False)
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
        #bucketIdx = self.sensorRegion.getBucketIndices(actualValue)[0]
        bucketIdx = actualValue
        
        classificationIn = {"bucketIdx": int(bucketIdx),
                            "actValue": int(actualValue)}
      
        # List the indices of active cells (non-zero pattern)
        activeCells = self.temporalMemoryRegion.getOutputData("bottomUpOut")
        patternNZ = activeCells.nonzero()[0]
        
        # Call classifier
        clResults = self.classifierRegion.getSelf().customCompute(
            recordNum=i, patternNZ=patternNZ, classification=classificationIn)
        
        inferredValue = clResults["actualValues"][clResults[int(CLA_CLASSIFIER_PARAMS["steps"])].argmax()]
        
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
  
    # Create and run network on this data.
    dataSource = FileRecordStream(streamID=inputFile)
    network = ClassificationNetwork()
    network.createNetwork(dataSource)
    network.runNetwork()
    
    print "results written to: %s" %_OUT_FILE
