#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
#5
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import time
import json
import numpy

from nupic.bindings.math import GetNTAReal
from nupic.engine import Network
from nupic.algorithms import fdrutilities

"""
Setups a simple Network and runs it on the MNIST dataset. Assumes you have
the mnist data in subdirectories called mnist/training and mnist/testing.

The current network, using a random SP, gets about 95.5% correct on the test set
if trained with the full training set.  By no means is this a complete  HTM
vision system. There is no sequence memory or temporal pooling  here. In
addition there is a single SP whose receptive field is the entire image.  A
system with local receptive fields, temporal pooling, and some minimal
hierarchy would be required to get really good invariance and recognition rates.

Best SP params so far:

------------CPP SpatialPooler Parameters ------------------
numInputs                   = 1024
numColumns                  = 4096
numActiveColumnsPerInhArea  = 240
potentialPct                = 0.9
globalInhibition            = 1
localAreaDensity            = -1
stimulusThreshold           = 0
synPermActiveInc            = 0
synPermInactiveDec          = 0
synPermConnected            = 0.2
minPctOverlapDutyCycles     = 0.001
dutyCyclePeriod             = 1000
boostStrength               = 0
wrapAround                  = 1
CPP SP seed                 = 1956

"""


DEFAULT_IMAGESENSOR_PARAMS ={
  'width': 32,
  'height': 32,
  'mode': 'bw',
  'background': 0,
  'explorer': 'RandomFlash'
}

DEFAULT_SP_PARAMS = {
  'columnCount': 4096,
  'spatialImp': 'cpp',
  'inputWidth': 1024,
  'spVerbosity': 1,
  'synPermConnected': 0.2,
  'synPermActiveInc': 0.0,
  'synPermInactiveDec': 0.0,
  'seed': 1956,
  'numActiveColumnsPerInhArea': 240,
  'globalInhibition': 1,
  'potentialPct': 0.9,
  'boostStrength': 0.0
}

DEFAULT_CLASSIFIER_PARAMS = {
  'distThreshold': 0.000001,
  'maxCategoryCount': 10,
  #'distanceMethod': 'rawOverlap',  # Default is Euclidean distance
}

def createNetwork():
  """
  Set up the following simple network and return it:

    ImageSensorRegion -> SP -> KNNClassifier Region

  """
  net = Network()

  # Add the three regions
  net.addRegion("sensor", "py.ImageSensor",
                json.dumps(DEFAULT_IMAGESENSOR_PARAMS))
  net.addRegion("SP", "py.SPRegion", json.dumps(DEFAULT_SP_PARAMS))
  net.addRegion("classifier","py.KNNClassifierRegion",
                json.dumps(DEFAULT_CLASSIFIER_PARAMS))

  # Link up the regions. Note that we need to create a link from the sensor
  # to the classifier to send in the category labels.
  net.link("sensor", "SP", "UniformLink", "",
           srcOutput = "dataOut", destInput = "bottomUpIn")
  net.link("SP", "classifier", "UniformLink", "",
           srcOutput = "bottomUpOut", destInput = "bottomUpIn")
  net.link("sensor", "classifier", "UniformLink", "",
           srcOutput = "categoryOut", destInput = "categoryIn")

  return net


def trainNetwork(net, networkFile="mnist_net.nta"):
  # Some stuff we will need later
  sensor = net.regions['sensor']
  sp = net.regions["SP"]
  pysp = sp.getSelf()
  classifier = net.regions['classifier']
  dutyCycles = numpy.zeros(DEFAULT_SP_PARAMS['columnCount'], dtype=GetNTAReal())

  print "============= Loading training images ================="
  t1 = time.time()
  sensor.executeCommand(["loadMultipleImages", "mnist/training"])
  numTrainingImages = sensor.getParameter('numImages')
  start = time.time()
  print "Load time for training images:",start-t1
  print "Number of training images",numTrainingImages

  # First train just the SP
  print "============= SP training ================="
  classifier.setParameter('inferenceMode', 0)
  classifier.setParameter('learningMode', 0)
  sp.setParameter('learningMode', 0)
  sp.setParameter('inferenceMode', 1)
  nTrainingIterations = numTrainingImages
  for i in range(nTrainingIterations):
    net.run(1)
    dutyCycles += pysp._spatialPoolerOutput
    if i%(nTrainingIterations/100)== 0:
      print "Iteration",i,"Category:",sensor.getOutputData('categoryOut')

  # Now train just the classifier sequentially on all training images
  print "============= Classifier training ================="
  sensor.setParameter('explorer','Flash')
  classifier.setParameter('inferenceMode', 0)
  classifier.setParameter('learningMode', 1)
  sp.setParameter('learningMode', 0)
  sp.setParameter('inferenceMode', 1)
  for i in range(numTrainingImages):
    net.run(1)
    if i%(numTrainingImages/100)== 0:
      print "Iteration",i,"Category:",sensor.getOutputData('categoryOut')

  # Save the trained network
  net.save(networkFile)

  # Print various statistics
  print "============= Training statistics ================="
  print "Training time:",time.time() - start
  tenPct= nTrainingIterations/10
  print "My duty cycles:",fdrutilities.numpyStr(dutyCycles, format="%g")
  print "Number of nonzero duty cycles:",len(dutyCycles.nonzero()[0])
  print "Mean/Max duty cycles:",dutyCycles.mean(), dutyCycles.max()
  print "Number of columns that won for > 10% patterns",\
            (dutyCycles>tenPct).sum()
  print "Number of columns that won for > 20% patterns",\
            (dutyCycles>2*tenPct).sum()
  print "Num categories learned",classifier.getParameter('categoryCount')
  print "Number of patterns stored",classifier.getParameter('patternCount')

  return net


def testNetwork(testPath="mnist/testing", savedNetworkFile="mnist_net.nta"):
  net = Network(savedNetworkFile)
  sensor = net.regions['sensor']
  sp = net.regions["SP"]
  classifier = net.regions['classifier']

  print "Reading test images"
  sensor.executeCommand(["loadMultipleImages",testPath])
  numTestImages = sensor.getParameter('numImages')
  print "Number of test images",numTestImages

  start = time.time()

  # Various region parameters
  sensor.setParameter('explorer','Flash')
  classifier.setParameter('inferenceMode', 1)
  classifier.setParameter('learningMode', 0)
  sp.setParameter('inferenceMode', 1)
  sp.setParameter('learningMode', 0)

  numCorrect = 0
  for i in range(numTestImages):
    net.run(1)
    inferredCategory = classifier.getOutputData('categoriesOut').argmax()
    if sensor.getOutputData('categoryOut') == inferredCategory:
      numCorrect += 1
    if i%(numTestImages/100)== 0:
      print "Iteration",i,"numCorrect=",numCorrect

  # Some interesting statistics
  print "Testing time:",time.time()-start
  print "Number of test images",numTestImages
  print "num correct=",numCorrect
  print "pct correct=",(100.0*numCorrect) / numTestImages


def checkNet(net):
  # DEBUG: Verify we set parameters correctly
  # This is the 'correct' way to access internal region parameters. It will
  # work across all languages
  sensor = net.regions['sensor']
  classifier = net.regions['classifier']
  sp = net.regions["SP"]
  width = sensor.getParameter('width')
  height = sensor.getParameter('height')
  print "width/height=",width,height
  print "Classifier distance threshold",classifier.getParameter('distThreshold')
  print "Log path:",sp.getParameter('logPathInput')

  print "min/max phase",net.getMinEnabledPhase(), net.getMaxEnabledPhase()

  # This is a convenient method that only works for Python regions.
  # Here we get a pointer to the actual Python instance of that region.
  pysensor = sensor.getSelf()
  print "Python width/height",pysensor.height, pysensor.width

  print "Explorer:",pysensor.getParameter('explorer')
  print "Filters:",pysensor.getParameter('filters')





if __name__ == "__main__":
  net = createNetwork()
  trainNetwork(net, "mnist_net.nta")

  # As a debugging step, verify we've learned the training set well
  # This assumes you have a small subset of the training images in
  # mnist/small_training
  print "Test on small part of training set"
  testNetwork("mnist/small_training", "mnist_net.nta")

  print "Test on full test set"
  testNetwork(savedNetworkFile="mnist_net.nta")
