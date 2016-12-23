#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
#5
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import time
import json
import numpy

from nupic.bindings.math import GetNTAReal
from nupic.engine import Network
from nupic.research import fdrutilities

from PIL import Image

"""
This script visually illustrates SP learning on the MNIST dataset. The
function plotPermanences() plots the top columns in the SP and saves them in
an image file named 'master_ITERATION.png'.  This experiment takes a long
time to run since it goes over the training set multiple times.

Note that column activity changes during the course of learning. This means
that in order to have a consistent set of column images you have to run the
experiment twice. First uncomment the line '#print columnList' to get the end most active
columns. Second, run it again but pass that list of columns in to the
plotPermanences function.
"""


DEFAULT_IMAGESENSOR_PARAMS ={
  'width': 32,
  'height': 32,
  'mode': 'bw',
  'background': 0,
  'explorer': 'RandomFlash'
}

DEFAULT_SP_PARAMS = {
  'columnCount': 12288,
  'spatialImp': 'cpp',
  'inputWidth': 1024,
  'spVerbosity': 2,
  'synPermConnected': 0.5,
  'synPermActiveInc': 0.001,
  'synPermInactiveDec': 0.0005,
  'seed': 1956,
  'numActiveColumnsPerInhArea': 1600,
  'globalInhibition': 1,
  'potentialPct': 0.4,
  'boostStrength': 3.0
}

DEFAULT_CLASSIFIER_PARAMS = {
  'distThreshold': 0.000001,
  'maxCategoryCount': 10,
  'distanceMethod': 'rawOverlap',  # Default is Euclidean distance
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

  # Make sure all objects are initialized
  net.initialize()

  return net

def plotPermanences(network = None, savedNetworkFile = "mnist_net.nta",
                    columnList = None, iteration=0):
  """
  Plots the permanences of the top columns into a single master image
  If columnList is specified, uses those columns otherwise extracts the
  most active columns from the spatial pooler using duty cycle.
  """
  # Get the spatial pooler from the network, otherwise read it from checkpoint.
  if network is None:
    network = Network(savedNetworkFile)
  spRegion = network.regions["SP"]
  spSelf = spRegion.getSelf()
  sp = spSelf._sfdr

  # If we are not given a column list, retrieve columns with highest duty cycles
  dutyCycles = numpy.zeros(sp.getNumColumns(), dtype=GetNTAReal())
  sp.getActiveDutyCycles(dutyCycles)
  if columnList is None:
    mostActiveColumns = list(dutyCycles.argsort())
    mostActiveColumns.reverse()
    columnList = mostActiveColumns[0:400]
    #print columnList

  # Create empty master image with the top 25 columns. We will paste
  # individual column images into this image
  numImagesPerRowInMaster = 20
  masterImage = Image.new("L",((32+2)*numImagesPerRowInMaster,
                               (32+2)*numImagesPerRowInMaster),255)

  for rank,col in enumerate(columnList):
    #print "Col=",col,"rank=",rank,"dutyCycle=",dutyCycles[col]
    pyPerm = numpy.zeros(sp.getNumInputs(), dtype=GetNTAReal())
    sp.getPermanence(col,pyPerm)

    # Create small image for each column
    pyPerm = pyPerm/pyPerm.max()
    pyPerm = (pyPerm*255.0)
    pyPerm = pyPerm.reshape((32,32))
    pyPerm = (pyPerm).astype('uint8')
    img = Image.fromarray(pyPerm)

    # Paste it into master image
    if rank < numImagesPerRowInMaster*numImagesPerRowInMaster:
      x = rank%numImagesPerRowInMaster*(32+2)
      y = (rank/numImagesPerRowInMaster)*(32+2)
      masterImage.paste(img,(x,y))

  # Save master image
  masterImage.save("master_%05d.png"%(iteration))



def trainNetwork(net, dataPath="mnist/small_training",
                 networkFile="mnist_net.nta"):
  # Some stuff we will need later
  sensor = net.regions['sensor']
  imSensor = sensor.getSelf()
  sp = net.regions["SP"]
  pysp = sp.getSelf()
  classifier = net.regions['classifier']
  dutyCycles = numpy.zeros(DEFAULT_SP_PARAMS['columnCount'], dtype=GetNTAReal())

  # Plot untrained permanences
  plotPermanences(network = net)

  print "============= Loading training images ================="
  t1 = time.time()
  sensor.executeCommand(["loadMultipleImages", dataPath])
  numTrainingImages = sensor.getParameter('numImages')
  start = time.time()
  print "Load time for training images:",start-t1
  print "Number of training images",numTrainingImages

  # First train just the SP
  print "============= SP training ================="
  imSensor.setParameter('explorer',0, ['RandomFlash', {'seed':0}])
  classifier.setParameter('inferenceMode', 0)
  classifier.setParameter('learningMode', 0)
  sp.setParameter('learningMode', 1)
  sp.setParameter('inferenceMode', 1)
  nTrainingIterations = 3*numTrainingImages
  for i in range(nTrainingIterations):
    net.run(1)
    dutyCycles += pysp._spatialPoolerOutput
    if i%(nTrainingIterations/200)== 0:
      print "Iteration",i,"Category:",sensor.getOutputData('categoryOut')
      plotPermanences(network = net, iteration = i)

  # Now train just the classifier sequentially on all training images
  print "============= Classifier training ================="
  sensor.setParameter('explorer','Flash')
  classifier.setParameter('inferenceMode', 0)
  classifier.setParameter('learningMode', 1)
  sp.setParameter('learningMode', 0)
  sp.setParameter('inferenceMode', 1)
  for i in range(numTrainingImages):
    net.run(1)
    if i%(numTrainingImages/10)== 0:
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
    if i%(numTestImages/10)== 0:
      print "Iteration",i,"numCorrect=",numCorrect

  # Some interesting statistics
  print "Testing time:",time.time()-start
  print "Number of test images",numTestImages
  print "num correct=",numCorrect
  print "pct correct=",(100.0*numCorrect) / numTestImages


if __name__ == "__main__":
  checkpointFile = "mnist_net.nta"
  # net = createNetwork()
  # trainNetwork(net, "mnist/training", checkpointFile)
  #
  # # Save final images
  # plotPermanences(network = net, savedNetworkFile = checkpointFile,
  #                 iteration=500000)

  # As a debugging step, verify we've learned the training set well
  # This assumes you have a small subset of the training images in
  # mnist/small_training
  # print "Test on small part of training set"
  # testNetwork("mnist/small_training", checkpointFile)

  # print "Test on full test set"
  # testNetwork(savedNetworkFile=checkpointFile)
  #
  # print "Test on 10% occlusion test set"
  # testNetwork(testPath="mnist/testing_10", savedNetworkFile=checkpointFile)
  #
  # print "Test on 30% occlusion test set"
  # testNetwork(testPath="mnist/testing_30", savedNetworkFile=checkpointFile)
  #
  # print "Test on 50% occlusion test set"
  # testNetwork(testPath="mnist/testing_50", savedNetworkFile=checkpointFile)

  print "Test on 75% occlusion test set"
  testNetwork(testPath="mnist/testing_75", savedNetworkFile=checkpointFile)
