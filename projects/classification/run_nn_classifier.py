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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
from htmresearch.algorithms.neural_net_classifier import NeuralNetClassifier
from nupic.algorithms.CLAClassifier import CLAClassifier


from nupic.encoders.sdrcategory import SDRCategoryEncoder

plt.ion()
plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42

def initializeEncoder(Nelements, seed):

  # initialize classifiers
  encoder = SDRCategoryEncoder(1024, 40,
                               categoryList=range(0, Nelements),
                               encoderSeed=seed)
  return encoder


def initializeClassifiers(Nelements, encoder):
  cla = CLAClassifier(steps=[0])

  nn_classifier = NeuralNetClassifier(numInputs=encoder.n,
                                      steps=[0], alpha=0.1)

  patternNZ = list(numpy.where(encoder.encode(Nelements-1))[0])
  classification = {'bucketIdx': Nelements-1, 'actValue': Nelements-1}

  # feed in the pattern with the highest bucket index
  claRetval = cla.compute(0, patternNZ, classification,
                           learn=True, infer=True)
  nnRetval = nn_classifier.compute(0, patternNZ, classification,
                                    learn=True, infer=True)
  return cla, nn_classifier


def runSimulation(encoder, cla, nn_classifier,
                  noiseLevel=0.0, changeTaskAfter=-1):
  accuracyTrack = []
  negLLTrack = []

  for recordNum in xrange(1000):
    # use a different encoder to test continuous learning
    if recordNum == changeTaskAfter:
      encoder = initializeEncoder(encoder.ncategories-1, seed=2)

    inputSymbol = numpy.random.randint(encoder.ncategories-1)
    activation = encoder.encode(inputSymbol)

    # add noise to the SDR to increase task difficulty
    if noiseLevel > 0:
      numMissBits = numpy.int(encoder.w * noiseLevel)
      activeBits = numpy.where(activation)[0]
      activation[activeBits[:numMissBits]] = 0

      numRandBits = numpy.int(encoder.w * noiseLevel)
      newBits = numpy.random.randint(encoder.n, size=(numRandBits,))
      activation[newBits] = 1

    patternNZ = list(numpy.where(activation)[0])
    classification = {'bucketIdx': inputSymbol, 'actValue': inputSymbol}

    claRetval = cla.compute(recordNum, patternNZ, classification,
                             learn=True, infer=True)

    nnRetval = nn_classifier.compute(recordNum, patternNZ, classification,
                                      learn=True, infer=True)

    NNNegLL = numpy.log(nnRetval[0][inputSymbol])
    ClaNegLL = numpy.log(claRetval[0][inputSymbol])

    NNBestPrediction = numpy.argmax(nnRetval[0])
    NNAccuracy = (NNBestPrediction == inputSymbol)

    ClaBestPrediction = numpy.argmax(claRetval[0])
    ClaAccuracy = (ClaBestPrediction == inputSymbol)

    negLLTrack.append([ClaNegLL, NNNegLL])
    accuracyTrack.append([ClaAccuracy, NNAccuracy])
  return (negLLTrack, accuracyTrack)


def runExperiemnt1():
  """
  Run both classifiers on noise-free streams
  :return:
  """
  Nelements = 20
  noiseLevel = 0.0
  encoder = initializeEncoder(Nelements, seed=1)
  cla, nn_classifier = initializeClassifiers(Nelements, encoder)
  (negLLTrack,
   accuracyTrack) = runSimulation(encoder, cla, nn_classifier, noiseLevel)

  plt.figure(1)
  negLLTrack = numpy.array(negLLTrack)
  plt.plot(negLLTrack[:, 1])
  plt.plot(negLLTrack[:, 0])
  plt.ylabel(' Log-Likelihood')
  plt.xlabel(' Iteration ')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['NeuralNet', 'CLA'], loc=4)
  plt.savefig('./result/LLvsTraining.pdf')

  # prediction of one input element after training
  patternNZ = list(numpy.where(encoder.encode(10))[0])
  classification = {'bucketIdx': 10, 'actValue': 10}

  claRetval = cla.compute(cla._learnIteration+1, patternNZ,
                          classification, learn=False, infer=True)
  nnRetval = nn_classifier.compute(nn_classifier._learnIteration+1, patternNZ,
                                   classification, learn=False, infer=True)

  plt.figure(3)
  plt.plot(nnRetval[0])
  plt.plot(claRetval[0])
  plt.xlabel('Possible Inputs')
  plt.ylabel(' Predicted Probability')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['NeuralNet', 'CLA'])
  plt.savefig('./result/ExamplePredictionAfterTraining.pdf')
  # plt.figure(2)
  # accuracyTrack = numpy.array(accuracyTrack)
  # plt.plot(accuracyTrack)
  # plt.ylabel(' Predictino Accuracy ')
  # plt.xlabel(' Iteration ')


def runExperiment2():
  """
  plot LL after training vs. noise level
  """

  Nelements = 20
  noiseLevelList = numpy.linspace(0, 1.0, num=20)
  negLLCLA = []
  negLLNN = []
  for noiseLevel in noiseLevelList:
    encoder = initializeEncoder(Nelements, seed=1)
    cla, nn_classifier = initializeClassifiers(Nelements, encoder)
    (negLLTrack,
     accuracyTrack) = runSimulation(encoder, cla, nn_classifier, noiseLevel)

    negLLTrack = numpy.array(negLLTrack)
    negLLCLA.append(numpy.mean(negLLTrack[-100, 0]))
    negLLNN.append(numpy.mean(negLLTrack[-100, 1]))

  noiseLevelList = noiseLevelList * 40

  plt.figure(4)
  plt.plot(noiseLevelList, negLLNN)
  plt.plot(noiseLevelList, negLLCLA)
  plt.xlabel(' Noise Level (# random bits) ')
  plt.ylabel(' Log-likelihood')
  plt.legend(['NeuralNet', 'CLA'], loc=3)
  plt.savefig('./result/LLvsNoise.pdf')

def runExperiment3():
  """
  Change task at iteration=500, test continuous learning
  :return:
  """
  Nelements = 20
  noiseLevel = 0.0
  encoder = initializeEncoder(Nelements, seed=1)
  cla, nn_classifier = initializeClassifiers(Nelements, encoder)
  (negLLTrack,
   accuracyTrack) = runSimulation(encoder, cla, nn_classifier,
                                  noiseLevel, changeTaskAfter=500)

  plt.figure(5)
  negLLTrack = numpy.array(negLLTrack)
  plt.plot(negLLTrack[:, 1])
  plt.plot(negLLTrack[:, 0])
  plt.ylabel(' Log-Likelihood')
  plt.xlabel(' Iteration ')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['NeuralNet', 'CLA'], loc=4)
  plt.savefig('./result/LLvsTraining_ChangeAt500.pdf')

if __name__ == "__main__":
  # Example prediction with noise-free streams
  runExperiemnt1()

  # LL vs Noise
  runExperiment2()

  # Continus learning task
  runExperiment3()