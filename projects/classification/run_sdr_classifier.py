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
from nupic.algorithms.sdr_classifier import SDRClassifier
from nupic.algorithms.CLAClassifier import CLAClassifier


from nupic.encoders.sdrcategory import SDRCategoryEncoder

plt.ion()
plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42

def myLog(x):
  x = 0.001 if x<0.001 else x
  return numpy.log(x)

def initializeEncoder(Nelements, seed):

  # initialize classifiers
  encoder = SDRCategoryEncoder(1024, 40,
                               categoryList=range(0, Nelements),
                               encoderSeed=seed)
  return encoder



def initializeClassifiers(Nelements, encoder):
  claClassiiier = CLAClassifier(steps=[0])

  sdrClassifier = SDRClassifier(steps=[0], alpha=0.1)

  patternNZ = list(numpy.where(encoder.encode(Nelements-1))[0])
  classification = {'bucketIdx': Nelements-1, 'actValue': Nelements-1}

  # feed in the pattern with the highest bucket index
  claRetval = claClassiiier.compute(0, patternNZ, classification,
                           learn=True, infer=True)
  sdrRetval = sdrClassifier.compute(0, patternNZ, classification,
                                    learn=True, infer=True)
  return claClassiiier, sdrClassifier



def runSimulation(encoder, cla, sdrClassifier,
                  noiseLevel=0.0, changeTaskAfter=-1, nstep=1000):
  accuracyTrack = []
  negLLTrack = []

  for recordNum in xrange(nstep):
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

    sdrRetval = sdrClassifier.compute(recordNum, patternNZ, classification,
                                      learn=True, infer=True)

    NNNegLL = myLog(sdrRetval[0][inputSymbol])
    ClaNegLL = myLog(claRetval[0][inputSymbol])

    NNBestPrediction = numpy.argmax(sdrRetval[0])
    NNAccuracy = (NNBestPrediction == inputSymbol)

    ClaBestPrediction = numpy.argmax(claRetval[0])
    ClaAccuracy = (ClaBestPrediction == inputSymbol)

    negLLTrack.append([ClaNegLL, NNNegLL])
    accuracyTrack.append([int(ClaAccuracy), int(NNAccuracy)])
  return (negLLTrack, accuracyTrack)



def runExperiemnt1():
  """
  Run both classifiers on noise-free streams
  :return:
  """

  negLLTrackSum = 0
  accuracyTrackSum = 0
  Nrpt = 10
  for rpt in range(Nrpt):
    Nelements = 20
    noiseLevel = 0.0
    encoder = initializeEncoder(Nelements, seed=1)
    cla, sdrClassifier = initializeClassifiers(Nelements, encoder)
    (negLLTrack,
     accuracyTrack) = runSimulation(encoder, cla, sdrClassifier, noiseLevel, nstep=500)

    negLLTrack = numpy.array(negLLTrack)
    accuracyTrack = numpy.array(accuracyTrack).astype('float32')

    negLLTrackSum = negLLTrackSum + negLLTrack
    accuracyTrackSum = accuracyTrackSum + accuracyTrack

  negLLTrackSum /= Nrpt
  accuracyTrackSum /= Nrpt
  plt.figure(1)
  plt.subplot(2,2,1)
  v = numpy.ones((5, ))/5
  plt.plot(numpy.convolve(negLLTrackSum[:, 1], v, 'valid'))
  plt.plot(numpy.convolve(negLLTrackSum[:, 0], v, 'valid'), '--')
  plt.ylim([-4, 0.1])
  plt.ylabel(' Log-Likelihood')
  plt.xlabel(' Iteration ')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['SDR Classifier', 'CLA Classifier'], loc=4)

  plt.subplot(2,2,2)
  plt.plot(numpy.convolve(accuracyTrackSum[:, 1], v, 'valid'))
  plt.plot(numpy.convolve(accuracyTrackSum[:, 0], v, 'valid'), '--')
  plt.ylim([0, 1.05])
  plt.ylabel(' Accuracy ')
  plt.xlabel(' Iteration ')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['SDR Classifier', 'CLA Classifier'], loc=4)

  plt.savefig('./result/LLvsTraining.pdf')

  # prediction of one input element after training
  patternNZ = list(numpy.where(encoder.encode(10))[0])
  classification = {'bucketIdx': 10, 'actValue': 10}

  claRetval = cla.compute(cla._learnIteration+1, patternNZ,
                          classification, learn=False, infer=True)
  sdrRetval = sdrClassifier.compute(sdrClassifier._learnIteration+1, patternNZ,
                                   classification, learn=False, infer=True)

  plt.figure(3)
  plt.plot(sdrRetval[0])
  plt.plot(claRetval[0])
  plt.xlabel('Possible Inputs')
  plt.ylabel(' Predicted Probability')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['SDR', 'CLA'])
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
  noiseLevelList = numpy.linspace(0, 1.0, num=21)
  negLLCLA = []
  negLLSDR = []
  accuracyCLA = []
  accuracySDR = []
  for noiseLevel in noiseLevelList:
    encoder = initializeEncoder(Nelements, seed=1)
    claClassifier, sdrClassifier = initializeClassifiers(Nelements, encoder)
    (negLLTrack, accuracyTrack) = runSimulation(
      encoder, claClassifier, sdrClassifier, noiseLevel)

    negLLTrack = numpy.array(negLLTrack)
    accuracyTrack = numpy.array(accuracyTrack)
    negLLCLA.append(numpy.mean(negLLTrack[-100:, 0]))
    negLLSDR.append(numpy.mean(negLLTrack[-100:, 1]))
    accuracyCLA.append(numpy.mean(accuracyTrack[-100:, 0]))
    accuracySDR.append(numpy.mean(accuracyTrack[-100:, 1]))

  noiseLevelList = noiseLevelList * 40

  plt.figure(4)
  plt.subplot(2,2,1)
  plt.plot(noiseLevelList, negLLSDR, '-o')
  plt.plot(noiseLevelList, negLLCLA, '-s')
  plt.xlabel(' Noise Level (# random bits) ')
  plt.ylabel(' Log-likelihood')
  plt.legend(['SDR Classifier', 'CLA Classifier'], loc=3)
  plt.subplot(2,2,2)
  plt.plot(noiseLevelList, accuracySDR, '-o')
  plt.plot(noiseLevelList, accuracyCLA, '-s')
  plt.xlabel(' Noise Level (# random bits) ')
  plt.ylabel(' Accuracy ')
  plt.legend(['SDR Classifier', 'CLA Classifier'], loc=3)

  plt.savefig('./result/LLvsNoise.pdf')



def runExperiment3():
  """
  Change task at iteration=500, test continuous learning
  :return:
  """
  Nelements = 20
  noiseLevel = 0.0
  encoder = initializeEncoder(Nelements, seed=1)
  cla, sdrClassifier = initializeClassifiers(Nelements, encoder)
  (negLLTrack,
   accuracyTrack) = runSimulation(encoder, cla, sdrClassifier,
                                  noiseLevel, changeTaskAfter=500)

  plt.figure(5)
  negLLTrack = numpy.array(negLLTrack)
  v = numpy.ones((5, ))/5
  plt.subplot(2,2,1)
  plt.plot(numpy.convolve(negLLTrack[:, 1], v, 'valid'))
  plt.plot(numpy.convolve(negLLTrack[:, 0], v, 'valid'))
  plt.ylim([-4, .1])
  plt.ylabel(' Log-Likelihood')
  plt.xlabel(' Iteration ')
  plt.title(' Noise Level: ' + str(noiseLevel))
  plt.legend(['SDR Classifier', 'CLA Classifier'], loc=4)
  plt.savefig('./result/LLvsTraining_ChangeAt500.pdf')

if __name__ == "__main__":
  # Example prediction with noise-free streams
  runExperiemnt1()

  # LL vs Noise
  runExperiment2()

  # Continus learning task
  runExperiment3()