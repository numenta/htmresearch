"""
Implementation of a simple 1-layer feedforward classification network
Training is achieved with gradient descent
"""
import numpy as np
import scipy


def L2regularization(w, regularizationLambda):
  dW = np.zeros(w.shape)
  costLL = 0
  if regularizationLambda is None:
    return costLL, dW
  else:
    for i in range(len(regularizationLambda['lambdaL2'])):
      lambdaL2 = regularizationLambda['lambdaL2'][i]
      idx = regularizationLambda['wIndice']
      costLL += np.sum(np.square(w[idx])) * lambdaL2
      dW[idx] += 2 * lambdaL2 * np.sum(w[idx])
    return costLL, dW



def costFuncClassifier(w, sdrInputs, classLabels, regularizationLambda=0):
  """
  :param w: feedforward weight matrix (numInputs, numClass)
  :param sdrInputs: list of sdr inputs
  :param classLabels: list of class labels
  :return: costLL negative log likelihood
  """

  numSamples = len(sdrInputs)
  numInputs = len(sdrInputs[0])
  numClass = len(w)/numInputs

  costLLL2, dWL2 = L2regularization(w, regularizationLambda)
  w = np.reshape(w, (numInputs, numClass))
  # cost: negative log-likelihood
  costLL = 0
  dW = np.zeros((numInputs, numClass))
  for i in range(numSamples):
    activation = np.dot(sdrInputs[i], w)
    y = np.exp(activation)
    y = y/np.sum(y)

    target = np.zeros((numClass, ))
    target[classLabels[i]] = 1

    costLL -= np.log(y[classLabels[i]])
    dW += np.dot(np.reshape(sdrInputs[i], (numInputs, 1)),
                 np.reshape((y - target), (1, numClass)))

  dW = np.reshape(dW, (numInputs * numClass,))

  costLL += costLLL2
  dW += dWL2
  return costLL, dW



class classificationNetwork(object):
  def __init__(self, numInputs, numClass, regularizationLambda=None):
    self.numInputs = numInputs
    self.numClass = numClass
    self.w = np.zeros((numInputs*numClass, ))
    self.regularizationLambda = regularizationLambda


  def optimize(self, sdrInputs, trainLabel, wInit):
    (wOpt, nfeval, rc) = scipy.optimize.fmin_tnc(costFuncClassifier, wInit,
                                                 args=(sdrInputs, trainLabel,
                                                       self.regularizationLambda))
    self.w = wOpt


  def classify(self, sdrInputs):
    w = np.reshape(self.w, (self.numInputs, self.numClass))
    numSample = len(sdrInputs)
    classProb = np.zeros((numSample, self.numClass))
    for i in range(numSample):
      activation = np.dot(sdrInputs[i], w)
      y = np.exp(activation)
      classProb[i, :] = y / np.sum(y)
    return classProb


  def accuracy(self, sdrInputs, labels):
    classProb = self.classify(sdrInputs)
    numSample = len(sdrInputs)
    numCorrect = 0
    for i in range(numSample):
      numCorrect += (np.argmax(classProb[i, :]) == labels[i])

    accuracy = np.float(numCorrect)/numSample
    return accuracy