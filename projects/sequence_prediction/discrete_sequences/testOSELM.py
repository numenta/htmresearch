import numpy as np
from htmresearch.algorithms.online_extreme_learning_machine import OSELM


dat = np.loadtxt('segment_test.csv', delimiter=',')

features = dat[:, 1:]
targetLabel = dat[:, 0]
uniqueLabels = sorted(np.unique(targetLabel))

(numSamples, numInputs) = features.shape
numOutputs = len(uniqueLabels)

targets = np.zeros(shape=(numSamples, numOutputs))
for i in range(numSamples):
  targets[i, int(targetLabel[i]-1)] = 1

net = OSELM(numInputs, numOutputs,
            numHiddenNeurons=180, activationFunction='sig')

# Initialization
numInitial = 200
net.initializePhase(features[:numInitial, :], targets[:numInitial, :])

# Online learning
for i in range(numInitial, numSamples):
  net.train(features[[i], :], targets[[i], :])
Y = net.predict(features)

correctPredictions = 0
for i in range(Y.shape[0]):
  predictedLabel = np.argmax(Y[i, :])
  actualLabel = np.argmax(targets[i, :])

  correctPredictions += (predictedLabel == actualLabel)

print "Training Accuracy {}".format(float(correctPredictions)/numSamples)