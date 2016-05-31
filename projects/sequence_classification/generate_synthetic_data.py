"""
Generate synthetic sequences using a pool of sequence motifs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import decomposition
import random
plt.ion()
plt.close('all')
# Generate a set of sequence motifs
def generateSequenceMotifs(numMotif, motifLength, seed=None):
  if seed is not None:
    np.random.seed(seed)
  sequenceMotifs = np.random.randn(motifLength, motifLength)
  pca = decomposition.PCA(n_components=numMotif)
  pca.fit(sequenceMotifs)
  sequenceMotifs = pca.components_

  for i in range(numMotif):
    sequenceMotifs[i, :] = sequenceMotifs[i, :]-min(sequenceMotifs[i, :])
    sequenceMotifs[i, :] = sequenceMotifs[i, :]/max(sequenceMotifs[i, :])

  return sequenceMotifs


def generateSequence(sequenceLength, useMotif, currentClass, sequenceMotifs):
  motifLength = sequenceMotifs.shape[1]

  sequence = np.zeros((sequenceLength + 20,))
  motifState = np.zeros((sequenceLength + 20,))
  randomLengthList = np.linspace(1, 10, 10).astype('int')
  # randomLengthList = [1]
  t = 0
  while t < sequenceLength:
    randomLength = np.random.choice(randomLengthList)
    sequence[t:t + randomLength] = np.random.rand(randomLength)
    motifState[t:t + randomLength] = -1
    t += randomLength

    motifIdx = np.random.choice(useMotif[currentClass])
    print "motifIdx: ", motifIdx
    sequence[t:t + motifLength] = sequenceMotifs[motifIdx]
    motifState[t:t + motifLength] = motifIdx
    t += motifLength

  sequence = sequence[:sequenceLength]
  motifState = motifState[:sequenceLength]
  return sequence, motifState



def generateSequences(numSeq, numClass, sequenceLength, useMotif, sequenceMotifs):
  trainData = np.zeros((numSeq, sequenceLength+1))
  numSeqPerClass = numSeq/numClass
  classList = []
  for classIdx in range(numClass):
    classList += [classIdx] * numSeqPerClass

  # classList = np.random.permutation(classList)
  for seq in range(numSeq):
    currentClass = classList[seq]
    # print "useMotif, {}".format(useMotif)
    sequence, motifState = generateSequence(sequenceLength, useMotif,
                                            currentClass, sequenceMotifs)
    trainData[seq, 0] = currentClass
    trainData[seq, 1:] = sequence
  return trainData


numMotif = 5
motifLength = 5
sequenceMotifs = generateSequenceMotifs(numMotif, 5, seed=42)

numTrain = 100
numTest = 100
numClass = 2
motifPerClass = 2

np.random.seed(2)
useMotif = {}
motifList = set(range(numMotif))
for classIdx in range(numClass):
  useMotifForClass = []
  for _ in range(motifPerClass):
    useMotifForClass.append(np.random.choice(list(motifList)))
    motifList.remove(useMotifForClass[-1])
  useMotif[classIdx] = useMotifForClass

sequenceLength = 100


currentClass = 0
sequence, motifState = generateSequence(sequenceLength, useMotif, currentClass, sequenceMotifs)

MotifColor = {}
colorList = ['r','g','b','c','m','y']
i = 0
for c in useMotif.keys():
  for v in useMotif[c]:
    MotifColor[v] = colorList[i]
    i += 1

fig, ax = plt.subplots(nrows=4, ncols=1)

for plti in xrange(4):
  currentClass = [0 if plti < 2 else 1][0]
  sequence, motifState = generateSequence(sequenceLength, useMotif, currentClass, sequenceMotifs)
  ax[plti].plot(sequence, 'k-')

  startPatch = False
  for t in range(len(motifState)):
    if motifState[t] >= 0 and startPatch is False:
      startPatchAt = t
      startPatch = True
      currentMotif = motifState[t]

    if startPatch and (motifState[t] < 0):
      endPatchAt = t-1

      ax[plti].add_patch(
        patches.Rectangle(
          (startPatchAt, 0),
          endPatchAt-startPatchAt, 1, alpha=0.5,
          color=MotifColor[currentMotif]
        )
      )
      startPatch = False

  ax[plti].set_xlim([0, 100])
  ax[plti].set_ylabel('class {}'.format(currentClass))

# ax[1].plot(motifState)


trainData = generateSequences(numTrain, numClass, sequenceLength, useMotif, sequenceMotifs)
testData = generateSequences(numTest, numClass, sequenceLength, useMotif, sequenceMotifs)
np.savetxt('SyntheticData/Test1/Test1_TRAIN', trainData, delimiter=',')
np.savetxt('SyntheticData/Test1/Test1_TEST', testData, delimiter=',')
# writeSequenceToFile('SyntheticData/Test1/Test1_TRAIN', 100, numClass, sequenceLength, useMotif, sequenceMotifs)
# writeSequenceToFile('SyntheticData/Test1/Test1_TEST', 100, numClass, sequenceLength, useMotif, sequenceMotifs)
#


plt.figure()
trainLabel = trainData[:, 0].astype('int')
trainData = trainData[:, 1:]
plt.imshow(trainData[np.where(trainLabel==0)[0],:])
# plt.plot(motifState)