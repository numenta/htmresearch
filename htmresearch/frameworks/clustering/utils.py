import copy
import numpy as np



def percentOverlap(x1, x2):
  """
  Computes the percentage of overlap between vectors x1 and x2.

  :param x1:   (array) binary vector
  :param x2:  (array) binary vector
  :param size: (int)   length of binary vectors

  :return percentOverlap: (float) percentage overlap between x1 and x2
  """
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)
  minX1X2 = min(nonZeroX1, nonZeroX2)
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float(np.dot(x1.T, x2)) / float(minX1X2)
  return percentOverlap



def generateSDR(n, w):
  """
  Generate a random n-dimensional SDR with w bits active
  """
  sdr = np.zeros((n,))
  randomOrder = np.random.permutation(np.arange(n))
  activeBits = randomOrder[:w]
  sdr[activeBits] = 1
  return sdr



def corruptSparseVector(sdr, noiseLevel):
  """
  Add noise to sdr by turning off numNoiseBits active bits and turning on
  numNoiseBits in active bits
  :param sdr: (array) Numpy array of the  SDR
  :param noiseLevel: (float) amount of noise to be applied on the vector.
  """
  numNoiseBits = int(noiseLevel * np.sum(sdr))
  if numNoiseBits <= 0:
    return sdr
  activeBits = np.where(sdr > 0)[0]
  inactiveBits = np.where(sdr == 0)[0]

  turnOffBits = np.random.permutation(activeBits)
  turnOnBits = np.random.permutation(inactiveBits)
  turnOffBits = turnOffBits[:numNoiseBits]
  turnOnBits = turnOnBits[:numNoiseBits]

  sdr[turnOffBits] = 0
  sdr[turnOnBits] = 1



def generateSDRs(numSDRclasses, numSDRsPerClass, n, w, noiseLevel):
  sdrs = []
  for _ in range(numSDRclasses):
    templateSDR = generateSDR(n, w)
    for _ in range(numSDRsPerClass):
      noisySDR = copy.copy(templateSDR)
      corruptSparseVector(noisySDR, noiseLevel)
      sdrs.append(noisySDR)
  return sdrs
