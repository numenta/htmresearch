import copy
import numpy as np



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
