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


def clusterToClusterDist(c1, c2):
  """
  Distance between 2 clusters

  :param c1: (np.array) cluster 1
  :param c2: (np.array) cluster 2
  :return: distance between 2 clusters
  """
  minDists = []
  for sdr1 in c1:
    d = []
    for sdr2 in c2:
      d.append(percentOverlap(sdr1, sdr2))
    minDists.append(min(d))

  return sum(minDists)



def kernel_dist(kernel):
  return lambda x, y: kernel(x, x) - 2 * kernel(x, y) + kernel(y, y)
