import numpy as np



def percentOverlap(x1, x2):
  """
  Computes the percentage of overlap between vectors x1 and x2.

  :param x1:   (array) binary vector
  :param x2:  (array) binary vector
  :param size: (int)   length of binary vectors

  :return percentOverlap: (float) percentage overlap between x1 and x2
  """
  nonZeroX1 = float(np.count_nonzero(x1))
  nonZeroX2 = float(np.count_nonzero(x2))
  minX1X2 = min(nonZeroX1, nonZeroX2)
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float(np.dot(x1.T, x2)) / np.sqrt(nonZeroX1 * nonZeroX2)

  return percentOverlap



def kernel_dist(kernel):
  return lambda x, y: kernel(x, x) - 2 * kernel(x, y) + kernel(y, y)
