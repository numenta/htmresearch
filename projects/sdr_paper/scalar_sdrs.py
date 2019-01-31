# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from __future__ import print_function

import math

import torch

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Need to run it from htmresearch top level:
# python projects/sdr_paper/pytorch_experiments/analyze_model.py model_path


def getSparseTensor(numNonzeros, inputSize, outputSize,
                    onlyPositive=False,
                    initializationMethod="fixed",
                    fixedRange=1.0/24):
  """
  Return a random tensor that is initialized like a weight matrix
  Size is outputSize X inputSize, where weightSparsity% of each row is non-zero
  """
  # Initialize weights in the typical fashion.
  w = torch.Tensor(outputSize, inputSize, )
  if initializationMethod == "uniform":
    a = 1. / numNonzeros
  elif initializationMethod == "fixed":
    a = fixedRange
  else:
    a = 1. / math.sqrt(numNonzeros)

  if onlyPositive:
    w.data.uniform_(0, 2*a)
  else:
    w.data.uniform_(-a, a)

  # Zero out weights for sparse weight matrices
  if numNonzeros < inputSize:
    numZeros = inputSize - numNonzeros

    outputIndices = np.arange(outputSize)
    inputIndices = np.array([np.random.permutation(inputSize)[:numZeros]
                             for _ in outputIndices], dtype=np.long)

    # Create tensor indices for all non-zero weights
    zeroIndices = np.empty((outputSize, numZeros, 2), dtype=np.long)
    zeroIndices[:, :, 0] = outputIndices[:, None]
    zeroIndices[:, :, 1] = inputIndices
    zeroIndices = torch.LongTensor(zeroIndices.reshape(-1, 2))

    zeroWts = (zeroIndices[:, 0], zeroIndices[:, 1])
    w.data[zeroWts] = 0.0

  return w


def plotDot(dot, title="Histogram of dot products",
            path="dot.pdf"):
  bins = np.linspace(dot.min(), dot.max(), 100)
  plt.hist(dot, bins, alpha=0.5, label='All cols')
  plt.title(title)
  plt.xlabel("Dot product")
  plt.ylabel("Number")
  plt.savefig(path)
  plt.close()

# def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
#   """
#   Utility function for computing output of convolutions
#   takes a tuple of (h,w) and returns a tuple of (h,w)
#   """
#   if type(kernel_size) is not tuple:
#       kernel_size = (kernel_size, kernel_size)
#   h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
#   w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
#   return h, w


def getTheta(k, nTrials = 100000, strategy = "mean"):
  """
  Estimate a reasonable value of theta for this k.
  """
  w1 = getSparseTensor(k, k, nTrials,
                       initializationMethod="fixed",
                       fixedRange=1.0/k)
  dotSum = 0.0
  dotMin = 1000000
  for i in range(nTrials):
    dot = w1[i].dot(w1[i])
    dotSum += dot
    dotMin = min(dotMin, dot)

  dotMean = dotSum / nTrials
  print("k=", k, "min/mean diag of w dot products", dotMin, dotMean)

  if strategy == "mean":
    theta = dotMean / 2.0
    print("Using theta as mean / 2.0 = ", theta)
  else:
    theta = dotMin / 2.0
    print("Using theta as min / 2.0 = ", theta)

  return theta


def returnMatches(kw, kv, n, theta):
  """
  :param kw: k for the weight vectors
  :param kv: k for the input vectors
  :param n:  dimensionality of input vector
  :param theta: threshold for matching after dot product

  :return: percent that matched, number that matched, total match comparisons
  """
  # How many prototypes to store
  m1 = 2
  m2 = 1000

  weights = getSparseTensor(kw, n, m1,
                            initializationMethod="fixed",
                            fixedRange=1.0 / kw,
                            )

  # Initialize input vectors using similar range as the weights, but scaled
  # to be positive
  inputVectors = getSparseTensor(kv, n, m2,
                                 onlyPositive=True,
                                 initializationMethod="fixed",
                                 fixedRange=1.0 / kw,
                                 )
  dot = inputVectors.matmul(weights.t())

  # print("min/max/mean w dot products", wd.min(), wd.max(), wd.mean())

  # Let wd.diag().min() be a decent minimal theta for matching
  numMatches = ((dot>theta).sum()).item()
  pctMatches = numMatches / float(m1*m2)

  # plotDot(dot,
  #         title="Histogram of overlaps 100 out of 2000",
  #         path="dot_100_2000.pdf")

  return pctMatches, numMatches, m1*m2


def computeMatchProbability(kw, kv, n, theta, nTrials = 500):
  """
  Runs a number of trials of returnMatches() and returns an overall probability
  of matches given the parameters.

  :param kw: k for the weight vectors
  :param kv: k for the input vectors. If -1, kv is set to n/2
  :param n:  dimensionality of input vector
  :param theta: threshold for matching after dot product
  :param nTrials: number of trials to run

  :return: percent that matched, number that matched, total match comparisons
  """

  if kv == -1:
    kv = int(round(n/2.0))
    print("n,kv",n,kv)

  numMatches = 0
  totalComparisons = 0
  for t in range(nTrials):
    pct, num, total = returnMatches(kw, kv, n, theta)
    numMatches += num
    totalComparisons += total

  pctMatches = float(numMatches) / totalComparisons
  print("kw, kv, n:", kw, kv, n, ", matches:", numMatches,
        ", comparisons:", totalComparisons,
        ", pct matches:", pctMatches)

  return pctMatches, numMatches, totalComparisons


def plotMatches(listofaValues, listofNValues, errors,
                fileName = "images/scalar_effect_of_n.pdf"):
  fig, ax = plt.subplots()

  fig.suptitle("Match probability for sparse vectors")
  ax.set_xlabel("Dimensionality (n)")
  ax.set_ylabel("Frequency of matches")
  ax.set_yscale("log")

  ax.plot(listofNValues, errors[0,:], 'k:',
          label="a=64 (predicted)", marker="o", color='black')
  ax.plot(listofNValues, errors[1,:], 'k:',
          label="a=128 (predicted)", marker="o", color='black')
  ax.plot(listofNValues, errors[2,:], 'k:',
          label="a=256 (predicted)", marker="o", color='black')
  ax.plot(listofNValues, errors[3,:], 'k:',
          label="a=n/2 (predicted)", marker="o", color='black')

  ax.annotate(r"$a = 64$", xy=(listofNValues[3]+100, errors[0,3]),
              xytext=(-5, 2), textcoords="offset points", ha="left",
              color='black')
  ax.annotate(r"$a = 128$", xy=(listofNValues[3]+100, errors[1,3]),
               ha="left",color='black')
  ax.annotate(r"$a = 256$", xy=(listofNValues[3]+100, errors[2,3]),
               ha="left",color='black')
  ax.annotate(r"$a = \frac{n}{2}$", xy=(listofNValues[3]+100, errors[3,3]),
               ha="left",color='black')

  plt.minorticks_off()
  plt.grid(True, alpha=0.3)

  plt.savefig(fileName)
  plt.close()


if __name__ == '__main__':

  listofkValues = [64, 128, 256, -1]
  listofNValues = [250, 500, 1000, 1500, 2000, 2500]

  kw = 24
  theta = getTheta(kw, strategy="mean")
  errors = np.zeros((len(listofkValues), len(listofNValues)))
  for ki, k in enumerate(listofkValues):
    for ni, n in enumerate(listofNValues):
      errors[ki, ni], numMatches, totalComparisons = computeMatchProbability(
        kw, k, n, theta, nTrials=500)
      print()


  # TODO Compute false negatives
  # TODO Compute error as a function of overall vector scaling

  # Results for each setting:
  # kw = 24
  # [[1.02480e-02 1.44700e-03 1.66000e-04 4.70000e-05 1.30000e-05 8.00000e-06]
  #  [4.48810e-02 1.01170e-02 1.35600e-03 4.41000e-04 1.76000e-04 9.50000e-05]
  # [1.16207e-01 4.29550e-02 1.02990e-02 3.39400e-03 1.58100e-03 8.37000e-04]
  # [3.98980e-02 4.27310e-02 4.30160e-02 3.96750e-02 4.52850e-02 3.92980e-02]]

  # kw = 36
  # [[2.2910e-03 1.4400e-04 7.0000e-06 0.0000e+00 0.0000e+00 0.0000e+00]
  #  [2.0024e-02 2.0570e-03 1.6000e-04 1.9000e-05 1.0000e-05 2.0000e-06]
  # [6.1982e-02 1.8586e-02 2.3760e-03 5.1800e-04 1.4100e-04 6.1000e-05]]

  # kw = 64, 5 million comparisons
  # [[7.96000e-05 2.00000e-06 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00]
  #  [2.72540e-03 8.60000e-05 1.00000e-06 0.00000e+00 0.00000e+00 0.00000e+00]
  # [1.97212e-02 2.49180e-03 9.76000e-05 9.00000e-06 4.00000e-07 4.00000e-07]]

  print(errors)

  plotMatches(listofkValues, listofNValues, errors,
              "images/scalar_effect_of_n_mean_theta_kw24.pdf")
