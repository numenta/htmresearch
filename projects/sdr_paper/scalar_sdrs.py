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
import time
from multiprocessing import Pool

import torch

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Need to run it from htmresearch top level:
# python projects/sdr_paper/pytorch_experiments/analyze_model.py model_path


def getSparseTensor(numNonzeros, inputSize, outputSize,
                    onlyPositive=False,
                    fixedRange=1.0/24):
  """
  Return a random tensor that is initialized like a weight matrix
  Size is outputSize X inputSize, where weightSparsity% of each row is non-zero
  """
  # Initialize weights in the typical fashion.
  w = torch.Tensor(outputSize, inputSize, )

  if onlyPositive:
    w.data.uniform_(0, fixedRange)
  else:
    w.data.uniform_(-fixedRange, fixedRange)

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


def getTheta(k, nTrials=100000):
  """
  Estimate a reasonable value of theta for this k.
  """
  w1 = getSparseTensor(k, k, nTrials,
                       fixedRange=1.0/k)
  dotSum = 0.0
  dotMin = 1000000
  for i in range(nTrials):
    dot = w1[i].dot(w1[i])
    dotSum += dot
    dotMin = min(dotMin, dot)

  dotMean = dotSum / nTrials
  print("k=", k, "min/mean diag of w dot products", dotMin, dotMean)

  theta = dotMean / 2.0
  print("Using theta as mean/2 = ", theta)

  return theta


def returnMatches(kw, kv, n, theta, inputScaling=1.0):
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
                            fixedRange=1.0 / kw,
                            )

  # Initialize input vectors using similar range as the weights, but scaled
  # to be positive
  inputVectors = getSparseTensor(kv, n, m2,
                                 onlyPositive=True,
                                 fixedRange=inputScaling / kw,
                                 )
  dot = inputVectors.matmul(weights.t())

  # print("min/max/mean w dot products", wd.min(), wd.max(), wd.mean())

  # Let wd.diag().min() be a decent minimal theta for matching
  numMatches = ((dot > theta).sum()).item()
  pctMatches = numMatches / float(m1*m2)

  # plotDot(dot,
  #         title="Histogram of overlaps 100 out of 2000",
  #         path="dot_100_2000.pdf")

  return pctMatches, numMatches, m1*m2


def computeMatchProbability(args):
  """
  Runs a number of trials of returnMatches() and returns an overall probability
  of matches given the parameters.

  :param args is a dictionary containing the following keys:

  kw: k for the weight vectors

  kv: k for the input vectors. If -1, kv is set to n/2

  n:  dimensionality of input vector

  theta: threshold for matching after dot product

  nTrials: number of trials to run

  inputScaling: scale factor for the input vectors. 1.0 means the scaling
    is the same as the stored weight vectors.

  :return: args updated with the percent that matched
  """
  kv = args["k"]
  n = args["n"]
  kw = args["kw"]
  theta = args["theta"]

  if kv == -1:
    kv = int(round(n/2.0))

  numMatches = 0
  totalComparisons = 0
  for t in range(args["nTrials"]):
    pct, num, total = returnMatches(kw, kv, n, theta, args["inputScaling"])
    numMatches += num
    totalComparisons += total

  pctMatches = float(numMatches) / totalComparisons
  print("kw, kv, n, s:", kw, kv, n, args["inputScaling"],
        ", matches:", numMatches,
        ", comparisons:", totalComparisons,
        ", pct matches:", pctMatches)

  args.update({"pctMatches": pctMatches})

  return args


def plotMatches(listofKValues, listofNValues, errors,
                fileName = "images/scalar_effect_of_n.pdf"):
  fig, ax = plt.subplots()

  fig.suptitle("Prob. of matching a sparse random input")
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
               ha="left", color='black')
  ax.annotate(r"$a = 256$", xy=(listofNValues[3]+100, errors[2,3]),
               ha="left", color='black')
  ax.annotate(r"$a = \frac{n}{2}$", xy=(listofNValues[3]+100, errors[3,3]),
               ha="left", color='black')

  plt.minorticks_off()
  plt.grid(True, alpha=0.3)

  plt.savefig(fileName)
  plt.close()


def plotScaledMatches(listofKValues, listOfScales, errors,
                fileName = "images/scalar_effect_of_scale.pdf"):
  fig, ax = plt.subplots()

  fig.suptitle("Prob. of matching a sparse random input")
  ax.set_xlabel("Scale factor (s)")
  ax.set_ylabel("Frequency of matches")
  ax.set_yscale("log")

  ax.plot(listOfScales, errors[0, :], 'k:',
          label="a=64 (predicted)", marker="o", color='black')
  ax.plot(listOfScales, errors[1, :], 'k:',
          label="a=128 (predicted)", marker="o", color='black')
  ax.plot(listOfScales, errors[2, :], 'k:',
          label="a=128 (predicted)", marker="o", color='black')


  ax.annotate(r"$a = 64$", xy=(listOfScales[3]+0.2, errors[0, 3]),
              xytext=(-5, 2), textcoords="offset points", ha="left",
              color='black')
  ax.annotate(r"$a = 128$", xy=(listOfScales[3]+0.2, errors[1, 3]),
               ha="left", color='black')
  ax.annotate(r"$a = 256$", xy=(listOfScales[3]+0.2, errors[2, 3]),
               ha="left", color='black')

  plt.minorticks_off()
  plt.grid(True, alpha=0.3)

  plt.savefig(fileName)
  plt.close()


def computeMatchProbabilities(listofkValues=[64, 128, 256, -1],
                              listofNValues=[250, 500, 1000, 1500, 2000, 2500],
                              inputScale=2.0,
                              kw=24,
                              numWorkers=8,
                              nTrials=1000,
                              ):

  print("Computing match probabilities for input scale=", inputScale)

  # Create arguments for the possibilities we want to test
  args = []
  theta = getTheta(kw)
  for ki, k in enumerate(listofkValues):
    for ni, n in enumerate(listofNValues):
      args.append({
          "k": k, "kw": kw, "n": n, "theta": theta,
          "nTrials": nTrials, "inputScaling": 2.0,
          "errorIndex": [ki, ni],
          })

  numExperiments = len(args)
  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    rs = pool.map_async(computeMatchProbability, args, chunksize=1)
    while not rs.ready():
      remaining = rs._number_left
      pctDone = 100.0 - (100.0*remaining) / numExperiments
      print("    =>", remaining,
            "experiments remaining, percent complete=",pctDone)
      time.sleep(5)
    pool.close()  # No more work
    pool.join()
    result = rs.get()
  else:
    result = []
    for arg in args:
      result.append(computeMatchProbability(arg))


  # Read out results and store in numpy array for plotting
  errors = np.zeros((len(listofkValues), len(listofNValues)))
  for r in result:
    errors[r["errorIndex"][0], r["errorIndex"][1]] = r["pctMatches"]

  print("Errors for kw=", kw)
  print(errors)
  plotMatches(listofkValues, listofNValues, errors,
              "images/scalar_effect_of_n_kw" + str(kw) + ".pdf")


  # Errors for kw=24
  # errors= np.array([
  #   [1.005700e-02, 1.538500e-03, 1.820000e-04, 5.300000e-05, 2.300000e-05, 9.500000e-06],
  #   [4.330450e-02, 9.127500e-03, 1.580500e-03, 4.075000e-04, 1.885000e-04, 8.500000e-05],
  #   [1.189445e-01, 4.365250e-02, 9.500000e-03, 3.274000e-03, 1.528500e-03, 7.790000e-04],
  #   [4.220400e-02, 4.227200e-02, 4.286450e-02, 4.149700e-02, 4.143650e-02, 4.135850e-02],
  # ])
  # plotMatches(listofkValues, listofNValues, errors,
  #             "images/scalar_effect_of_n_kw" + str(kw) + ".pdf")


def computeScaledProbabilities(listOfScales=[0.5, 1.0, 1.5, 2.0, 2.5],
                               listofkValues=[64, 128, 256],
                              ):
  # print("Scale test")
  # for kw in [24, 36]:
  #   theta = getTheta(kw)
  #   errors = np.zeros((len(listofkValues), len(listOfScales)))
  #   for ki, k in enumerate(listofkValues):
  #     for si, inputScaling in enumerate(listOfScales):
  #       n = 1000
  #       errors[ki, si] = computeMatchProbability(
  #         kw, k, n, theta, nTrials=1000, inputScaling=inputScaling)
  #       print()
  #
  #   print("Errors for kw=", kw)
  #   print(errors)
  #   plotScaledMatches(listofkValues, listOfScales, errors,
  #               "images/scalar_effect_of_scale_kw" + str(kw) + ".pdf")

  # Errors for kw= 24
  errors = np.array([
    [0.0000e+00, 0.0000e+00, 2.5000e-06, 2.0650e-04, 1.1655e-03],
    [0.0000e+00, 1.0000e-06, 1.0150e-04, 1.4270e-03, 6.5985e-03],
    [0.0000e+00, 1.4500e-05, 1.3850e-03, 9.7485e-03, 2.9147e-02]
  ])
  plotScaledMatches(listofkValues, listOfScales, errors,
                  "images/scalar_effect_of_scale_kw24.pdf")


def computeFalseNegatives(listOfNoises=[0.5, 1.0, 1.5, 2.0, 2.5],
                          listofkValues=[64, 128, 256],
                          ):
  pass


if __name__ == '__main__':

  # computeMatchProbabilities(kw=24, nTrials=1000)
  computeMatchProbabilities(kw=25, nTrials=1000)
  computeMatchProbabilities(kw=50, nTrials=1000)

  # computeScaledProbabilities()

  # TODO Compute false negatives

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

