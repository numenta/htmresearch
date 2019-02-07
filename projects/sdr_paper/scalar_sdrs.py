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
  theDots = np.zeros(nTrials)
  w1 = getSparseTensor(k, k, nTrials, fixedRange=1.0/k)
  for i in range(nTrials):
    theDots[i] = w1[i].dot(w1[i])

  dotMean = theDots.mean()
  print("k=", k, "min/mean/max diag of w dot products",
        theDots.min(), dotMean, theDots.max())

  theta = dotMean
  print("Using theta as mean = ", theta)

  return theta, theDots


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


def plotThetaDistribution(kw, fileName = "images/theta_distribution.pdf"):
  theta, theDots = getTheta(kw)

  # Plot histogram of overlaps
  bins = np.linspace(float(theDots.min()), float(theDots.max()), 50)
  plt.hist(theDots, bins, alpha=0.5, label='Dot products')
  plt.legend(loc='upper right')
  plt.xlabel("Dot product")
  plt.ylabel("Frequency")
  plt.title("Distribution of dot products, kw=" + str(kw))
  plt.savefig(fileName)
  plt.close()


def computeMatchProbabilityParallel(args, numWorkers=8):
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

  return result


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
  theta, _ = getTheta(kw)
  for ki, k in enumerate(listofkValues):
    for ni, n in enumerate(listofNValues):
      args.append({
          "k": k, "kw": kw, "n": n, "theta": theta,
          "nTrials": nTrials, "inputScaling": 1.0,
          "errorIndex": [ki, ni],
          })

  result = computeMatchProbabilityParallel(args, numWorkers)


  # Read out results and store in numpy array for plotting
  errors = np.zeros((len(listofkValues), len(listofNValues)))
  for r in result:
    errors[r["errorIndex"][0], r["errorIndex"][1]] = r["pctMatches"]

  print("Errors for kw=", kw)
  print(errors)
  plotMatches(listofkValues, listofNValues, errors,
              "images/scalar_effect_of_n_kw" + str(kw) + ".pdf")

  # Errors for kw= 16
  # [[0.026898   0.006951   0.00167767 0.000688   0.00038083 0.00023117]
  #  [0.07918917 0.02689133 0.00705783 0.00307383 0.0016235  0.00102417]
  #  [0.157351   0.07791967 0.0272435  0.012492   0.00713717 0.00450233]
  #  [0.0761305  0.07731667 0.08056233 0.07661033 0.07579167 0.07790967]]

  # Errors for kw=24
  # errors= np.array([
  #   [1.005700e-02, 1.538500e-03, 1.820000e-04, 5.300000e-05, 2.300000e-05, 9.500000e-06],
  #   [4.330450e-02, 9.127500e-03, 1.580500e-03, 4.075000e-04, 1.885000e-04, 8.500000e-05],
  #   [1.189445e-01, 4.365250e-02, 9.500000e-03, 3.274000e-03, 1.528500e-03, 7.790000e-04],
  #   [4.220400e-02, 4.227200e-02, 4.286450e-02, 4.149700e-02, 4.143650e-02, 4.135850e-02],
  # ])
  # plotMatches(listofkValues, listofNValues, errors,
  #             "images/scalar_effect_of_n_kw" + str(kw) + ".pdf")

  # Errors for kw= 16
  # [[0.026898   0.006951   0.00167767 0.000688   0.00038083 0.00023117]
  #  [0.07918917 0.02689133 0.00705783 0.00307383 0.0016235  0.00102417]
  #  [0.157351   0.07791967 0.0272435  0.012492   0.00713717 0.00450233]
  #  [0.0761305  0.07731667 0.08056233 0.07661033 0.07579167 0.07790967]]

  # Errors for kw= 32
  # [[3.69650000e-03 3.24833333e-04 1.81666667e-05 4.00000000e-06
  #   2.00000000e-06 1.00000000e-06]
  #  [2.46621667e-02 3.71833333e-03 3.40166667e-04 7.38333333e-05
  #   1.95000000e-05 8.00000000e-06]
  #  [7.99258333e-02 2.53685000e-02 3.83033333e-03 9.25500000e-04
  #   3.49500000e-04 1.47000000e-04]
  #  [2.34080000e-02 2.21905000e-02 2.32210000e-02 2.29433333e-02
  #   2.40363333e-02 2.32640000e-02]]

  # Errors for kw= 48
  # [[5.65333333e-04 1.73333333e-05 0.00000000e+00 0.00000000e+00
  #   0.00000000e+00 0.00000000e+00]
  #  [8.45000000e-03 5.87000000e-04 1.73333333e-05 1.83333333e-06
  #   3.33333333e-07 1.66666667e-07]
  #  [4.06141667e-02 7.93183333e-03 5.47833333e-04 8.23333333e-05
  #   1.28333333e-05 5.33333333e-06]
  #  [7.42150000e-03 7.47083333e-03 7.44666667e-03 7.81733333e-03
  #   7.35816667e-03 7.42566667e-03]]

  # Errors for kw= 64
  # [[7.98333333e-05 1.16666667e-06 0.00000000e+00 0.00000000e+00
  #   0.00000000e+00 0.00000000e+00]
  #  [2.86883333e-03 8.25000000e-05 1.33333333e-06 1.66666667e-07
  #   0.00000000e+00 0.00000000e+00]
  #  [2.07278333e-02 2.64733333e-03 8.36666667e-05 6.00000000e-06
  #   5.00000000e-07 3.33333333e-07]
  #  [2.35616667e-03 2.58966667e-03 2.56033333e-03 2.65183333e-03
  #   2.41166667e-03 2.52116667e-03]]


def computeScaledProbabilities(listOfScales=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                               listofkValues=[64, 128, 256],
                               kw=32,
                               n=1000,
                               numWorkers=8,
                               nTrials=1000,
                               ):
  # Create arguments for the possibilities we want to test
  args = []
  theta, _ = getTheta(kw)
  for ki, k in enumerate(listofkValues):
    for si, s in enumerate(listOfScales):
      args.append({
          "k": k, "kw": kw, "n": n, "theta": theta,
          "nTrials": nTrials, "inputScaling": s,
          "errorIndex": [ki, si],
          })

  result = computeMatchProbabilityParallel(args, numWorkers)

  errors = np.zeros((len(listofkValues), len(listOfScales)))
  for r in result:
    errors[r["errorIndex"][0], r["errorIndex"][1]] = r["pctMatches"]

  print("Errors using scaled inputs, for kw=", kw)
  print(errors)
  plotScaledMatches(listofkValues, listOfScales, errors,
              "images/scalar_effect_of_scale_kw" + str(kw) + ".pdf")


  # Nice errors using scaled inputs, for kw= 32 and nTrials=6000 (12,000,000
  # matches per datapoint.
  listOfScales = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
  listofkValues = [64, 128, 256]
  kw = 32
  errors = np.array([
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.50000000e-07,
         2.58333333e-06, 2.13333333e-05, 8.45833333e-05, 2.52750000e-04],
        [0.00000000e+00, 0.00000000e+00, 2.50000000e-07, 9.66666667e-06,
         7.62500000e-05, 3.23666667e-04, 9.75666667e-04, 2.22650000e-03],
        [0.00000000e+00, 6.66666667e-07, 2.89166667e-05, 2.93000000e-04,
         1.29500000e-03, 3.71925000e-03, 7.98591667e-03, 1.42828333e-02]
  ])
  plotScaledMatches(listofkValues, listOfScales, errors,
                    "images/scalar_effect_of_scale_kw" + str(kw) + ".pdf")


def computeFalseNegatives(listOfNoises=[0.5, 1.0, 1.5, 2.0, 2.5],
                          listofkValues=[64, 128, 256],
                          ):
  pass


if __name__ == '__main__':

  # computeMatchProbabilities(kw=24, nTrials=1000)
  # computeMatchProbabilities(kw=16, nTrials=3000)
  # computeMatchProbabilities(kw=32, nTrials=3000)
  # computeMatchProbabilities(kw=48, nTrials=3000)
  # computeMatchProbabilities(kw=64, nTrials=3000)
  # computeMatchProbabilities(kw=96, nTrials=3000)

  plotThetaDistribution(32)

  # computeScaledProbabilities(nTrials=6000)

  # TODO Compute false negatives