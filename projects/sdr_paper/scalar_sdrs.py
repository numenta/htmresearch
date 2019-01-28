# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
                    onlyPositive=False):
  """
  Return a randomly tensor that is initialized like a weight matrix
  Size is outputSize X inputSize, where weightSparsity% of each row is non-zero
  """
  # Initialize weights in the typical fashion.
  w = torch.Tensor(outputSize, inputSize, )
  stdv = 1. / math.sqrt(numNonzeros)
  if onlyPositive:
    w.data.uniform_(0, 2*stdv)
  else:
    w.data.uniform_(-stdv, stdv)

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


def returnMatches(k, n):

  # How many prototypes to store
  m1 = 10000
  m2 = 100

  w1 = getSparseTensor(k, n, m1)
  v1 = getSparseTensor(k, n, m2, onlyPositive=True)
  dot = v1.matmul(w1.t())

  wd = w1.matmul(w1.t())
  theta = wd.diag().min() / 2.0
  print("min/max/mean diag of w dot products", wd.diag().min(), wd.diag().max(), wd.diag().mean())
  # print("min/max/mean w dot products", wd.min(), wd.max(), wd.mean())

  # Let wd.diag().min() be a decent minimal theta for matching
  numMatches = ((dot>theta).sum()).item()
  pctMatches = numMatches / float(m1*m2)
  print("a,n", k, n, "number that match above theta", numMatches)
  print("pct matches:", pctMatches)

  # plotDot(dot,
  #         title="Histogram of overlaps 100 out of 2000",
  #         path="dot_100_2000.pdf")

  return pctMatches


def plotMatches(listofaValues, listofNValues, errors):
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
  ax.plot(listofNValues, errors[3, :], 'k:',
          label="a=256 (predicted)", marker="o", color='black')

  # ax.plot(listofNValues, errorsDense, 'k:', label="a=n/2 (predicted)", color='black')
  #
  # ax.plot(listofNValues[0:3], theoreticalErrorsA64, 'k:', label="a=64 (observed)")
  # ax.plot(listofNValues[0:9], theoreticalErrorsA128, 'k:', label="a=128 (observed)", color='black')
  # ax.plot(listofNValues, theoreticalErrorsA256, 'k:', label="a=256 (observed)")

  ax.annotate(r"$a = 20$", xy=(listofNValues[3], errors[0,3]),
              xytext=(-5, 2), textcoords="offset points", ha="right",
              color='black')
  ax.annotate(r"$a = 50$", xy=(listofNValues[3], errors[1,3]),
               ha="center",color='black')
  ax.annotate(r"$a = 75$", xy=(listofNValues[3], errors[2,3]),
               ha="center",color='black')

  plt.minorticks_off()
  plt.grid(True, alpha=0.3)

  plt.savefig("images/scalar_effect_of_n.pdf")
  plt.close()


if __name__ == '__main__':

  listofaValues = [20, 50, 75, 100]
  listofNValues = [125, 250, 500, 1000, 2000]
  errors = np.zeros((len(listofaValues), len(listofNValues)))
  for ai, a in enumerate(listofaValues):
    for ni, n in enumerate(listofNValues):
      errors[ai, ni] = returnMatches(a, n)
      print()

  print(errors)

  plotMatches(listofaValues, listofNValues, errors)
