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
  m = 10000

  w1 = getSparseTensor(k, n, m)
  v1 = getSparseTensor(k, n, 100, onlyPositive=True)
  dot = v1.matmul(w1.t())

  wd = w1.matmul(w1.t())
  theta = wd.diag().min() / 2.0
  print("min/max/mean diag of w dot products", wd.diag().min(), wd.diag().max(), wd.diag().mean())
  # print("min/max/mean w dot products", wd.min(), wd.max(), wd.mean())

  # Let wd.diag().min() be a decent minimal theta for matching
  print("k,n", k, n, "number that match above theta", (dot>theta).sum())

  # plotDot(dot,
  #         title="Histogram of overlaps 100 out of 2000",
  #         path="dot_100_2000.pdf")



if __name__ == '__main__':

  for k in [20, 50, 75, 100]:
    for n in [125, 250, 500, 1000, 2000]:
      returnMatches(k, n)
      print()

