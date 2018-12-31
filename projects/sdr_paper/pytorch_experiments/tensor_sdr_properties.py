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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision import datasets, transforms


def getSparseWeights(weightSparsity, inputSize, outputSize):
  """
  Return a randomly initialized weight matrix
  Size is outputSize X inputSize, with sparsity weightSparsity%
  """
  # Initialize weights in the typical fashion.
  w = torch.Tensor(outputSize, inputSize)
  stdv = 1. / math.sqrt(w.size(1))
  w.data.uniform_(-stdv, stdv)

  numZeros = int(round((1.0 - weightSparsity) * inputSize))

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


def plotOverlapHistogram(v, w, title, base="random"):
  """
  Given a vector v, compute the overlap with the weight matrix w and save
  the histogram of overlaps.
  """
  overlaps = v.matmul(w.t())

  # Plot histogram of overlaps
  bins = np.linspace(float(overlaps.min()), float(overlaps.max()), 28)
  plt.hist(overlaps, bins, alpha=0.5, label='All cols')
  plt.legend(loc='upper right')
  plt.xlabel("Overlap scores")
  plt.ylabel("Frequency")
  plt.title(title)
  plt.savefig(base+"_1")
  plt.close()

  return overlaps


def plotOverlaps(vList, w, base="random", k=20):
  """
  Given a list of vectors v, compute the overlap of each with the weight matrix
  w and plot the overlap curves.
  """
  for i,v in enumerate(vList):
    if i==0:
      col = "m"
      label = "Random vector"
    else:
      col="c"
      label = ""
      if i==1: label="Test images"
    # Get a sorted list of overlap values, in decreasing order
    overlaps = v.matmul(w.t())
    sortedOverlaps = overlaps.sort()[0].tolist()[0][::-1]
    plt.plot(sortedOverlaps,col,label=label)

  plt.axvspan(0, k, facecolor="g", alpha=0.3, label="Active units")
  plt.legend(loc="upper right")
  plt.xlabel("Units")
  plt.ylabel("Overlap scores")
  plt.title("Sorted unit overlaps of a sparse net.")
  plt.savefig(base+"_2")
  plt.close()


def analyzeModelOverlaps(modelName):
  model = torch.load(modelName)
  model.eval()
  inputSize = model.l1.weight.data.shape[1]
  outputSize = model.l1.weight.data.shape[0]

  v = torch.Tensor(1, inputSize)
  v.data.uniform_(-1.0, 1.0)

  plotOverlapHistogram(v, model.l1.weight.data,
                       title="Overlap histogram of model with random vector",
                       base="model23_random")

  dutyCycle = model.dutyCycle.numpy()
  dutyCycleSortedIndices = dutyCycle.argsort()[::-1]

  kwargs = {}
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=True, **kwargs)
  iterator = test_loader.__iter__()
  v1 = next(iterator)[0].view(1, 28*28)
  vectors = [
    v,
    v1,
    next(iterator)[0].view(1, 28 * 28),
    next(iterator)[0].view(1, 28 * 28),
    next(iterator)[0].view(1, 28 * 28),
    next(iterator)[0].view(1, 28 * 28),
    next(iterator)[0].view(1, 28 * 28),
    next(iterator)[0].view(1, 28 * 28),
    next(iterator)[0].view(1, 28 * 28)
  ]

  plotOverlapHistogram(v1, model.l1.weight.data,
                       title="Overlap histogram of model with test vector",
                       base="model23_mnist")
  plotOverlaps(vectors, model.l1.weight.data, base="model23_mnist", k=model.k)


def analyzeWeightDistribution(weights,base):
  """Plot histogram of non-zero weight values."""
  weights = weights.numpy()
  weights = weights.reshape(28*28*500)
  bins = np.linspace(-0.1, 0.1, 50)
  nz = weights.nonzero()[0]
  nzw = weights[nz]
  print("zero before and after:", (weights==0).sum(), (nzw==0).sum())
  print("Small weights:", (abs(nzw)>0.0).sum(), (abs(nzw)<0.01).sum(), (abs(nzw)<0.02).sum(), (abs(nzw)<0.03).sum())
  plt.hist(nzw, bins, alpha=0.5, label='All cols')
  plt.legend(loc='upper right')
  plt.xlabel("Weight values")
  plt.ylabel("Frequency")
  plt.title("Histogram of non-zero weight values")
  plt.savefig(base+"_weights")
  plt.close()


def analyzeModelWeightDistribution(modelName,base):
  """Plot histogram of non-zero weight values."""
  model = torch.load(modelName)
  model.eval()
  analyzeWeightDistribution(model.l1.weight.data, base)



def analyzeModelNoiseRobustness(modelName):
  """TODO"""
  model = torch.load(modelName)
  model.eval()


if __name__ == '__main__':

  inputSize = 28*28
  outputSize = 500
  w = getSparseWeights(0.4, inputSize, outputSize)

  v = torch.Tensor(1, inputSize)
  v.data.uniform_(-1.0, 1.0)

  analyzeWeightDistribution(w,base="untrained")
  overlaps = plotOverlapHistogram(v,w,
                                  title="Overlap histogram for untrained net.",
                                  base="untrained")
  plotOverlaps([v], w, k=50)


  # modelName = "results/experiment23/k_inference_factor2.0boost_strength_factor0.90learning_rate0.040batch_size4.0n500.0boost_strength1.0k50.0/model.pt"
  modelName = "results/exp31/boost_strength1.0k50.0n500.0/model.pt"
  analyzeModelOverlaps(modelName)

  analyzeModelWeightDistribution(modelName, base="model31")
