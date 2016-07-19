# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""
This experiment uses ImageSparseNet as well as data from Bruno Olshausen's lab
(part of his Neural Computation course), to train an ImageSparseNet for sparse
coding on natural images.

The ImageSparseNet is trained on a few images of natural scences, using
random patches of the encoder's expected input dimension.
"""

from htmresearch.algorithms.image_sparse_net import ImageSparseNet


DEFAULT_SPARSENET_PARAMS = {
  "inputDim" : 64,
  "outputDim" : 64,
  "batchSize" : 100,
  "numIterations" : 1000,
  "numLcaIterations" : 75,
  "learningRate" : 2.0,
  "decayCycle" : 100,
  "learningRateDecay" : 1.0,
  "lcaLearningRate" : 0.1,
  "thresholdDecay" : 0.95,
  "minThreshold" : 0.1,
  "thresholdType" : 'soft',
  "verbosity" : 0,  # can be changed to print training loss
  "seed" : 42
}

DATA_PATH = "data/IMAGES.mat"
DATA_NAME = "IMAGES"

LOSS_HISTORY_PATH = "output/loss_history.png"
BASIS_FUNCTIONS_PATH = "output/basis_functions.png"


def runExperiment():
  print "Creating network..."
  network = ImageSparseNet(**DEFAULT_SPARSENET_PARAMS)

  print "Loading training data..."
  images = network.loadMatlabImages(DATA_PATH, DATA_NAME)

  print "Training {0}...".format(network)
  network.train(images)

  print "Saving loss history and function basis..."
  network.plotLoss(filename=LOSS_HISTORY_PATH)
  network.plotBasis(filename=BASIS_FUNCTIONS_PATH)


if __name__ == "__main__":
  runExperiment()