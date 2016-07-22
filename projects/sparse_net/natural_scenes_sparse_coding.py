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

The ImageSparseNet is trained on a few images of natural scenes, using
random patches of the encoder's expected input dimension.
"""

try:
  import capnp
except ImportError:
  capnp = None

from htmresearch.algorithms.image_sparse_net import ImageSparseNet

if capnp:
  from htmresearch.algorithms.sparse_net_capnp import SparseNetProto


DEFAULT_SPARSENET_PARAMS = {
  "filterDim" : 64,
  "outputDim" : 64,
  "batchSize" : 100,
  "numLcaIterations" : 75,
  "learningRate" : 2.0,
  "decayCycle" : 100,
  "learningRateDecay" : 1.0,
  "lcaLearningRate" : 0.1,
  "thresholdDecay" : 0.95,
  "minThreshold" : 1.0,
  "thresholdType" : 'soft',
  "verbosity" : 0,  # can be changed to print training loss
  "showEvery" : 500,
  "seed" : 42,
}

DATA_PATH = "data/IMAGES.mat"
DATA_NAME = "IMAGES"

LOSS_HISTORY_PATH = "output/loss_history.png"
BASIS_FUNCTIONS_PATH = "output/basis_functions.png"

SERIALIZATION_PATH = "output/model.txt"
LOSS_HISTORY_PATH2 = "output/loss_history_2.png"
BASIS_FUNCTIONS_PATH2 = "output/basis_functions_2.png"


def runExperiment():
  print "Creating network..."
  network = ImageSparseNet(**DEFAULT_SPARSENET_PARAMS)

  print "Loading training data..."
  images = network.loadMatlabImages(DATA_PATH, DATA_NAME)
  print

  print "Training {0}...".format(network)
  network.train(images, numIterations=5000)

  print "Saving loss history and function basis..."
  network.plotLoss(filename=LOSS_HISTORY_PATH)
  network.plotBasis(filename=BASIS_FUNCTIONS_PATH)

  if capnp:
    print
    print "Saving model..."
    proto1 = SparseNetProto.new_message()
    network.write(proto1)
    with open(SERIALIZATION_PATH, 'wb') as f:
      proto1.write(f)

    print "Loading model..."
    with open(SERIALIZATION_PATH, 'rb') as f:
      proto2 = SparseNetProto.read(f)
      newNetwork = ImageSparseNet.read(proto2)

      print "Checking that loaded model is the same as before..."
      if newNetwork != network:
        raise ValueError("Model is different!")
      else:
        print "Model is the same."

      print
      print "Training {0} again...".format(network)
      newNetwork.train(images, numIterations=5000)

      print "Saving loss history and function basis..."
      newNetwork.plotLoss(filename=LOSS_HISTORY_PATH2)
      newNetwork.plotBasis(filename=BASIS_FUNCTIONS_PATH2)



if __name__ == "__main__":
  runExperiment()
