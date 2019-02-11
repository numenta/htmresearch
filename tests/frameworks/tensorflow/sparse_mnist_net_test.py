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
"""
  Test Tensorflow version of SparseMNISTNet model
"""
import random
import unittest
from datetime import datetime

import numpy as np
import tensorflow as tf
from htmresearch.frameworks import tensorflow as htm
from htmresearch.frameworks.tensorflow.layers.kwinner_layer import compute_kwinners
from tensorflow import keras

# Parameters from the ExperimentQuick
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.02
MOMENTUM = 0.25
BOOST_STRENGTH = 2.0
BOOST_STRENGTH_FACTOR = 1.0
SEED = 42
N = 50
K = 10
WEIGHT_SPARSITY = 0.4
K_INFERENCE_FACTOR = 1.0
INPUT_SIZE = 28 * 28

OPTIMIZER = "Adam"
LOSS = "sparse_categorical_crossentropy"

# Tensorflow configuration.
# Make sure to use one thread in order to keep the results deterministic
CONFIG = tf.ConfigProto(
  intra_op_parallelism_threads=1,
  inter_op_parallelism_threads=1,
  device_count={'CPU': 1}
)



class SparseMNISTNetTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load MNIST dataset into tensors
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    cls.x_train = x_train.reshape(-1, INPUT_SIZE) / 255.0
    cls.x_test = x_test.reshape(-1, INPUT_SIZE) / 255.0
    cls.y_train = y_train
    cls.y_test = y_test


  # @unittest.skip("DEBUG: Enable if you need to collect baseline logs for tensorboard")
  def testMNISTBaseline(self):
    # Collect tensorboard baseline logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testMNISTBaseline/{}".format(datetime.now()),
      write_graph=True,
      write_images=True)]

    # Create Simple Dense NN as baseline
    model = keras.Sequential([
      keras.layers.Dense(N, activation=tf.nn.relu, name="l1"),
      keras.layers.Dense(10, activation=tf.nn.softmax, name="l2")
    ])

    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=['accuracy'])

    with self.test_session(config=CONFIG):
      # train
      model.fit(self.x_train, self.y_train,
                epochs=EPOCHS,
                verbose=1,
                batch_size=BATCH_SIZE,
                callbacks=callbacks)
      # test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)

      print 'Test accuracy:', accuracy, "Test loss:", loss
      self.assertAlmostEqual(accuracy, 0.9693, places=4)
      self.assertAlmostEqual(loss, 0.1037, places=4)


  def testSparseConstraint(self):
    expected = [float(round(N * WEIGHT_SPARSITY))] * BATCH_SIZE
    constraint = htm.constraints.Sparse(sparsity=WEIGHT_SPARSITY)
    with self.test_session(config=CONFIG):
      actual = constraint(tf.ones([BATCH_SIZE, N]))
      tf.global_variables_initializer().run()
      self.assertAllEqual(tf.count_nonzero(actual, axis=1).eval(), expected)


  def testComputeKwinners(self):
    x = np.float32(np.random.uniform(size=(BATCH_SIZE, N)))
    dutyCycles = np.random.uniform(size=(N,))

    # Compute k-winner using numpy
    density = float(K) / N
    boostFactors = np.exp((density - dutyCycles) * BOOST_STRENGTH)
    boosted = x * boostFactors

    # top k
    indices = np.argsort(-boosted, axis=1)[:, :K]
    expected = np.zeros_like(x)
    for i in xrange(BATCH_SIZE):
      expected[i, indices[i]] = x[i, indices[i]]

    # Compute k-winner using tensorflow
    with self.test_session(config=CONFIG):
      actual = compute_kwinners(x, K, dutyCycles, BOOST_STRENGTH)
      self.assertAllEqual(actual, expected)


  def testSparseMNISTNet(self):
    # Collect tensorboard logs
    callbacks = [keras.callbacks.TensorBoard(
      log_dir="logs/testSparseMNISTNet/{}".format(datetime.now()),
      batch_size=BATCH_SIZE,
      write_graph=True,
      write_grads=True,
      write_images=True)]

    # Keep weights sparse
    constraint = htm.constraints.Sparse(sparsity=WEIGHT_SPARSITY,
                                        name="{}_constraint".format(WEIGHT_SPARSITY))
    glorot_uniform = keras.initializers.get("glorot_uniform")
    initializer = lambda *args, **kwargs: constraint(glorot_uniform(*args, **kwargs))

    model = keras.Sequential([
      # Hidden sparse NN layer
      keras.layers.Dense(units=N, name="l1",
                         activation=tf.nn.relu,
                         kernel_initializer=initializer,
                         kernel_constraint=constraint),
      # K-Winners
      htm.layers.KWinner(k=K, kInferenceFactor=K_INFERENCE_FACTOR,
                         boostStrength=BOOST_STRENGTH,
                         boostStrengthFactor=BOOST_STRENGTH_FACTOR,
                         name="kwinner"),
      # Output NN layer
      keras.layers.Dense(units=10, activation=tf.nn.softmax,
                         name="l2")
    ])

    # Build
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
    with self.test_session(config=CONFIG):
      # Train
      model.fit(self.x_train, self.y_train, callbacks=callbacks,
                epochs=EPOCHS, batch_size=BATCH_SIZE)

      # Test
      loss, accuracy = model.evaluate(self.x_test, self.y_test,
                                      batch_size=BATCH_SIZE)
      print 'Test accuracy:', accuracy, "Test loss:", loss

      self.assertAlmostEqual(accuracy, 0.9527, places=4)
      self.assertAlmostEqual(loss, 0.1575, places=4)



if __name__ == "__main__":
  tf.test.main()
