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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras



def compute_kwinners(x, k, dutyCycles, boostStrength):
  """
  Use the boost strength to compute a boost factor for each unit represented
  in x. These factors are used to increase the impact of each unit to improve
  their chances of being chosen. This encourages participation of more columns
  in the learning process.

  The boosting function is a curve defined as: boostFactors = exp[ -
  boostStrength * (dutyCycle - targetDensity)] Intuitively this means that
  units that have been active (i.e. in the top-k) at the target activation
  level have a boost factor of 1, meaning their activity is not boosted.
  Columns whose duty cycle drops too much below that of their neighbors are
  boosted depending on how infrequently they have been active. Unit that has
  been active more than the target activation level have a boost factor below
  1, meaning their activity is suppressed and they are less likely to be in
  the top-k.

  Note that we do not transmit the boosted values. We only use boosting to
  determine the winning units.

  The target activation density for each unit is k / number of units. The
  boostFactor depends on the dutyCycle via an exponential function:

          boostFactor
              ^
              |
              |\
              | \
        1  _  |  \
              |    _
              |      _ _
              |          _ _ _ _
              +--------------------> dutyCycle
                 |
            targetDensity

  :param x:
    Current activity of each unit.

  :param k:
    The activity of the top k units will be allowed to remain, the rest are
    set to zero.

  :param dutyCycles:
    The averaged duty cycle of each unit.

  :param boostStrength:
    A boost strength of 0.0 has no effect on x.

  :return:
    A tensor representing the activity of x after k-winner take all.
  """

  k = tf.convert_to_tensor(k, dtype=tf.int32)
  boostStrength = tf.math.maximum(boostStrength, 0.0, name="boostStrength")
  targetDensity = tf.cast(k / x.shape[1], tf.float32, name="targetDensity")
  boostFactors = tf.exp((targetDensity - dutyCycles) * boostStrength,
                        name="boostFactors")
  boosted = tf.multiply(x, boostFactors, name="boosted")

  # Take the boosted version of the input x, find the top k winners.
  # Compute an output that contains the values of x corresponding to the top k
  # boosted values
  topk, _ = tf.math.top_k(input=boosted, k=k, sorted=False,
                          name="compute_kwinners")
  bottom = tf.reduce_min(topk, axis=1, keepdims=True,
                         name="compute_kwinners")
  mask = tf.cast(tf.greater_equal(boosted, bottom), dtype=x.dtype,
                 name="compute_kwinners")

  return x * mask



class KWinner(keras.layers.Layer):
  """
    K-winner activation layer.

  :param k:
    The activity of the top k units will be allowed to remain, the rest are set
    to zero
  :type k: int

  :param kInferenceFactor:
    During inference (training=False) we increase k by this factor.
  :type kInferenceFactor: float

  :param boostStrength:
    boost strength (0.0 implies no boosting).
  :type boostStrength: float

  :param boostStrengthFactor:
    boost strength is multiplied by this factor after each epoch
    A value < 1.0 will decrement it every epoch
  :type boostStrength: float

  :param dutyCyclePeriod:
    The period used to calculate duty cycles
  :type dutyCyclePeriod: int
  """


  def __init__(self, k, kInferenceFactor=1.0,
               boostStrength=1.0, boostStrengthFactor=1.0,
               dutyCyclePeriod=1000.0, **kwargs):
    super(KWinner, self).__init__(**kwargs)
    assert (boostStrength >= 0.0)

    self.k = k
    # N is not known until the model is built with an input shape
    self.n = k

    self.kInferenceFactor = kInferenceFactor
    self.learningIterations = 0.0

    # Boosting related parameters
    self.boostStrength = boostStrength
    self.boostStrengthFactor = boostStrengthFactor
    self.dutyCyclePeriod = dutyCyclePeriod
    self.dutyCycles = None


  def build(self, input_shape):
    # Update N value with input shape
    self.n = input_shape[-1].value
    assert (self.k <= self.n)

    self.dutyCycles = self.add_variable(name="dutyCycles",
                                        shape=[self.n],
                                        initializer=tf.zeros_initializer,
                                        trainable=False)
    super(KWinner, self).build(input_shape)


  def _updateDutyCycles(self, inputs):
    batchSize = tf.cast(tf.shape(inputs)[0], tf.float32)
    self.learningIterations = self.learningIterations + batchSize

    # TODO: Add condition for if k != self.n
    period = tf.minimum(self.dutyCyclePeriod, self.learningIterations)
    newInputs = tf.reduce_sum(tf.clip_by_value(inputs, 0, inputs.dtype.max), axis=0)
    return (self.dutyCycles * period - batchSize * newInputs) / period


  def call(self, inputs, training=None):
    # if not self.training:
    #   k = min(int(round(self.k * self.kInferenceFactor)), self.n)
    # else:
    #   k = self.k
    k = keras.backend.in_test_phase(
      tf.minimum(tf.cast(
        tf.math.round(self.k * self.kInferenceFactor), tf.int32, name="k_mul_factor"),
        self.n, name="min_k_n"),
      self.k)

    # Apply k-winner algorithm if k < n, otherwise default to standard RELU
    x = tf.cond(tf.math.not_equal(k, self.n),
                lambda: compute_kwinners(inputs, k=k,
                                         dutyCycles=self.dutyCycles,
                                         boostStrength=self.boostStrength),
                lambda: tf.nn.relu(inputs))

    # Update moving average of duty cycle for training iterations only
    # During inference this is kept static.
    self.dutyCycles = keras.backend.in_train_phase(
      self._updateDutyCycles(x),
      self.dutyCycles)

    return x


  def get_config(self):
    config = {
      "k": self.k,
      "kInferenceFactor": self.kInferenceFactor,
      "boostStrength": self.boostStrength,
      "boostStrengthFactor": self.boostStrengthFactor,
      "dutyCyclePeriod": self.dutyCyclePeriod
    }
    config.update(super(KWinner, self).get_config())
    return config


  def compute_output_shape(self, input_shape):
    return input_shape
