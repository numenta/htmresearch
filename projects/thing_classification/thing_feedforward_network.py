# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
This file is used to run Thing experiments using simulated sensations with
a simple logistic encoder, with/without location signal


Example usage

Train network with (feature, location) as input and a spatial pooler
The spatial pooler speeds up convergence
python thing_feedforward_network.py --spatial_pooler 1 --location 1

Train network with only feature input
python thing_feedforward_network.py --spatial_pooler 0 --location 0
"""

from optparse import OptionParser
import numpy as np
from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
import tensorflow as tf
from thing_convergence import loadThingObjects


def createSpatialPooler(inputWidth):
  spParam = {
    "inputDimensions": (inputWidth, ),
    "columnDimensions": (inputWidth, ),
    "potentialRadius": inputWidth,
    "potentialPct": 0.5,
    "globalInhibition": True,
    "localAreaDensity": .02,
    "numActiveColumnsPerInhArea": -1,
    "stimulusThreshold": 1,
    "synPermInactiveDec": 0.004,
    "synPermActiveInc": 0.02,
    "synPermConnected": 0.5,
    "minPctOverlapDutyCycle": 0.0,
    "dutyCyclePeriod": 1000,
    "boostStrength": 0.0,
    "seed": 1936
  }
  print "use spatial pooler to encode feature location pairs "
  print " initializing spatial pooler ... "
  sp = CPPSpatialPooler(**spParam)
  return sp



def _getArgs():
  parser = OptionParser(usage="Train HTM Spatial Pooler")

  parser.add_option("-l",
                    "--location",
                    type=int,
                    default=1,
                    dest="useLocation",
                    help="Whether to use location signal")

  parser.add_option("--spatial_pooler",
                    type=int,
                    default=1,
                    dest="useSP",
                    help="Whether to use spatial pooler")

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder


if __name__ == "__main__":
  (expConfig, _args) = _getArgs()

  objects, OnBitsList = loadThingObjects(1, './data')
  objects = objects.provideObjectsToLearn()
  objectNames = objects.keys()
  numObjs = len(objectNames)
  featureWidth = 256
  locationWidth = 1024
  useLocation = expConfig.useLocation
  useSpatialPooler = expConfig.useSP

  numInputVectors = 0
  for i in range(numObjs):
    numInputVectors += len(objects[objectNames[i]])
  if useLocation:
    inputWidth = featureWidth + locationWidth
  else:
    inputWidth = featureWidth
  data = np.zeros((numInputVectors, inputWidth))
  label = np.zeros((numInputVectors, numObjs))


  if useSpatialPooler:
    sp = createSpatialPooler(inputWidth)
  else:
    sp = None

  k = 0
  for i in range(numObjs):
    print "converting object {} ...".format(i)
    numSenses = len( objects[objectNames[i]])
    for j in range(numSenses):
      activeBitsLocation = np.array(list(objects[objectNames[i]][j][0][0]))
      activeBitsFeature = np.array(list(objects[objectNames[i]][j][0][1]))

      data[k, activeBitsFeature] = 1
      if useLocation:
        data[k, featureWidth+activeBitsLocation] = 1
      label[k, i] = 1

      if useSpatialPooler:
        inputVector = data[k, :]
        outputColumns = np.zeros((inputWidth, ))
        sp.compute(inputVector, False, outputColumns)
        activeBits = np.where(outputColumns)[0]
        data[k, activeBits] = 1

      k += 1

  print "run logistic regression ... "
  x = tf.placeholder(tf.float32, [None, inputWidth])
  W = tf.Variable(tf.zeros([inputWidth, numObjs]))
  b = tf.Variable(tf.zeros([numObjs]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)

  y_ = tf.placeholder(tf.float32, [None, numObjs])
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  for i in range(100):
    sess.run(train_step, feed_dict={x: data, y_: label})
    if i%10 == 0:
      print("iteration {} accuracy = {}".format(i,
      sess.run(accuracy, feed_dict={x: data, y_: label})))

  print "final recognition accuracy: ", sess.run(accuracy, feed_dict={x: data, y_: label})

  predicted_label = tf.argmax(y, 1)
  ypred = sess.run(predicted_label, feed_dict={x: data})

  correct = 0
  k = 0
  for i in range(numObjs):
    numSenses = len(objects[objectNames[i]])
    label = np.zeros((numSenses, ))
    for j in range(numSenses):
      label[j] = ypred[k]
      k += 1

    label = label.astype('int32')
    counts = np.bincount(label)
    correct += (np.argmax(counts)==i)
  print " accuracy after majority voting {}".format(float(correct)/numObjs)




