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

"""TODO"""

import numpy as np

from htmresearch.algorithms.column_pooler import ColumnPooler
from nupic.algorithms.knn_classifier import KNNClassifier


def train(pooler, classifier, objs):
  for label, obj in objs:
    pooler.reset()

    np.random.shuffle(obj)
    for feature in obj:
      sortedFeature = np.sort(feature)
      print "A"
      poolerOutput = pooler.compute(feedforwardInput=sortedFeature,
                                    learn=True,
                                    predictedInput=sortedFeature)
      print "B"
      classifierInput = np.zeros(4096, dtype=np.int32)
      classifierInput[poolerOutput] = 1
      classifier.learn(classifierInput, label)


def test(pooler, classifier, objs):
  pass


def run():
  numObjects = 100
  objSize = 10
  allIndices = np.array(xrange(1024), dtype=np.int32)
  objs = [
      (
          label,
          [np.random.choice(allIndices, 20) for _ in xrange(objSize)]
      )
      for label in xrange(numObjects)
  ]

  pooler = ColumnPooler(
    inputWidth=1024,
  )
  classifier = KNNClassifier(k=1, distanceMethod="rawOverlap")

  train(pooler, classifier, objs)

  test(pooler, classifier, objs)


if __name__ == "__main__":
  run()
