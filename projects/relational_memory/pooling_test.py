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

import collections

import numpy as np

from htmresearch.algorithms.union_temporal_pooler import UnionTemporalPooler
from nupic.algorithms.knn_classifier import KNNClassifier


def train(pooler, classifier, objs, numPasses):
  for label, obj in objs:
    pooler.reset()

    for _ in xrange(numPasses):
      np.random.shuffle(obj)
      for feature in obj:
        denseFeature = np.zeros((1024,), dtype=np.uint32)
        denseFeature[feature] = 1
        poolerOutput = pooler.compute(denseFeature,
                                      denseFeature,
                                      learn=True)

        classifierInput = np.zeros((1024,), dtype=np.uint32)
        classifierInput[poolerOutput] = 1
        classifier.learn(classifierInput, label)


def test(pooler, classifier, objs):
  total = 0
  correct = 0
  for label, obj in objs:
    pooler.reset()

    np.random.shuffle(obj)
    classifierGuesses = collections.defaultdict(int)
    for feature in obj:
      denseFeature = np.zeros((1024,), dtype=np.uint32)
      denseFeature[feature] = 1
      poolerOutput = pooler.compute(denseFeature,
                                    denseFeature,
                                    learn=False)

      classifierInput = np.zeros((1024,), dtype=np.uint32)
      classifierInput[poolerOutput] = 1
      classifierResult = classifier.infer(classifierInput)

      classifierGuesses[classifierResult[0]] += 1

    #bestGuess = sorted(classifierGuesses.iteritems(), key=lambda x: x[1])[-1][0]
    bestGuess = classifierResult[0]
    if bestGuess == label:
      correct += 1
    total += 1
  return float(correct) / float(total)


def run():
  numObjects = 10
  objSize = 10
  numPasses = 2

  allIndices = np.array(xrange(1024), dtype=np.uint32)
  objs = [
      (
          label,
          [[int(i) for i in np.random.choice(allIndices, 20)] for _ in xrange(objSize)]
      )
      for label in xrange(numObjects)
  ]

  pooler = UnionTemporalPooler(
    inputDimensions=(1024,),
    columnDimensions=(1024,),
    potentialRadius=1024,
    potentialPct=0.8,
    globalInhibition=True,
    numActiveColumnsPerInhArea=20.0,
    #boostStrength=10.0,
    #dutyCyclePeriod=50,
  )
  classifier = KNNClassifier(k=1, distanceMethod="rawOverlap")

  train(pooler, classifier, objs, numPasses)

  result = test(pooler, classifier, objs)
  print result


if __name__ == "__main__":
  run()
