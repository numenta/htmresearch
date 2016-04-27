# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

import numpy
import random

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier


class ClassificationModelKeywords(ClassificationModel):
  """
  Class to run NLP classification task with random SDRs.
  """

  def __init__(self,
               n=100,
               w=20,
               verbosity=1,
               classifierMetric="rawOverlap",
               k=1,
               **kwargs
               ):

    super(ClassificationModelKeywords, self).__init__(**kwargs)

    self.classifier = KNNClassifier(exact=True,
                                    distanceMethod=classifierMetric,
                                    k=k,
                                    verbosity=verbosity-1)

    self.n = n
    self.w = w


  def getClassifier(self):
    """
    Returns the classifier instance for the model.
    """
    return self.classifier


  def trainToken(self, token, labels, tokenId, resetSequence=0):
    """
    Train the model with the given text token, associated labels, and ID
    associated with this token.

    See base class for description of parameters.
    """
    bitmap = self._encodeToken(token)
    if self.verbosity >= 2:
      print "Keywords training with:",token
      print "labels=",labels
      print "  bitmap:",bitmap
    for label in labels:
      self.classifier.learn(bitmap,
                            label,
                            isSparse=self.n,
                            partitionId=tokenId)


  def inferToken(self, token, resetSequence=0, returnDetailedResults=False,
                 sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and a list of sampleIds and distances.
    Repeated sampleIds are NOT removed from the results.

    See base class for description of parameters.
    """
    bitmap = self._encodeToken(token)
    densePattern = self._densifyPattern(bitmap, self.n)
    if self.verbosity >= 2:
      print "Inference with token=",token,"bitmap=", bitmap
      print "Dense version:", densePattern

    (_, inferenceResult, dist, _) = self.classifier.infer(densePattern)
    if self.verbosity >= 2:
      print "Inference result=", inferenceResult
      print "Distances=", dist

    if not returnDetailedResults:
      return inferenceResult, None, None

    # Accumulate the ids. Sort results if requested
    if sortResults:
      sortedIndices = dist.argsort()
      idList = [self.classifier.getPartitionId(i) for i in sortedIndices]
      sortedDistances = dist[sortedIndices]
      return inferenceResult, idList, sortedDistances

    else:
      # Unsorted results
      idList = [self.classifier.getPartitionId(i) for i in xrange(len(dist))]
      return inferenceResult, idList, dist


  def _encodeToken(self, token):
    """
    Randomly encode an SDR of the input token. We seed the random number
    generator such that a given string will return the same SDR each time this
    method is called.

    @param token  (str)      String token
    @return       (list)     Numpy arrays, each with a bitmap of the
                             encoding.
    """
    random.seed(token)
    return numpy.sort(random.sample(xrange(self.n), self.w))


  def _densifyPattern(self, bitmap, n):
    """Return a numpy array of 0s and 1s to represent the input bitmap."""
    sparsePattern = numpy.zeros(n)
    for i in bitmap:
      sparsePattern[i] = 1.0
    return sparsePattern
