# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
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
  # TODO: use nupic.bindings.math import Random?

  def __init__(self,
               n=100,
               w=20,
               verbosity=1,
               classifierMetric="rawOverlap",
               k=None,
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


  def trainToken(self, token, labels, sampleId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sampleId.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param sampleId   (int)  An integer ID associated with this token.
    @param reset      (int)  Ignored.

    """
    bitmap = self._encodeToken(token)
    if self.verbosity >= 2:
      print "Keywords training with:",token
      print "labels=",labels
      print "  bitmap:",bitmap
    for label in labels:
      self.classifier.learn(bitmap, label, isSparse=self.n,
                            partitionId=sampleId)


  def inferToken(self, token, reset=0, returnDetailedResults=False,
                 sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and a list of sampleIds and distances.
    Repeated sampleIds are NOT removed from the results.

    @param token    (str)     The text token to train on
    @param reset    (int)     Ignored
    @param sortResults (bool) If true the list of sampleIds and distances
                              will be sorted in order of increasing distances.

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this token belongs to the
                           i'th category. An array containing all zeros
                           implies no decision could be made.
             (list)        A list of sampleIds
             (numpy array) An array of distances from each stored sample
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

    if returnDetailedResults:
      # Accumulate the ids. Sort results if requested
      if sortResults:
        idList = []
        sortedIndices = dist.argsort()
        for i in sortedIndices:
          idList.append(self.classifier.getPartitionId(i))
        sortedDistances = dist[sortedIndices]
        return inferenceResult, idList, sortedDistances

      else:
        idList = []
        for i in range(len(dist)):
          idList.append(self.classifier.getPartitionId(i))
        return inferenceResult, idList, dist

    else:
        # Return non-detailed results.
        return inferenceResult, None, None

  def _encodeToken(self, token):
    """
    Randomly encode an SDR of the input token. We seed the random number
    generator such that a given string will yield the same SDR each time this
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
