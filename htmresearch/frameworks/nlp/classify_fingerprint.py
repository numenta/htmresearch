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
import os

from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelFingerprint(ClassificationModel):
  """
  Class to run the survey response classification task with Coritcal.io
  fingerprint encodings.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self,
               fingerprintType=EncoderTypes.word,
               unionSparsity=0.20,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               classifierMetric="rawOverlap",
               **kwargs):

    super(ClassificationModelFingerprint, self).__init__(**kwargs)

    self.classifier = KNNClassifier(k=self.numLabels,
                                    distanceMethod=classifierMetric,
                                    exact=False,
                                    verbosity=self.verbosity-1)

    # Need a valid API key for the Cortical.io encoder (see CioEncoder
    # constructor for details).
    if fingerprintType is (not EncoderTypes.document or not EncoderTypes.word):
      raise ValueError("Invalid type of fingerprint encoding; see the "
                       "EncoderTypes class for eligble types.")

    self.encoder = CioEncoder(retinaScaling=retinaScaling,
                              fingerprintType=fingerprintType,
                              unionSparsity=unionSparsity,
                              retina=retina,
                              apiKey=apiKey)

    self.currentDocument = None


  def trainToken(self, token, labels, sampleId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sampleId.

    See base class for params and return type descriptions.
    """
    if self.currentDocument is None:
      # start of a new document
      self.currentDocument = [token]
    else:
      # accumulate text for this document
      self.currentDocument.append(token)

    if reset == 1:
      # all text accumulated, proceed w/ training on this document
      document = " ".join(self.currentDocument)
      bitmap = self.encoder.encode(document)["fingerprint"]["positions"]


      if self.verbosity >= 1:
        print "CioFP model training with: '{}'".format(document)
        print "\tBitmap:", bitmap

      for label in labels:
        self.classifier.learn(bitmap, label, isSparse=self.encoder.n,
                              partitionId=sampleId)

      self.currentDocument = None


  def inferToken(self, token, reset=0, returnDetailedResults=False,
                 sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and (optionally) a list of sampleIds and
    distances.   Repeated sampleIds are NOT removed from the results.

    See base class for params and return type descriptions.
    """
    if self.currentDocument is None:
      # start of a new document
      self.currentDocument = [token]
    else:
      # accumulate text for this document
      self.currentDocument.append(token)

    if reset == 0:
      return numpy.zeros(self.numLabels), [], numpy.zeros(0)

    # With reset=1, all text accumulated, proceed w/ classifying this document
    document = " ".join(self.currentDocument)
    bitmap = self.encoder.encode(document)["fingerprint"]["positions"]

    densePattern  =self.encoder.densifyPattern(bitmap)

    (_, inferenceResult, dist, _) = self.classifier.infer(densePattern)

    if self.verbosity >= 2:
      print "CioFP model inference with: '{}'".format(document)
      print "\tBitmap:", bitmap
      print "\tInference result=", inferenceResult
      print "\tDistances=", dist

    self.currentDocument = None

    # Figure out format of returned results

    if not returnDetailedResults:
      # Return non-detailed results.
      return inferenceResult, None, None

    if not sortResults:
      idList = []
      for i in range(len(dist)):
        idList.append(self.classifier.getPartitionId(i))
      return inferenceResult, idList, dist

    # Return sorted results
    idList = []
    sortedIndices = dist.argsort()
    for i in sortedIndices:
      idList.append(self.classifier.getPartitionId(i))
    sortedDistances = dist[sortedIndices]
    return inferenceResult, idList, sortedDistances


  def getClassifier(self):
    """
    Returns the classifier instance for the model.
    """
    return self.classifier






















