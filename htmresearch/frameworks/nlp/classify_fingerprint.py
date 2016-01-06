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
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelFingerprint",
               fingerprintType=EncoderTypes.word,
               unionSparsity=0.20,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               classifierMetric="rawOverlap",
               cacheRoot=None):

    super(ClassificationModelFingerprint, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    self.classifier = KNNClassifier(k=numLabels,
                                    distanceMethod=classifierMetric,
                                    exact=False,
                                    verbosity=verbosity-1)

    # Need a valid API key for the Cortical.io encoder (see CioEncoder
    # constructor for details).
    if fingerprintType is (not EncoderTypes.document or not EncoderTypes.word):
      raise ValueError("Invaid type of fingerprint encoding; see the "
                       "EncoderTypes class for eligble types.")

    cacheRoot = cacheRoot or os.path.dirname(os.path.realpath(__file__))

    self.encoder = CioEncoder(retinaScaling=retinaScaling,
                              cacheDir=os.path.join(cacheRoot, "CioCache"),
                              fingerprintType=fingerprintType,
                              unionSparsity=unionSparsity,
                              retina=retina,
                              apiKey=apiKey)

    self.currentDocument = None


  def encodeSample(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API. If the
    client returns None, we create a random SDR with the model's dimensions n
    and w.

    @param sample     (list)        Tokenized sample, where each item is a str.
    @return fp        (dict)        The sample text, sparsity, and bitmap.
    Example return dict:
      {
        "text": "Example text",
        "sparsity": 0.03,
        "bitmap": numpy.array([])
      }
    """
    sample = " ".join(sample)
    fpInfo = self.encoder.encode(sample)
    if fpInfo:
      fp = {"text":fpInfo["text"] if "text" in fpInfo else fpInfo["term"],
            "sparsity":fpInfo["sparsity"],
            "bitmap":numpy.array(fpInfo["fingerprint"]["positions"])}
    else:
      fp = {"text":sample,
            "sparsity":float(self.encoder.w)/self.encoder.n,
            "bitmap":self.encodeRandomly(
              sample, self.encoder.n, self.encoder.w)}

    return fp


  def trainModel(self, i):
    # TODO: add batch training, where i is a list
    """
    Train the classifier on the sample and labels for record i. The list
    sampleReference is populated to correlate classifier prototypes to sample
    IDs.
    """
    bitmap = self.patterns[i]["pattern"]["bitmap"]
    count = 0
    if bitmap.any():
      for count, label in enumerate(self.patterns[i]["labels"]):
        self.classifier.learn(bitmap, label, isSparse=self.encoder.n)
        self.sampleReference.append(self.patterns[i]["ID"])
      count += 1

    return count


  def testModel(self, i, seed=42):
    """
    Test the model on record i. The random seed is used in getWinningLabels().

    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    (_, inferenceResult, _, _) = self.classifier.infer(self.sparsifyPattern(
      self.patterns[i]["pattern"]["bitmap"], self.encoder.n))
    return self.getWinningLabels(inferenceResult, seed)


  def trainText(self, token, labels, sequenceId=None, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID. The sequence ID is stored in sampleReference so we know which
    samples the model has been trained on, and specifically where they
    appear in the classifier space.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token. If the list is empty, the
                             classifier will not be trained.
    @param sequenceId (int)  An integer ID associated with this token and its
                             sequence (document).
    @param reset      (int)  Should be 0 or 1. If 1, assumes we are at the
                             beginning of a new sequence.
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
        self.classifier.learn(bitmap, label, isSparse=self.encoder.n)
        self.sampleReference.append(sequenceId)

        # TODO: replace the need for sampleReference w/ partitionId.
        # There is a bug in how partitionId is handled during infer if it is
        # not passed in, so we won't pass it in for now (line 863 of
        # KNNClassifier.py)
        # self.classifier.learn(bitmap, label, isSparse=self.n,
        #                       partitionId=sequenceId)

      self.currentDocument = None


  def classifyText(self, token, reset=0, seed=42):
    """
    Classify the token

    @param token    (str)  The text token to train on
    @param reset    (int)  Should be 0 or 1. If 1, assumes we are at the
                           beginning of a new sequence.
    @param seed     (int)  Random seed used for deciding ties in
                           getWinnningLabels().

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this sample belongs to the
                           i'th category. An array containing all zeros
                           implies no decision could be made.
    """
    if self.currentDocument is None:
      # start of a new document
      self.currentDocument = [token]
    else:
      # accumulate text for this document
      self.currentDocument.append(token)

    if reset == 1:
      # all text accumulated, proceed w/ classifying this document
      document = " ".join(self.currentDocument)
      bitmap = self.encoder.encode(document)["fingerprint"]["positions"]

      densePattern  =self.sparsifyPattern(bitmap, self.encoder.n)

      (_, inferenceResult, _, _) = self.classifier.infer(densePattern)

      winningLabels = self.getWinningLabels(inferenceResult, seed)

      if self.verbosity >= 1:
        print "CioFP model inference with: '{}'".format(document)
        print "\tBitmap:", bitmap
        print "\tInference result=", inferenceResult
        print "\tWinning labels=", winningLabels

      self.currentDocument = None

      return winningLabels

    else:
      return numpy.zeros(self.numLabels)
