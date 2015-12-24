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

import copy
import numpy
import os

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier

import simplejson as json



class ClassificationModelKeywords(ClassificationModel):
  """
  Class to run the survey response classification task with random SDRs.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """
  # TODO: use nupic.bindings.math import Random

  def __init__(self,
               n=100,
               w=20,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelKeywords",
               classifierMetric="rawOverlap",
               k=None,
               ):

    super(ClassificationModelKeywords, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    # Backward compatibility to support previous odd behavior
    if k == None:
      k = numLabels

    # We use the pctOverlapOfInput distance metric for this model so the
    # queryModel() output is consistent (i.e. 0.0-1.0). The KNN classifications
    # aren't affected b/c the raw overlap distance is still used under the hood.
    self.classifier = KNNClassifier(exact=True,
                                    distanceMethod=classifierMetric,
                                    k=k,
                                    verbosity=verbosity-1)

    self.n = n
    self.w = w


  def encodeToken(self, token):
    """
    Randomly encode an SDR of the input token. We seed the random number
    generator such that a given string will yield the same SDR each time this
    method is called.

    @param token  (str)      String token
    @return       (list)     Numpy arrays, each with a bitmap of the
                                        encoding.
    """
    return self.encodeRandomly(token, self.n, self.w)


  def encodeSample(self, sample):
    """
    Randomly encode an SDR of the input strings. We seed the random number
    generator such that a given string will yield the same SDR each time this
    method is called.

    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (list)            Numpy arrays, each with a bitmap of the
                                        encoding.
    """
    patterns = []
    for token in sample:
      patterns.append({"text":token,
                       "sparsity":float(self.w)/self.n,
                       "bitmap":self.encodeToken(token)})
    return patterns


  def writeOutEncodings(self):
    """
    Log the encoding dictionaries to a txt file; overrides the superclass
    implementation.
    """
    if not os.path.isdir(self.modelDir):
      raise ValueError("Invalid path to write file.")

    # Cast numpy arrays to list objects for serialization.
    jsonPatterns = copy.deepcopy(self.patterns)
    for jp in jsonPatterns:
      for tokenPattern in jp["pattern"]:
        tokenPattern["bitmap"] = tokenPattern.get(
          "bitmap", numpy.array([])).tolist()
      jp["labels"] = jp.get("labels", numpy.array([])).tolist()

    with open(os.path.join(self.modelDir, "encoding_log.txt"), "w") as f:
      f.write(json.dumps(jsonPatterns, indent=1))


  def trainModel(self, i):
    # TODO: add batch training, where i is a list
    """
    Train the classifier on the sample and labels for record i. The list
    sampleReference is populated to correlate classifier prototypes to sample
    IDs. This model is unique in that a single sample contains multiple encoded
    patterns.
    """
    count = 0
    for token in self.patterns[i]["pattern"]:
      if token["bitmap"].any():
        for label in self.patterns[i]["labels"]:
          self.classifier.learn(token["bitmap"], label, isSparse=self.n)
          self.sampleReference.append(self.patterns[i]["ID"])
          count += 1
  
    return count


  def testModel(self, i, seed=42):
    """
    Test the model on record i.  Returns the classifications most frequent 
    amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classifications among those that are detected.
    The random seed is used in getWinningLabels().

    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    totalInferenceResult = None
    for pattern in self.patterns[i]["pattern"]:
      if not pattern:
        continue

      (_, inferenceResult, _, _) = self.classifier.infer(
        self.sparsifyPattern(pattern["bitmap"], self.n))

      if totalInferenceResult is None:
        totalInferenceResult = inferenceResult
      else:
        totalInferenceResult += inferenceResult

    return self.getWinningLabels(totalInferenceResult, seed)


  def infer(self, patterns):
    """
    Get the classifier output for a single input pattern; assumes classifier
    has an infer() method (as specified in NuPIC kNN implementation). For this
    model we sum the distances across the patterns and normalize
    before returning.
    @return       (numpy.array)       Each entry is the distance from the
        input pattern to that prototype (pattern in the classifier). All
        distances are between 0.0 and 1.0
    """
    # TODO: implement getNumPatterns() method in classifier.
    distances = numpy.zeros((self.classifier._numPatterns))
    for i, p in enumerate(patterns):
      (_, _, dist, _) = self.classifier.infer(
        self.sparsifyPattern(p["bitmap"], self.n))

      distances = numpy.array([sum(x) for x in zip(dist, distances)])

    return distances / (i+1)


  def trainText(self, token, labels, sequenceId=None, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token. If the list is empty, the
                             classifier will not be trained.
    @param sequenceId (int)  An integer ID associated with this token and its
                             sequence (document).
    @param reset      (int)  Ignored.
    """
    bitmap = self.encodeToken(token)
    if self.verbosity >= 1:
      print "Keywords training with:",token
      print "  bitmap:",bitmap
    for label in labels:
      self.classifier.learn(bitmap, label, isSparse=self.n)

      # There is a bug in how partitionId is handled during infer if it is
      # not passed in, so we won't pass it in for now (line 863 of
      # KNNClassifier.py)
      # self.classifier.learn(bitmap, label, isSparse=self.n,
      #                       partitionId=sequenceId)


  def classifyText(self, token, reset=0):
    """
    Classify the token

    @param token    (str)  The text token to train on
    @param reset    (int)  Ignored

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this sample belongs to the
                           i'th category. An array containing all zeros
                           implies no decision could be made.
    """
    bitmap = self.encodeToken(token)
    densePattern = self.sparsifyPattern(bitmap,self.n)
    if self.verbosity >= 1:
      print "Inference with token=",token,"bitmap=",bitmap
      print "Dense version:",densePattern
    (_, inferenceResult, _, _) = self.classifier.infer(densePattern)
    if self.verbosity >= 1:
      print "Inference result=",inferenceResult

    return inferenceResult
