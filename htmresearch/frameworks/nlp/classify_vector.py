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

from htmresearch.encoders.vec_encoder import VecEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier



class ClassificationModelVector(ClassificationModel):
  """
  Class to run classification with word embeddings.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self,
               embeddingsPath,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelVector"):

    super(ClassificationModelVector, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    self.classifier = KNNClassifier(k=numLabels,
                                    distanceMethod='rawOverlap',
                                    exact=False,
                                    verbosity=verbosity-1)
    self.encoder = VecEncoder(embeddingsPath, verbosity=verbosity)


  def encodeSample(self, sample):
    """
    Populate an encoding dict with the word vector.

    @param sample     (list)        Tokenized sample, where each item is a str.
    @return           (dict)        The sample text, sparsity, and bitmap.
    """
    encodedEmbedding = self.encoder.encode(sample)

    return {"text": " ".join(sample),
            "sparsity": encodedEmbedding["sparsity"],
            "bitmap": numpy.array(encodedEmbedding["bitmap"])}


  def trainModel(self, i):
    """
    Train the classifier on the sample and labels for record i. The list
    sampleReference is populated to correlate classifier prototypes to sample
    IDs.
    """
    bitmap = self.patterns[i]["pattern"]["bitmap"]
    if bitmap.any():
      for label in self.patterns[i]["labels"]:
        self.classifier.learn(bitmap, label, isSparse=self.encoder.nMacro)
        self.sampleReference.append(self.patterns[i]["ID"])


  def testModel(self, i, numLabels=3):
    """
    Test the model on record i.

    @param numLabels  (int)           Number of classification predictions.
    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    (_, inferenceResult, _, _) = self.classifier.infer(self.sparsifyPattern(
      self.patterns[i]["pattern"]["bitmap"], self.encoder.n))
    return self.getWinningLabels(inferenceResult, numLabels)
