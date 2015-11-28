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

from collections import Counter, OrderedDict

from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from nupic.algorithms.KNNClassifier import KNNClassifier

try:
  import simplejson as json
except ImportError:
  import json



class ClassificationModelWindows(ClassificationModel):
  """
  Class to run classification tasks with a sliding windwo of Coritcal.io word
  fingerprint encodings.
  """

  def __init__(self,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelWindow",
               unionSparsity=20.0,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None):

    super(ClassificationModelWindows, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    # Init kNN classifier and Cortical.io encoder; need valid API key (see
    # CioEncoder init for details).
    self.classifier = KNNClassifier(k=numLabels,
                                    distanceMethod='rawOverlap',
                                    exact=False,
                                    verbosity=verbosity-1)

    root = os.path.dirname(os.path.realpath(__file__))
    self.encoder = CioEncoder(retinaScaling=retinaScaling,
                              cacheDir=os.path.join(root, "CioCache"),
                              fingerprintType=EncoderTypes.word,
                              unionSparsity=unionSparsity,
                              retina=retina,
                              apiKey=apiKey)


  def encodeSample(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API for each
    word. The resulting bitmaps are unionized in a sliding window.

    @param sample     (list)        Tokenized sample, where each item is a str.
    @return           (list)        Pattern dicts for the windows, each with the
                                    sample text, sparsity, and bitmap.
    """
    return self.encoder.getWindowEncoding(sample)


  def writeOutEncodings(self):
    """
    Write the encoding dictionaries to a txt file; overrides the superclass
    implementation.
    """
    if not os.path.isdir(self.modelDir):
      raise ValueError("Invalid path to write file.")

    # Cast numpy arrays to list objects for serialization.
    jsonPatterns = copy.deepcopy(self.patterns)
    for jp in jsonPatterns:
      for tokenPattern in jp["pattern"]:
        tokenPattern["bitmap"] = tokenPattern.get("bitmap", None).tolist()
      jp["labels"] = jp.get("labels", None).tolist()

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
    for token in self.patterns[i]["pattern"]:
      if token["bitmap"].any():
        for label in self.patterns[i]["labels"]:
          self.classifier.learn(
            token["bitmap"], label, isSparse=self.encoder.n)
          self.sampleReference.append(self.patterns[i]["ID"])


  def testModel(self, i, seed=42):
    """
    Test the model on record i. Returns the classifications most frequent
    amongst the classifications of the sample's individual tokens.
    We ignore the terms that are unclassified, picking the most frequent
    classifications among those that are detected; in getWinningLabels().

    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    totalInferenceResult = None
    for pattern in self.patterns[i]["pattern"]:
      if not pattern:
        continue

      _, inferenceResult, _, _ = self.classifier.infer(
        self.sparsifyPattern(pattern["bitmap"], self.encoder.n))

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
    distances = numpy.zeros((self.classifier._numPatterns))
    for i, p in enumerate(patterns):
      (_, _, dist, _) = self.classifier.infer(
        self.sparsifyPattern(p["bitmap"], self.encoder.n))

      distances = numpy.array([sum(x) for x in zip(dist, distances)])

    return distances / (i+1)
