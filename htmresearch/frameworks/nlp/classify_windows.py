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
import operator
import os

from collections import Counter, defaultdict, OrderedDict

from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.support.text_preprocess import TextPreprocess
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
               unionSparsity=0.20,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None):

    super(ClassificationModelWindows, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    # window patterns below minSparsity will be skipped over
    self.minSparsity = 0.9 * unionSparsity

    self.classifier = KNNClassifier(k=numLabels,
                                    distanceMethod='rawOverlap',
                                    exact=False,
                                    verbosity=verbosity-1)

    # need valid API key (see CioEncoder init for details)
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
    return self.encoder.getWindowEncoding(sample, self.minSparsity)


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
    patterns, of which, any that are too sparse are skipped over.

    @return       (int)     Number of patterns trained on.
    """
    patternWindows = self.patterns[i]["pattern"]
    if len(patternWindows) == 0:
      # no patterns b/c no windows were large enough for encoding
      return
    count = 0
    for window in patternWindows:
      for label in self.patterns[i]["labels"]:
        self.classifier.learn(
          window["bitmap"], label, isSparse=self.encoder.n)
        self.sampleReference.append(self.patterns[i]["ID"])
        count += 1

    return count


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


  def queryModel(self, query, preprocess=False):
    """
    Preprocesses the query, encodes it into a pattern, then queries the
    classifier to infer distances to trained-on samples.
    @return       (list)          Two-tuples of sample ID and distance, sorted
                                  closest to farthest from the query.
    """
    if preprocess:
      sample = TextPreprocess().tokenize(query,
                                         ignoreCommon=100,
                                         removeStrings=["[identifier deleted]"],
                                         correctSpell=True)
    else:
      sample = TextPreprocess().tokenize(query)

    # Get window patterns for the query, but if the query is too small such that
    # the window encodings are too sparse, we default to a pure union.
    encodedQuery = self.encodeSample(sample)
    if len(encodedQuery) == 0:
      sample = " ".join(sample)
      fpInfo = self.encoder.getUnionEncoding(sample)
      encodedQuery = [{
        "text":fpInfo["text"],
        "sparsity":fpInfo["sparsity"],
        "bitmap":numpy.array(fpInfo["fingerprint"]["positions"])
      }]
    allDistances = self.infer(encodedQuery)

    if len(allDistances) != len(self.sampleReference):
      raise IndexError("Number of protoype distances must match number of "
                       "samples trained on.")

    sampleDistances = defaultdict()
    for uniqueID in self.sampleReference:
      sampleDistances[uniqueID] = min(
        [allDistances[i] for i, x in enumerate(self.sampleReference)
         if x == uniqueID])

    return sorted(sampleDistances.items(), key=operator.itemgetter(1))


  def infer(self, patterns):
    """
    Get the classifier output for a single input pattern; assumes classifier
    has an infer() method (as specified in NuPIC kNN implementation). For this
    model we sum the distances across the patterns and normalize
    before returning.

    NOTE: there is no check here that the pattern sparsities are > the minimum.

    @return       (numpy.array)       Each entry is the distance from the
        input pattern to that prototype (pattern in the classifier). All
        distances are between 0.0 and 1.0
    """
    distances = numpy.zeros((self.classifier._numPatterns))

    for i, p in enumerate(patterns):
      (_, _, dist, _) = self.classifier.infer(
        self.sparsifyPattern(p["bitmap"], self.encoder.n))

      distances = distances + dist

    return distances / float(i+1)
