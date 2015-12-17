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
import cPickle as pkl
import numpy
import operator
import os
import random

from collections import defaultdict, OrderedDict

from htmresearch.support.text_preprocess import TextPreprocess

try:
  import simplejson as json
except ImportError:
  import json



class ClassificationModel(object):
  """
  Base class for NLP models of classification tasks. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below.
  """
  # TODO: use nupic.bindings.math import Random

  def __init__(self,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModel"):
    """
    If there are no labels set numLabels=0.
    """
    self.numLabels = numLabels
    self.verbosity = verbosity
    self.modelDir = modelDir
    if not os.path.exists(self.modelDir):
      os.makedirs(self.modelDir)
    self.modelPath = None

    # each time a sample is trained on, its unique ID is appended
    self.sampleReference = []

    self.patterns = []


  def encodeSample(self, sample):
    """
    The subclass implementations must return the encoding in the following
    format:
      {
        ["text"]:sample,
        ["sparsity"]:sparsity,
        ["bitmap"]:bitmapSDR
      }
    Note: sample is a string, sparsity is float, and bitmapSDR is a numpy array.
    """
    raise NotImplementedError


  def trainModel(self, index):
    raise NotImplementedError


  def testModel(self, index, numLabels):
    raise NotImplementedError


  def saveModel(self, trial=None):
    """Save the serialized model."""
    try:
      if not os.path.exists(self.modelDir):
        os.makedirs(self.modelDir)
      if trial:
        self.modelPath = os.path.join(
          self.modelDir, "model_{}.pkl".format(trial))
      else:
        self.modelPath = os.path.join(self.modelDir, "model.pkl")
      with open(self.modelPath, "wb") as f:
        pkl.dump(self, f)
      if self.verbosity > 0:
        print "Model saved to '{}'.".format(self.modelPath)
    except IOError as e:
      print "Could not save model to '{}'.".format(self.modelPath)
      raise e


  def loadModel(self, modelDir):
    """Return the serialized model."""
    modelPath = os.path.join(modelDir, "model.pkl")
    try:
      with open(modelPath, "rb") as f:
        model = pkl.load(f)
      if self.verbosity > 0:
        print "Model loaded from \'{}\'.".format(modelPath)
      return model
    except IOError as e:
      print "Could not load model from \'{}\'.".format(modelPath)
      raise e


  def resetModel(self):
    """Reset the model by clearing the classifier."""
    self.classifier.clear()


  def prepData(self, dataDict, preprocess):
    """
    Returns a dict of same format as dataDict where the text data has been
    tokenized (and preprocessed if specified).

    @param dataDict     (dict)          Keys are data record IDs, values are
        two-tuples of text (str) and categories (numpy.array). If no labels,
        the categories array is empty. E.g.:

        dataDict = OrderedDict([
            ('0', ('Hello world!', array([3])),
            ('1', ('import this', array([0, 3]))
        ])
    """
    outDict = OrderedDict()
    for dataID, data in dataDict.iteritems():
      outDict[dataID] = (self.prepText(data[0], preprocess), data[1], data[2])

    return outDict


  @staticmethod
  def prepText(text, preprocess=False):
    """
    Returns a list of the text tokens.

    @param preprocess   (bool)    Whether or not to preprocess the text data.
    """
    if preprocess:
      sample = TextPreprocess().tokenize(text,
                                         ignoreCommon=100,
                                         removeStrings=["[identifier deleted]"],
                                         correctSpell=True)
    else:
      sample = TextPreprocess().tokenize(text)

    return sample


  def writeOutCategories(self, dirName, comparisons=None, labelRefs=None):
    """
    For models which define categories with bitmaps, log the categories (and
    their relative distances) to a JSON specified by the dirName. The JSON will
    contain the dict of category bitmaps, and if specified, dicts of label
    references and category bitmap comparisons.
    """
    if not hasattr(self, "categoryBitmaps"):
      raise TypeError("This model does not have category encodings compatible "
                      "for logging.")

    if not os.path.isdir(dirName):
      raise ValueError("Invalid path to write file.")

    with open(os.path.join(dirName, "category_distances.json"), "w") as f:
      catDict = {"categoryBitmaps":self.categoryBitmaps,
                 "labelRefs":dict(enumerate(labelRefs)) if labelRefs else None,
                 "comparisons":comparisons if comparisons else None}
      json.dump(catDict,
                f,
                sort_keys=True,
                indent=2,
                separators=(",", ": "))


  @staticmethod
  def classifyRandomly(labels):
    """Return accuracy of random classifications for the labels."""
    randomLabels = numpy.random.randint(0, labels.max(), labels.shape)
    return (randomLabels == labels).sum() / float(labels.shape[0])


  def getWinningLabels(self, labelFreq, seed=None):
    """
    Returns indices of input array, sorted for highest to lowest value. E.g.
      >>> labelFreq = array([ 0., 4., 0., 1.])
      >>> winners = getWinningLabels(labelFreq, seed=42)
      >>> print winners
      array([1, 3])
    Note:
      - indices of nonzero values are not included in the returned array
      - ties are handled randomly

    @param labelFreq    (numpy.array)   Ints that (in this context) represent
                                        the frequency of inferred labels.
    @param seed         (int)           Seed the numpy random generator.
    @return             (numpy.array)   Indicates largest elements in labelFreq,
                                        sorted greatest to least. Length is up
                                        to numLabels.
    """
    if labelFreq is None:
      return numpy.array([])

    if seed:
      numpy.random.seed(seed)
    randomValues = numpy.random.random(labelFreq.size)

    # First sort by labelFreq, then sort by random values.
    winners = numpy.lexsort((randomValues, labelFreq))[::-1][:self.numLabels]

    return numpy.array([i for i in winners if labelFreq[i] > 0])


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

    encodedQuery = self.encodeSample(sample)

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


  def infer(self, pattern):
    """
    Get the classifier output for a single input pattern; assumes classifier
    has an infer() method (as specified in NuPIC kNN implementation).

    @return dist    (numpy.array)       Each entry is the distance from the
        input pattern to that prototype (pattern in the classifier). We divide
        by the width of the input pattern such that all distances are between
        0.0 and 1.0.
    """
    (_, _, dist, _) = self.classifier.infer(
      self.sparsifyPattern(pattern["bitmap"], self.encoder.n))

    return dist.astype("float64")


  @staticmethod
  def sparsifyPattern(bitmap, n):
    """Return a numpy array of 0s and 1s to represent the input bitmap."""
    sparsePattern = numpy.zeros(n)
    for i in bitmap:
      sparsePattern[i] = 1.0
    return sparsePattern


  def encodeSamples(self, samples, write=False):
    """
    Encode samples and store in self.patterns, write out encodings to a file.

    @param samples    (dict)  Keys are samples' record numbers, values are
                              3-tuples: list of tokens (str), list of labels
                              (int), unique ID (int or str).
    @param write      (bool)  True will write out encodings to a file.
    @return patterns  (list)  A dict for each encoded data sample.
    """
    if self.numLabels == 0:
      # No labels for classification, so populate labels with stand-ins
      self.patterns = [{"recordNumber": i,
                        "pattern": self.encodeSample(s[0]),
                        "labels": numpy.array([-1]),
                        "ID": s[2]}
                       for i, s in samples.iteritems()]
    else:
      self.patterns = [{"recordNumber": i,
                        "pattern": self.encodeSample(s[0]),
                        "labels": s[1],
                        "ID": s[2]}
                       for i, s in samples.iteritems()]

    if write:
      self.writeOutEncodings()

    return self.patterns


  def encodeRandomly(self, sample, n, w):
    """Return a random bitmap representation of the sample."""
    random.seed(sample)
    return numpy.sort(random.sample(xrange(n), w))


  def writeOutEncodings(self):
    """Log the encoding dictionaries to a txt file."""
    if not os.path.isdir(self.modelDir):
      raise ValueError("Invalid path to write encodings file.")

    # Cast numpy arrays to list objects for serialization.
    jsonPatterns = copy.deepcopy(self.patterns)
    for jp in jsonPatterns:
      jp["pattern"]["bitmap"] = jp["pattern"].get(
        "bitmap", numpy.array([])).tolist()
      jp["labels"] = jp.get("labels", numpy.array([])).tolist()

    with open(os.path.join(self.modelDir, "encoding_log.json"), "w") as f:
      json.dump(jsonPatterns, f, indent=2)
