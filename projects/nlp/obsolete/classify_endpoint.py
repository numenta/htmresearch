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

from collections import defaultdict, OrderedDict
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.encoders.language_encoder import LanguageEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel



class ClassificationModelEndpoint(ClassificationModel):
  """
  Class to run the survey response classification task with Cortical.io
  text endpoint encodings and classification system.

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelEndpoint",
               unionSparsity=0.20,
               cacheRoot=None):
    """
    Initializes the encoder as CioEncoder; requires a valid API key.
    """
    super(ClassificationModelEndpoint, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    cacheRoot = cacheRoot or os.path.dirname(os.path.realpath(__file__))

    self.encoder = CioEncoder(cacheDir=os.path.join(cacheRoot, "CioCache"),
                              unionSparsity=unionSparsity)
    self.compareEncoder = LanguageEncoder()

    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity/100) * self.n)

    self.categoryBitmaps = {}
    self.negatives = defaultdict(list)
    self.positives = defaultdict(list)


  def encodeSample(self, sample):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param sample         (list)          Tokenized sample, where each item is
                                          a string
    @return fp            (dict)          The sample text, sparsity, and bitmap.
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
            "sparsity":float(self.w)/self.n,
            "bitmap":self.encodeRandomly(sample, self.n, self.w)}

    return fp


  def resetModel(self):
    """Reset the model"""
    self.positives.clear()
    self.negatives.clear()
    self.categoryBitmaps.clear()


  def trainModel(self, i, negatives=None):
    # TODO: add batch training, where i is a list; note we should only add
    # negatives when training on one sample so we know which labels to use.
    """
    Train the classifier on the sample and labels for record i. Use
    Cortical.io's createClassification() to make a bitmap that represents the
    class. The list sampleReference is populated to correlate classifier
    prototypes to sample IDs.

    @param negative   (list)            Each item is the dictionary containing
                                        text, sparsity and bitmap for the
                                        negative samples.
    """
    record = self.patterns[i]
    labelsToUpdateBitmaps = set()
    for label in record["labels"]:
      if record["pattern"]["text"] and record["pattern"]["bitmap"].any():
        self.positives[label].append(record["pattern"]["text"])
        if negatives:
          for neg in negatives:
            if neg["text"]:
              self.negatives[label].append(neg["text"])
        labelsToUpdateBitmaps.add(label)

    for label in labelsToUpdateBitmaps:
      self.categoryBitmaps[label] = self.encoder.createCategory(
        str(label), self.positives[label], self.negatives[label])["positions"]
      self.sampleReference.append(i)


  def testModel(self, i, _, metric="overlappingAll"):
    """
    Test on record i. The Cortical.io classifier returns a dictionary
    containing various distance metrics between the sample and the classes.

    @param metric     (str)           Distance metric use by classifier.
    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    sampleBitmap = self.patterns[i]["pattern"]["bitmap"].tolist()

    distances = defaultdict(list)
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      distances[cat] = self.compareEncoder.compare(sampleBitmap, catBitmap)

    return self.getWinningLabels(distances, metric=metric)


  def getWinningLabels(self, distances, metric):
    """
    Return indices of winning categories, based off of the input metric.
    Overrides the base class implementation.
    """
    metricValues = numpy.array([v[metric] for v in distances.values()])
    sortedIdx = numpy.argsort(metricValues)

    # euclideanDistance and jaccardDistance are ascending
    descendingOrder = ("overlappingAll", "overlappingLeftRight",
                       "overlappingRightLeft", "cosineSimilarity",
                       "weightedScoring")
    if metric in descendingOrder:
      sortedIdx = sortedIdx[::-1]

    return numpy.array(
      [distances.keys()[catIdx] for catIdx in sortedIdx[:self.numLabels]])


  def getCategoryDistances(self, sort=True, save=None, labelRefs=None):
    """
    Return a dict where keys are categories and values are dicts of distances.

    @param sort      (bool)        Sort the inner dicts with compareCategories()
    @param save      (str)         Dump catDistances to a JSON in this dir.
    @return          (defaultdict)

    E.g. w/ categories 0 and 1:
      catDistances = {
          0: {
              0: {"cosineSimilarity": 1.0, ...},
              1: {"cosineSimilarity": 0.33, ...}
              },
          1: {
              0: {"cosineSimilarity": 0.33, ...},
              1: {"cosineSimilarity": 1.0, ...}
              }
    Note the inner-dicts of catDistances are OrderedDict objects.
    """
    catDistances = defaultdict(list)
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      catDistances[cat] = OrderedDict()
      for compareCat, compareBitmap in self.categoryBitmaps.iteritems():
        # List is in order of self.categoryBitmaps.keys()
        catDistances[cat][compareCat] = self.compareEncoder.compare(
          catBitmap, compareBitmap)

    if sort:
      # Order each inner dict of catDistances such that the ranking is most to
      # least similar.
      catDistances = self.compareCategories(catDistances)

    if save is not None:
      self.writeOutCategories(
        save, comparisons=catDistances, labelRefs=labelRefs)

    return catDistances


  @staticmethod
  def compareCategories(catDistances, metric="overlappingAll"):
    """
    Calculate category distances. Returns a defaultdict of category keys, where
    values are OrderedDicts sorted such that the most similar categories
    (according to the input metric) are listed first.
    """
    descendingOrder = ("overlappingAll", "overlappingLeftRight",
                       "overlappingRightLeft", "cosineSimilarity",
                       "weightedScoring")

    categoryComparisons = defaultdict(list)
    for k, v in catDistances.iteritems():
      # Create a dict for this category
      metricDict = {compareCat: distances[metric]
                    for compareCat, distances in v.iteritems()}
      # Sort the dict by the metric
      reverse = True if metric in descendingOrder else False
      categoryComparisons[k] = OrderedDict(
        sorted(metricDict.items(), key=lambda k: k[1], reverse=reverse))

    return categoryComparisons


  @staticmethod
  def query():
    print "The Classification Endpoint model doesn't support this method."


  @staticmethod
  def infer():
    print "The Classification Endpoint model doesn't support this method."
