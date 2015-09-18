# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy
import os
import random

from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel

from cortipy.cortical_client import CorticalClient
from cortipy.exceptions import UnsuccessfulEncodingError



class ClassificationModelContext(ClassificationModel):
  """
  Class to run the survey response classification task with Cortical.io
  text context, then AND the context

  From the experiment runner, the methods expect to be fed one sample at a time.
  """

  def __init__(self, verbosity=1, numLabels=1):
    """
    Initialize the CorticalClient and CioEncoder. Requires a valid API key.
    """
    super(ClassificationModelContext, self).__init__(verbosity)

    root = os.path.dirname(os.path.realpath(__file__))
    self.encoder = CioEncoder(cacheDir=os.path.join(root, "CioCache"))
    self.client = CorticalClient(self.encoder.apiKey)

    self.n = self.encoder.n
    self.w = int((self.encoder.targetSparsity / 100) * self.n)

    self.categoryBitmaps = {}
    self.numLabels = numLabels


  def encodePattern(self, pattern):
    """
    Encode an SDR of the input string by querying the Cortical.io API.

    @param pattern     (list)           Tokenized sample, where each item is a
                                        string
    @return            (dictionary)     Dictionary, containing text, sparsity,
                                        and bitmap
    Example return dict:
    {
      "text": "Example text",
      "sparsity": 0.0,
      "bitmap": numpy.zeros(0)
    }
    """
    text = " ".join(pattern)
    return {"text": text, "sparsity": 0.0, "bitmap": self._encodeText(text)}


  def _encodeText(self, text):
    fpInfo = self.encoder.encode(text)
    if self.verbosity > 1:
      print "Fingerprint sparsity = {0}%.".format(fpInfo["sparsity"])

    if fpInfo:
      bitmap = numpy.array(fpInfo["fingerprint"]["positions"])
    else:
      bitmap = self.encodeRandomly(text)

    return bitmap.astype(int)


  def resetModel(self):
    """Reset the model"""
    self.categoryBitmaps.clear()


  def trainModel(self, samples, labels):
    """
    Train the classifier on the input sample and label. Use Cortical.io's
    keyword extraction to get the most relevant terms then get the intersection
    of those bitmaps

    @param samples     (dictionary)      Dictionary, containing text, sparsity,
                                         and bitmap
    @param labels      (int)             Reference index for the classification
                                         of this sample.
    """
    for sample, sample_labels in zip(samples, labels):
      bitmaps = [sample["bitmap"].tolist()]
      context = self.client.getContextFromText(bitmaps, maxResults=5,
                                               getFingerprint=True)

      if len(context) != 0:
        union = numpy.zeros(0)
        for c in context:
          bitmap = c["fingerprint"]["positions"]
          union = numpy.union1d(bitmap, union).astype(int)

        for label in sample_labels:
          # Haven't seen the label before
          if label not in self.categoryBitmaps:
            self.categoryBitmaps[label] = union

          intersection = numpy.intersect1d(union, self.categoryBitmaps[label])
          if intersection.size == 0:
            # Don't want to lose all the old information
            union = numpy.union1d(union, self.categoryBitmaps[label]).astype(int)
            # Need to sample to stay sparse
            count = len(union)
            sampleIndices = random.sample(xrange(count), min(count, self.w))
            intersection = numpy.sort(union[sampleIndices])

          self.categoryBitmaps[label] = intersection


  def testModel(self, sample):
    """
    Test the intersection bitmap on the input sample. Returns a dictionary
    containing various distance metrics between the sample and the classes.

    @param sample     (dictionary)      Dictionary, containing text, sparsity,
                                        and bitmap
    @return           (dictionary)      The distances between the sample and
                                        the classes
    Example return dict:
      {
        0: {
          "cosineSimilarity": 0.6666666666666666,
          "euclideanDistance": 0.3333333333333333,
          "jaccardDistance": 0.5,
          "overlappingAll": 6,
          "overlappingLeftRight": 0.6666666666666666,
          "overlappingRightLeft": 0.6666666666666666,
          "sizeLeft": 9,
          "sizeRight": 9,
          "weightedScoring": 0.4436476984102028
        }
      }
    """

    sampleBitmap = sample["bitmap"].tolist()

    distances = {}
    for cat, catBitmap in self.categoryBitmaps.iteritems():
      distances[cat] = self.client.compare(sampleBitmap, catBitmap.tolist())

    return self.winningLabels(distances, numberCats=self.numLabels,
      metric="overlappingAll") 


  @staticmethod
  def winningLabels(distances, numberCats, metric):
    """
    Return indices of winning categories, based off of the input metric.
    Overrides the base class implementation.
    """
    metricValues = numpy.array([v[metric] for v in distances.values()])
    sortedIdx = numpy.argsort(metricValues)

    # euclideanDistance and jaccardDistance are ascending
    descendingOrder = set(["overlappingAll", "overlappingLeftRight",
      "overlappingRightLeft", "cosineSimilarity", "weightedScoring"])
    if metric in descendingOrder:
      sortedIdx = sortedIdx[::-1]

    return [distances.keys()[catIdx] for catIdx in sortedIdx[:numberCats]]
