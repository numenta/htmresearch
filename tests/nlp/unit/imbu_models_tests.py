#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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
import unittest

from htmresearch.frameworks.nlp.imbu import ImbuModels
from htmresearch.frameworks.nlp.model_factory import ClassificationModelTypes



TEST_DATA_DIR = os.path.join(
  os.path.dirname(os.path.realpath(__file__)), "test_data")



class TestImbu(unittest.TestCase):

  def setUp(self):
    self.dataPath = os.path.join(TEST_DATA_DIR, "sample_reviews_subset.csv")


  def checkResultsFormatting(self, results, modelName, windowSize=0):
    self.assertEquals(["scores", "text", "windowSize"], sorted(results[0]),
      "Results dict for {} has incorrect keys.".format(modelName))
    self.assertEquals(windowSize, results[0]["windowSize"],
      "Results give incorrect window size for {} model.".format(modelName))


  def testMappingModelNamesToModelTypes(self):
    imbu = ImbuModels(cacheRoot="fake_cache_root", dataPath=self.dataPath)

    for modelName, mappedName in imbu.modelMappings.iteritems():
      self.assertEquals(mappedName, imbu._mapModelName(modelName),
        "Incorrect mapping returned for model named '{}'".format(modelName))

    self.assertRaises(ValueError, imbu._mapModelName, "fakeModel")


  def testResultsFormatting(self):
    imbu = ImbuModels(cacheRoot="fake_cache_root", dataPath=self.dataPath)

    query = "Hello world!"
    distanceArray = numpy.random.rand(35)
    idList = range(27) + [x+1000 for x in xrange(8)]

    # Test a model with windows
    modelName = "HTM_sensor_simple_tp_knn"
    results = imbu.formatResults(modelName, query, distanceArray, idList)
    self.checkResultsFormatting(results, modelName, windowSize=10)

    # Test a word-level model
    modelName = "HTM_sensor_knn"
    results = imbu.formatResults(modelName, query, distanceArray, idList)
    self.checkResultsFormatting(results, modelName, windowSize=1)

    # Test a document-level model
    modelName = "CioDocumentFingerprint"
    distanceArray = numpy.random.rand(2)
    idList = range(1)
    results = imbu.formatResults(modelName, query, distanceArray, idList)
    self.checkResultsFormatting(results, modelName)



if __name__ == "__main__":
  unittest.main()
