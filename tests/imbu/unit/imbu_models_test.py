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
import shutil
import tempfile
import unittest

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.imbu import ImbuModels
from htmresearch.frameworks.nlp.classification_model import (
  ClassificationModel
)
from htmresearch.frameworks.nlp.classify_network_api import (
  ClassificationNetworkAPI
)



TEST_DATA_DIR = os.path.abspath(
  os.path.join(
    os.path.dirname(
      os.path.realpath(__file__)
    ),
    "..", "..", "nlp", "unit", "test_data"
  )
)



class TestImbu(unittest.TestCase):

  def setUp(self):
    self.dataPath = os.path.join(TEST_DATA_DIR, "sample_reviews_subset.csv")


  def _setupFakeImbuModelsInstance(self):
    return ImbuModels(
      cacheRoot="fake_cache_root",
      dataPath=self.dataPath,
      retina="en_associative",
      apiKey=os.environ.get("CORTICAL_API_KEY")
    )


  def _createTempModelCheckpoint(self):
    tmpDir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmpDir)
    return os.path.join(tmpDir, "checkpoint")


  def testCreateModel(self):
    imbu = self._setupFakeImbuModelsInstance()

    # You must specify a known model type
    self.assertRaises(TypeError, imbu.createModel, "random model type")

    # Assert that you can create models of known types from scratch (i.e. with
    # empty loadPath value)
    for modelType in imbu.modelMappings:
      # You _must_ specify loadPath and savePath.  Assert that attempts to not
      # specify will result in failure
      self.assertRaises(TypeError, imbu.createModel, modelType)
      self.assertRaises(TypeError, imbu.createModel, modelType, None)
      self.assertRaises(ValueError, imbu.createModel, modelType, None, None)

      # Attempt to create model using default arguments
      model = imbu.createModel(
        modelType, "", "", networkConfigName="imbu_sensor_knn.json"
      )

      # Assert model created was one of the expected types
      self.assertTrue(isinstance(model, ClassificationModel) or
                      isinstance(model, ClassificationNetworkAPI))


  def _exerciseModelLifecycle(self, modelType, queryTerm="unicorn",
                              networkConfigName="imbu_sensor_knn.json"):
    imbu = self._setupFakeImbuModelsInstance()

    checkpointLocation = self._createTempModelCheckpoint()

    # Create model with no load path.  This should trigger the creation of a
    # new intance, and train it.  Specify save path so that the trained model
    # gets saved
    originalModel = imbu.createModel(modelType,
                                     loadPath="",
                                     savePath=checkpointLocation,
                                     networkConfigName=networkConfigName)

    # Make a query, keep around for later comparison
    originalQueryResults = imbu.query(originalModel, queryTerm)

    del originalModel

    # Create model, specifying previous load path
    model = imbu.createModel(modelType,
                             loadPath=checkpointLocation,
                             savePath=checkpointLocation,
                             networkConfigName=networkConfigName)

    self.assertTrue(all(numpy.array_equal(original, new)
                        for (original, new)
                        in zip(originalQueryResults,
                               imbu.query(model, queryTerm))))


  def testCreateSaveLoadCioWordFingerprintModel(self):
    self._exerciseModelLifecycle("CioWordFingerprint")


  def testCreateSaveLoadCioDocumentFingerprintModel(self):
    self._exerciseModelLifecycle("CioDocumentFingerprint")


  def testCreateSaveLoadHTMNetworkModel(self):
    self._exerciseModelLifecycle("HTMNetwork")


  def testLoadKeywordsModel(self):
    self._exerciseModelLifecycle("Keywords")


  def testMappingModelNamesToModelTypes(self):
    imbu = ImbuModels(cacheRoot="fake_cache_root", dataPath=self.dataPath)

    for modelName, mappedName in imbu.modelMappings.iteritems():
      self.assertEquals(mappedName, imbu._mapModelName(modelName),
        "Incorrect mapping returned for model named '{}'".format(modelName))

    self.assertRaises(ValueError, imbu._mapModelName, "fakeModel")


  def _checkNLPObjectParams(self, nlpObject, paramsToCheck):
    for key, value in paramsToCheck.iteritems():
      param = getattr(nlpObject, key)
      self.assertEquals(value, param,
        "The {} param for {} is not as expected.".format(key, repr(nlpObject)))


  def testSetParamsInModelFactory(self):
    imbu = self._setupFakeImbuModelsInstance()

    checkpointLocation = self._createTempModelCheckpoint()

    # Base set of required Cio params to check
    cacheRoot = "fake_cache_root"
    paramsToCheck = dict(
      retina="en_associative",
      apiKey=os.environ.get("CORTICAL_API_KEY"),
      retinaScaling=1.0,
      _cacheDir=cacheRoot,
    )

    # Create Cio models and check their special params
    model = imbu.createModel("CioWordFingerprint",
                             loadPath="",
                             savePath=checkpointLocation)
    paramsToCheck.update(fingerprintType=EncoderTypes.word)
    self._checkNLPObjectParams(model.getEncoder(), paramsToCheck)

    model = imbu.createModel("CioDocumentFingerprint",
                             loadPath="",
                             savePath=checkpointLocation)
    paramsToCheck.update(fingerprintType=EncoderTypes.document)
    self._checkNLPObjectParams(model.getEncoder(), paramsToCheck)

    # Create HTM Network model and check Imbu specific config params
    model = imbu.createModel("HTMNetwork",
                             loadPath="",
                             savePath=checkpointLocation,
                             networkConfigName="imbu_sensor_knn.json")
    networkConfig = getattr(model, "networkConfig")
    self.assertEquals(
      cacheRoot,
      networkConfig["sensorRegionConfig"]["regionParams"]["cacheRoot"],
      "HTM Network model does not set cacheRoot correctly.")
    self.assertEquals(
      "pctOverlapOfInput",
      networkConfig["classifierRegionConfig"]["regionParams"]["distanceMethod"],
      "HTM Network model specifies an incorrect distance metric for Imbu.")

    # Create Keywords model and check specific params
    model = imbu.createModel("Keywords",
                             loadPath="",
                             savePath=checkpointLocation)
    paramsToCheck = dict(k=10*len(imbu.dataDict))
    self._checkNLPObjectParams(model.getClassifier(), paramsToCheck)


  def testDefaultParams(self):
    imbu = ImbuModels(cacheRoot=None, dataPath=self.dataPath)

    checkpointLocation = self._createTempModelCheckpoint()

    model = imbu.createModel("HTMNetwork",
                             loadPath="",
                             savePath=checkpointLocation,
                             networkConfigName="imbu_sensor_knn.json")

    # Test for default cache directory
    defaultCacheLocation = "nupic.research/htmresearch/encoders/CioCache"
    self.assertIn(
      defaultCacheLocation,
      getattr(model.getEncoder(), "_cacheDir"),
      "Cio encoder cache dir is not the expected default location.")


  def _checkResultsFormatting(self, results, modelName, windowSize=0):
    self.assertEquals(["scores", "text", "windowSize"], sorted(results[0]),
      "Results dict for {} has incorrect keys.".format(modelName))
    self.assertEquals(windowSize, results[0]["windowSize"],
      "Results give incorrect window size for {} model.".format(modelName))


  def testResultsFormatting(self):
    imbu = self._setupFakeImbuModelsInstance()

    query = "Hello world!"
    distanceArray = numpy.random.rand(35)
    idList = range(27) + [x+1000 for x in xrange(8)]

    # Test a model with windows
    modelName = "HTM_sensor_simple_tp_knn"
    results = imbu.formatResults(modelName, query, distanceArray, idList)
    self._checkResultsFormatting(results, modelName, windowSize=10)

    # Test a word-level model
    modelName = "HTM_sensor_knn"
    results = imbu.formatResults(modelName, query, distanceArray, idList)
    self._checkResultsFormatting(results, modelName, windowSize=1)

    # Test a document-level model
    modelName = "CioDocumentFingerprint"
    distanceArray = numpy.random.rand(2)
    idList = range(1)
    results = imbu.formatResults(modelName, query, distanceArray, idList)
    self._checkResultsFormatting(results, modelName)
