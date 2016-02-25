# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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
import shutil
import tempfile
import unittest

from collections import OrderedDict

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)



_ROOT = (
  os.path.dirname(
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.realpath(__file__)
        )
      )
    )
  )
)


def trainModel(model, trainingData):
  for (document, labels, docId) in trainingData:
    model.trainDocument(document, labels, docId)

  return model



class ClassificationModelInferenceTest(unittest.TestCase):
  """Test the inference methods of the classification models."""

  def setUp(self):
    tmpDir = tempfile.mkdtemp()
    self.modelDir = os.path.join(os.path.realpath(__file__), tmpDir)

    self.modelParams = dict(
      apiKey=None,
      retina="en_associative_64_univ",
      k=1,
      verbosity=0)

    self.dataSet = [
      ("John stopped smoking immediately.", [0], 0),
      ("Jane stopped smoking immediately.", [0], 1),
      ("Jane quit smoking immediately.", [0], 2),
      ("The car immediately stopped.", [1], 3),
      ("Davis trashed his cigarettes for good.", [0], 4)
    ]
    self.numLabels = 2


  def tearDown(self):
    if self.modelDir:
      shutil.rmtree(self.modelDir)



  def _executeModelLifecycle(self, modelName, modelDir):
    """ Create a model, train it, save it, reload it, return it."""
    model = createModel(modelName, **self.modelParams)
    model = trainModel(model, self.dataSet)
    model.save(modelDir)
    return ClassificationModel.load(modelDir)


  def _validateInference(self, model, modelName):
    """ Test that the best matching document is the inference document."""
    for i, doc in enumerate(self.dataSet):
      _, sortedIds, sortedDistances = model.inferDocument(
        doc[0], returnDetailedResults=True, sortResults=True)
      self.assertEquals(doc[2], sortedIds[0],
        "{} did not infer document {} as the best match to itself.".format(
        modelName, doc[2]))


  def _inferWithFirstDocument(self, model, modelName):
    """ Test the model accurately infers that the documents get progressively
    more dissimilar.
    """
    categoryVotes, sortedIds, sortedDistances = model.inferDocument(
      self.dataSet[0][0], returnDetailedResults=True, sortResults=True)
    self.assertEquals(range(len(self.dataSet)), sortedIds,
      "{} did not infer the expected order of sorted doc IDs.".format(
      modelName))
    self.assertEquals(1, categoryVotes[0],
      "{} did not classify document 0 as expected.".format(modelName))


  # @unittest.skip("hi")
  def testSensorKNN(self):
    # Build model
    modelName = "htm"
    modelDir = os.path.join(self.modelDir, "htm.checkpoint")

    networkConfigPath = os.path.join(
      _ROOT, "projects/nlp/data/network_configs/sensor_knn.json")

    self.modelParams.update(
      networkConfig=getNetworkConfig(networkConfigPath),
      numLabels=2,
      modelDir=modelDir,
    )

    model = self._executeModelLifecycle(modelName, modelDir)

    # Test model inference
    # import pdb; pdb.set_trace()
    self._validateInference(model, modelName)
    self._inferFirstDocument(model, modelName)


  # @unittest.skip("hi")
  def testSensorSimpleUPKNN(self):
    # Build model
    modelName = "htm"
    modelDir = os.path.join(self.modelDir, "htm.checkpoint")

    networkConfigPath = os.path.join(
      _ROOT, "projects/nlp/data/network_configs/sensor_simple_TP_knn.json")

    self.modelParams.update(
      networkConfig=getNetworkConfig(networkConfigPath),
      numLabels=2,
      modelDir=modelDir,
    )

    model = self._executeModelLifecycle(modelName, modelDir)

    # Test model inference
    # import pdb; pdb.set_trace()
    self._validateInference(model, modelName)
    self._inferFirstDocument(model, modelName)


  # @unittest.skip("hi")
  def testKeywords(self):
    # Build model
    modelName = "Keywords"
    modelDir = os.path.join(self.modelDir, "keywords.checkpoint")
    self.modelParams.update(
      numLabels=2,
      modelDir=modelDir,
      k=21,
    )
    model = self._executeModelLifecycle(modelName, modelDir)

    # Test model inference
    # import pdb; pdb.set_trace()
    self._validateInference(model, modelName)
    self._inferFirstDocument(model, modelName)
    # categoryVotes, sortedIds, sortedDistances = model.inferDocument(self.dataSet[0][0], returnDetailedResults=True, sortResults=True)
    # import pdb; pdb.set_trace()


  def testCioWordFingerprint(self):
    # Build model
    modelName = "CioWordFingerprint"
    modelDir = os.path.join(self.modelDir, "cioword.checkpoint")
    self.modelParams.update(
      numLabels=2,
      modelDir=modelDir
    )
    model = self._executeModelLifecycle(modelName, modelDir)

    # Test model inference
    self._validateInference(model, modelName)
    self._inferWithFirstDocument(model, modelName)


  def testCioDocumentFingerprint(self):
    # Build model
    modelName = "CioDocumentFingerprint"
    modelDir = os.path.join(self.modelDir, "ciodoc.checkpoint")
    self.modelParams.update(
      numLabels=2,
      modelDir=modelDir
    )
    model = self._executeModelLifecycle(modelName, modelDir)

    # Test model inference
    self._validateInference(model, modelName)
    self._inferWithFirstDocument(model, modelName)



if __name__ == "__main__":
  unittest.main()
