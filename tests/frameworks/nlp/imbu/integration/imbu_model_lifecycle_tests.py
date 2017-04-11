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


class TestImbuLifecycle(unittest.TestCase):

  def setUp(self):
    self.dataPath = os.path.join(TEST_DATA_DIR, "sample_reviews_subset.csv")


  def _createTempDir(self):
    tmpDir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmpDir)
    return tmpDir


  def testCacheSaveAndReload(self):
    imbuTempDir = self._createTempDir()
    originalCacheDir = os.path.join(imbuTempDir, "original_test_cache")
    imbu = ImbuModels(
      cacheRoot=originalCacheDir,
      dataPath=self.dataPath,
      retina="en_associative",
      apiKey=os.environ.get("CORTICAL_API_KEY")
    )

    # Train with standard cache location, save
    modelType = "HTMNetwork"
    networkConfigName = "imbu_sensor_knn.json"
    checkpointLocation = os.path.join(imbuTempDir, "checkpoint")
    originalModel = imbu.createModel(modelType,
                                     loadPath="",
                                     savePath=checkpointLocation,
                                     networkConfigName=networkConfigName)

    # Physically move the cache to a new directory; we copy it b/c moving it
    # would break the test cleanup method
    encoder = originalModel.getEncoder()
    newCacheDir = os.path.join(imbuTempDir, "new_test_cache")
    shutil.copytree(originalCacheDir, newCacheDir)
    self.addCleanup(shutil.rmtree, newCacheDir)
    self.assertGreater(len(os.listdir(newCacheDir)), 0,
      "The new cache directory is empty!")

    # Load a new model and set cache location to the new directory
    newModel = imbu.createModel(modelType,
                                loadPath=checkpointLocation,
                                savePath=checkpointLocation,
                                networkConfigName=networkConfigName)
    newEncoder = newModel.getEncoder()
    newEncoder.cacheDir = newCacheDir
    newEncoderCacheDir = getattr(newEncoder, "cacheDir")
    self.assertNotEquals(
      newEncoderCacheDir,
      getattr(encoder, "cacheDir"),
      "Old and new cache locations shouldn't be the same.")
    del originalModel

    # Run inference with old data, expecting no new caching
    sizeOfCache = len(os.listdir(newEncoderCacheDir))
    imbu.query(newModel, "unicorn")
    self.assertEquals(sizeOfCache, len(os.listdir(newEncoderCacheDir)), "")

    # Run inference with new data, adding to the new cache location
    imbu.query(newModel, "brains")
    self.assertEquals(sizeOfCache + 1, len(os.listdir(newEncoderCacheDir)), "")
