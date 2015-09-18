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
import shutil
import unittest

from collections import OrderedDict
from fluent.models.classification_model import ClassificationModel
from fluent.models.classify_endpoint import ClassificationModelEndpoint
from fluent.models.classify_fingerprint import ClassificationModelFingerprint
from fluent.models.classify_keywords import ClassificationModelKeywords



class ClassificationModelTest(unittest.TestCase):
  """Test the functionality of the classification models."""
  
  def setUp(self):
    self.modelDir = None


  def tearDown(self):
    if self.modelDir:
      shutil.rmtree(self.modelDir)


  def testWinningLabels(self):
    """
    Tests whether classification base class returns multiple labels correctly.
    """
    model = ClassificationModel()
    inferenceResult = numpy.array([3, 1, 4, 0, 1, 0])

    topLabels = model.getWinningLabels(inferenceResult, numLabels=1)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2])),
                    "Output should be label 2.")

    topLabels = model.getWinningLabels(inferenceResult, numLabels=2)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2, 0])),
                    "Output should be labels 2 and 0.")

    # Test only nonzero labels are returned.
    inferenceResult = numpy.array([3, 0, 4, 0, 0, 0])
    topLabels = model.getWinningLabels(inferenceResult, numLabels=5)
    self.assertTrue(numpy.allclose(topLabels, numpy.array([2, 0])),
                    "Output should be labels 2 and 0.")


  def testNoWinningLabels(self):
    """Inferring 0/4 classes should return 0 winning labels."""
    model = ClassificationModel()

    inferenceResult = numpy.array([0, 0, 0, 0])
    topLabels = model.getWinningLabels(inferenceResult)

    self.assertFalse(topLabels)


  def testCalculateAccuracyMixedSamples(self):
    """
    Tests testCalculateAccuracy() method of classification model base class for
    test samples with mixed classifications.
    """
    model = ClassificationModel()

    actualLabels = [numpy.array([0, 1, 2])]
    predictedLabels1 = [numpy.array([1, 2, 0])]
    predictedLabels2 = [numpy.array([1])]
    predictedLabels3 = [None]
    classifications1 = [predictedLabels1, actualLabels]
    classifications2 = [predictedLabels2, actualLabels]
    classifications3 = [predictedLabels3, actualLabels]

    self.assertAlmostEqual(model.calculateAccuracy(classifications1), 1.0)
    self.assertAlmostEqual(
        model.calculateAccuracy(classifications2), float(1)/3)
    self.assertAlmostEqual(model.calculateAccuracy(classifications3), 0.0)


  def testCalculateAccuracyMultipleSamples(self):
    """
    Tests testCalculateAccuracy() method of classification model base class for
    three test samples.
    """
    model = ClassificationModel()

    actualLabels = [numpy.array([0]),
                    numpy.array([0, 2]),
                    numpy.array([0, 1, 2])]
    predictedLabels = [numpy.array([0]),
                       [None],
                       numpy.array([1, 2, 0])]
    classifications = [predictedLabels, actualLabels]

    self.assertAlmostEqual(model.calculateAccuracy(classifications), float(2)/3)


  def testClassifyKeywordsSingleAndMultiClass(self):
    """Tests simple classification with multiple labels for Keywords model."""
    model = ClassificationModelKeywords()

    samples = {0: (["Pickachu"], numpy.array([0, 2, 2])),
               1: (["Eevee"], numpy.array([2])),
               2: (["Charmander"], numpy.array([0, 1, 1])),
               3: (["Abra"], numpy.array([1])),
               4: (["Squirtle"], numpy.array([1, 0, 1]))}

    patterns = model.encodeSamples(samples)
    for i in xrange(len(samples)):
      model.trainModel(i)

    output = [model.testModel(i) for i in xrange(len(patterns))]

    self.assertSequenceEqual(output[0].tolist(), [2, 0],
                             "Incorrect output for first sample.")
    self.assertSequenceEqual(output[1].tolist(), [2],
                             "Incorrect output for second sample.")
    self.assertSequenceEqual(output[2].tolist(), [1, 0],
                             "Incorrect output for third sample.")
    self.assertSequenceEqual(output[3].tolist(), [1],
                             "Incorrect output for fourth sample.")

    # Test the order of class labels doesn't matter when training.
    self.assertTrue(numpy.allclose(output[2], output[4]),
                    "Outputs for samples 2 and 4 should be identical.")


  def testCompareCategories(self):
    model = ClassificationModelEndpoint()

    # Fake distances between three categories (each of size three):
    catDistances = {
        0: OrderedDict([
          (0, {"overlappingAll": 3, "euclideanDistance": 0.0}),
          (1, {"overlappingAll": 0, "euclideanDistance": 1.0}),
          (2, {"overlappingAll": 1, "euclideanDistance": 0.2}),
          ]),
        1: OrderedDict([
          (0, {"overlappingAll": 0, "euclideanDistance": 1.0}),
          (1, {"overlappingAll": 3, "euclideanDistance": 0.0}),
          (2, {"overlappingAll": 2, "euclideanDistance": 0.2}),
          ]),
        2: OrderedDict([
          (0, {"overlappingAll": 1, "euclideanDistance": 0.7}),
          (1, {"overlappingAll": 2, "euclideanDistance": 0.2}),
          (2, {"overlappingAll": 3, "euclideanDistance": 0.0}),
          ]),
        }

    # Assert correct sorting of categories for different metrics.
    expected = {
        0: OrderedDict([(0, 3), (2, 1), (1, 0)]),
        1: OrderedDict([(1, 3), (2, 2), (0, 0)]),
        2: OrderedDict([(2, 3), (1, 2), (0, 1)]),
        }
    catComparisons = model.compareCategories(
        catDistances, metric="overlappingAll")
    self.assertDictEqual(expected, catComparisons,
        "Unexpected category comparison values for overlap metric.")

    expected = {
        0: OrderedDict([(0, 0.0), (2, 0.2), (1, 1.0)]),
        1: OrderedDict([(1, 0.0), (2, 0.2), (0, 1.0)]),
        2: OrderedDict([(2, 0.0), (1, 0.2), (0, 0.7)])
        }
    catComparisons = model.compareCategories(
        catDistances, metric="euclideanDistance")
    self.assertDictEqual(expected, catComparisons,
        "Unexpected category comparison values for Euclidean metric.")


  def testModelSaveAndLoad(self):
    # Keywords model uses the base class implementations of save/load methods.
    self.modelDir = "poke_model"
    model = ClassificationModelKeywords(modelDir=self.modelDir, verbosity=0)
    
    samples = {0: (["Pickachu"], numpy.array([0, 2, 2])),
               1: (["Eevee"], numpy.array([2])),
               2: (["Charmander"], numpy.array([0, 1, 1])),
               3: (["Abra"], numpy.array([1])),
               4: (["Squirtle"], numpy.array([1, 0, 1]))}

    patterns = model.encodeSamples(samples)
    for i in xrange(len(samples)):
      model.trainModel(i)

    output = [model.testModel(i) for i in xrange(len(patterns))]

    model.saveModel()

    loadedModel = ClassificationModel(verbosity=0).loadModel(self.modelDir)
    loadedModelOutput = [loadedModel.testModel(i)
                         for i in xrange(len(patterns))]
    
    for mClasses, lClasses in zip(output, loadedModelOutput):
      self.assertSequenceEqual(mClasses.tolist(), lClasses.tolist(), "Output "
          "classifcations from loaded model don't match original model's.")


  def testQueryMethods(self):
    """Tests the queryModel() and infer() methods in ClassifactionModel base."""
    model = ClassificationModelFingerprint(verbosity=0)

    samples = {0: (["Pickachu"], numpy.array([0, 2, 2])),
               1: (["Eevee"], numpy.array([2])),
               2: (["Charmander"], numpy.array([0, 1, 1])),
               3: (["Abra"], numpy.array([1])),
               4: (["Squirtle"], numpy.array([1, 0, 1]))}

    model.encodeSamples(samples)
    for i in xrange(len(samples)):
      model.trainModel(i)

    self.assertSequenceEqual([0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4],
        model.sampleReference, "List of indices for samples trained on does "
        "not match the expected.")

    sortedDistances = model.queryModel("Bulbasaur", False)
    
    expectedOrder = [4, 2, 1, 0, 3]
    expectedDistances = [0.47368422, 0.5, 0.81578946, 0.97368419, 0.97368419]
    for i, (sample, distance) in enumerate(zip(expectedOrder, expectedDistances)):
      self.assertEqual(sample, sortedDistances[i][0],
        "Query results do not give expected order of sample indices.")
      self.assertAlmostEqual(distance, sortedDistances[i][1], 5,
        "Query results do not give expected sample distances.")


if __name__ == "__main__":
  unittest.main()
