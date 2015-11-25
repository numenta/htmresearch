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

import cPickle as pkl
import numpy
import operator
import os

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork)
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.support.network_text_data_generator import NetworkDataGenerator
from nupic.data.file_record_stream import FileRecordStream



class ClassificationModelHTM(ClassificationModel):
  """Class to run the classification experiments with HTM network models."""

  def __init__(self,
               networkConfig,
               inputFilePath,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               verbosity=1,
               numLabels=3,
               modelDir="ClassificationModelHTM",
               prepData=True,
               stripCats=False):
    """
    @param networkConfig      (str)     Path to JSON of network configuration,
                                        with region parameters.
    @param inputFilePath      (str)     Path to data file.
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.
    @param prepData           (bool)    Prepare the input data into network API
                                        format.
    @param stripCats          (bool)    Remove the categories and replace them
                                        with the sequence_Id.
    See ClassificationModel for remaining parameters.
    """
    super(ClassificationModelHTM, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    self.networkConfig = networkConfig
    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey

    self.networkDataGen = NetworkDataGenerator()
    if prepData:
      self.networkDataPath = self.prepData(inputFilePath, stripCats=stripCats)
    else:
      self.networkDataPath = inputFilePath

    self.network = self.initModel()
    self.learningRegions = self._getLearningRegions()

    # Always a sensor and classifier region.
    self.sensorRegion = self.network.regions[
      self.networkConfig["sensorRegionConfig"].get("regionName")]
    self.classifierRegion = self.network.regions[
      self.networkConfig["classifierRegionConfig"].get("regionName")]


  def prepData(self, dataPath, ordered=False, stripCats=False, **kwargs):
    """
    Generate the data in network API format.

    @param dataPath          (str)  Path to input data file; format as expected
                                    by NetworkDataGenerator.
    @param ordered           (bool) Keep order of data, or randomize.
    @param stripCats         (bool) Remove the categories and replace them with
                                    the sequence_Id.
    @return networkDataPath  (str)  Path to data formtted for network API.
    """
    networkDataPath = self.networkDataGen.setupData(
      dataPath, self.numLabels, ordered, stripCats, **kwargs)

    return networkDataPath


  def initModel(self):
    """
    Initialize the network; self.networdDataPath must already be set.
    """
    recordStream = FileRecordStream(streamID=self.networkDataPath)
    root = os.path.dirname(os.path.realpath(__file__))
    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         cacheDir=os.path.join(root, "CioCache"),
                         retina=self.retina,
                         apiKey=self.apiKey)

    # This encoder specifies the LanguageSensor output width.
    return configureNetwork(recordStream, self.networkConfig, encoder)


  def _getLearningRegions(self):
    """Return tuple of the network's region objects that learn."""
    learningRegions = []
    for region in self.network.regions.values():
      try:
        _ = region.getParameter("learningMode")
        learningRegions.append(region)
      except:
        continue

    return learningRegions


  # TODO: is this still needed?
  def encodeSample(self, sample):
    """
    Put each token in its own dictionary with its bitmap
    @param sample     (list)            Tokenized sample, where each item is a
                                        string token.
    @return           (list)            The sample text, sparsity, and bitmap
                                        for each token. Since the network will
                                        do the actual encoding, the bitmap and
                                        sparsity will be None
    Example return list:
      [{
        "text": "Example text",
        "sparsity": 0.0,
        "bitmap": None
      }]
    """
    return [{"text": t,
             "sparsity": None,
             "bitmap": None} for t in sample]


  def resetModel(self):
    """
    Reset the model by creating a new network since the network API does not
    support resets.
    """
    # TODO: test this works as expected
    self.network = self.initModel()


  def saveModel(self):
    try:
      if not os.path.exists(self.modelDir):
        os.makedirs(self.modelDir)
      networkPath = os.path.join(self.modelDir, "network.nta")
      self.network.save(networkPath)
      # with open(networkPath, "wb") as f:
      #   pkl.dump(self, f)
      if self.verbosity > 0:
        print "Model saved to \'{}\'.".format(networkPath)
    except IOError as e:
      print "Could not save model to \'{}\'.".format(networkPath)
      raise e


  def trainModel(self, iterations=1):
    """
    Run the network with all regions learning.
    Note self.sampleReference doesn't get populated b/c in a network model
    there's a 1-to-1 mapping of training samples.
    """
    for region in self.learningRegions:
      # if region.name == 'UP': continue
      region.setParameter("learningMode", True)

    self.network.run(iterations)


  def trainNetwork(self, iterations):
    """Run the network with all regions learning but the classifier."""
    for region in self.learningRegions:
      if region.name == "classifier":
        region.setParameter("learningMode", False)
      else:
        region.setParameter("learningMode", True)

    self.network.run(iterations)


  def classifyNetwork(self, iterations):
    """
    For running after the network has been trained by trainNetwork(), this
    populates the KNN prototype space with the final network representations.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", False)

    sensor = self.sensorRegion.getSelf()
    sensor.rewind()

    self.classifierRegion.setParameter("learningMode", True)
    self.classifierRegion.setParameter("inferenceMode", True)

    sequenceIds = []
    for _ in xrange(iterations):
      self.network.run(1)
      sequenceIds.append(sensor.getOutputValues("sequenceIdOut")[0])

    return sequenceIds


  def inferNetwork(self, iterations, fileRecord=None, learn=False):
    """
    Run the network to infer distances to the classified samples.

    @param fileRecord (str)     If you want to change the file record stream.
    @param learn      (bool)    The classifier will learn the inferred sequnce.
    """
    if fileRecord:
      self.swapRecordStream(fileRecord)

    self.classifierRegion.setParameter("learningMode", learn)
    self.classifierRegion.setParameter("inferenceMode", True)

    sampleDistances = None
    for i in xrange(iterations):
      self.network.run(1)
      inferenceValues = self.classifierRegion.getOutputData("categoriesOut")
      # Sum together the inferred distances for each word of the sequence.
      if sampleDistances is None:
        sampleDistances = inferenceValues
      else:
        sampleDistances += inferenceValues

    return sampleDistances


  def swapRecordStream(self, dataPath):
    """Change the data source for the network's sensor region."""
    recordStream = FileRecordStream(streamID=dataPath)
    sensor = self.sensorRegion.getSelf()
    sensor.dataSource = recordStream  # TODO: implement this in network API


  def testModel(self, seed=42):
    """
    Test the classifier region on the input sample. Call this method for each
    word of a sequence. The random seed is used in getWinningLabels().

    @return           (numpy array)   numLabels most-frequent classifications
                                      for the data samples; int or empty.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", False)
    self.classifierRegion.setParameter("inferenceMode", True)

    self.network.run(1)

    inference = self._getClassifierInference(seed)
    activityBitmap = self.classifierRegion.getInputData("bottomUpIn")

    return inference, activityBitmap


  def _getClassifierInference(self, seed):
    """Return output categories from the classifier region."""
    relevantCats = self.classifierRegion.getParameter("categoryCount")

    if self.classifierRegion.type == "py.KNNClassifierRegion":
      # max number of inferences = k
      inferenceValues = self.classifierRegion.getOutputData(
        "categoriesOut")[:relevantCats]
      return self.getWinningLabels(inferenceValues, seed)

    elif self.classifierRegion.type == "py.CLAClassifierRegion":
      # TODO: test this
      return self.classifierRegion.getOutputData("categoriesOut")[:relevantCats]


  def queryModel(self, query, preprocess=False):
    """
    Run the query through the network, getting the classifer region's inferences
    for all words of the query sequence.
    @return       (list)          Two-tuples of sequence ID and distance, sorted
                                  closest to farthest from the query.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", False)
    self.classifierRegion.setParameter("inferenceMode", True)

    # Put query text in LanguageSensor data format.
    queryDicts = self.networkDataGen.generateSequence(query, preprocess)

    sampleDistances = None

    for qD in queryDicts:
      # Sum together the inferred distances for each word of the query sequence.
      sensor.queue.appendleft(qD)
      self.network.run(1)
      inferenceValues = self.classifierRegion.getOutputData("categoriesOut")
      if sampleDistances is None:
        sampleDistances = inferenceValues
      else:
        sampleDistances += inferenceValues

    catCount = self.classifierRegion.getParameter("categoryCount")
    # The use of numpy.lexsort() here is to first sort by labelFreq, then sort
    # by random values; this breaks ties in a random manner.
    randomValues = numpy.random.random(catCount)
    sortedSamples = numpy.lexsort((randomValues, sampleDistances[:catCount]))
    qTuple = [(a, b) for a, b in zip(sortedSamples, sampleDistances[:catCount])]

    return sorted(qTuple, key=operator.itemgetter(1))
