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
import operator
import os

from nupic.data.file_record_stream import FileRecordStream
from nupic.engine import Network

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork)
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.support.network_text_data_generator import NetworkDataGenerator



class ClassificationModelHTM(ClassificationModel):
  """Classify text using generic network-API based models."""

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
               stripCats=False,
               cacheRoot=None):
    """
    @param networkConfig      (dict)    Network configuration dict with region
                                        parameters.
    @param inputFilePath      (str)     Path to data file.
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.
    @param prepData           (bool)    Prepare the input data into network API
                                        format.
    @param stripCats          (bool)    Remove the categories and replace them
                                        with the sequence_Id.
    @param cacheRoot          (str)     Root cache directory for CioEncoder
    See ClassificationModel for remaining parameters.

    Note classifierMetric is not specified here as it is in other models. This
    is done in the network config file.
    """
    super(ClassificationModelHTM, self).__init__(
      verbosity=verbosity, numLabels=numLabels, modelDir=modelDir)

    self.networkConfig = networkConfig
    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
    self.inputFilePath = inputFilePath

    self.networkDataGen = NetworkDataGenerator()
    if prepData:
      self.networkDataPath = self.prepData(
          self.inputFilePath, stripCats=stripCats)
    else:
      self.networkDataPath = self.inputFilePath

    self.cacheRoot = cacheRoot or os.path.dirname(os.path.realpath(__file__))

    self.network = self.initModel()
    self._initializeRegionHelpers()



  def getClassifier(self):
    """
    Returns the classifier for the model.
    """
    return self.classifierRegion.getSelf().getAlgorithmInstance()


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
    if self.networkDataPath is not None:
      recordStream = FileRecordStream(streamID=self.networkDataPath)
    else:
      recordStream = None

    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         cacheDir=os.path.join(self.cacheRoot, "CioCache"),
                         retina=self.retina,
                         apiKey=self.apiKey)

    # This encoder specifies the LanguageSensor output width.
    return configureNetwork(recordStream, self.networkConfig, encoder)


  def _initializeRegionHelpers(self):
    """
    Set helper member variables once network has been initialized. This will
    also be called from _deSerializeExtraData()
    """
    learningRegions = []
    for region in self.network.regions.values():
      spec = region.getSpec()
      if spec.parameters.contains('learningMode'):
        learningRegions.append(region)

    # Always a sensor and classifier region.
    self.sensorRegion = self.network.regions[
      self.networkConfig["sensorRegionConfig"].get("regionName")]
    self.classifierRegion = self.network.regions[
      self.networkConfig["classifierRegionConfig"].get("regionName")]

    # There is sometimes a TP region
    self.tpRegion = None
    if self.networkConfig.has_key("tpRegionConfig"):
      self.tpRegion = self.network.regions[
        self.networkConfig["tpRegionConfig"].get("regionName")]

    self.learningRegions = learningRegions


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


  def saveModel(self, trial=None):
    try:
      if not os.path.exists(self.modelDir):
        os.makedirs(self.modelDir)
      if trial:
        netPath = os.path.join(self.modelDir, "network_{}.nta".format(trial))
      else:
        netPath = os.path.join(self.modelDir, "network.nta")
      self.network.save(netPath)
      if self.verbosity > 0:
        print "Model saved to '{}'.".format(netPath)
    except IOError as e:
      print "Could not save model to '{}'.".format(netPath)
      raise e


  def trainModel(self, iterations=1):
    """
    Run the network with all regions learning.
    Note self.sampleReference doesn't get populated b/c in a network model
    there's a 1-to-1 mapping of training samples.
    """
    for region in self.learningRegions:
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

    sensor = self.sensorRegion.getSelf()
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
    # The use of numpy.lexsort() here is to first sort by randomValues, then
    # sort by random values; this breaks ties in a random manner.
    randomValues = numpy.random.random(catCount)
    sortedSamples = numpy.lexsort((randomValues, sampleDistances[:catCount]))
    qTuple = [(a, b) for a, b in zip(sortedSamples, sampleDistances[:catCount])]

    return sorted(qTuple, key=operator.itemgetter(1))


  def reset(self):
    """
    Issue a reset signal to the model. The assumption is that a sequence has
    just ended and a new sequence is about to begin.  The default behavior is
    to do nothing - not all subclasses may re-implement this.
    """
    # TODO: Introduce a consistent reset method name.
    for r in self.learningRegions:
      if r.type == 'py.TemporalPoolerRegion':
        r.executeCommand(['reset'])
      elif r.type == 'py.TPRegion':
        r.executeCommand(['resetSequenceStates'])


  def trainText(self, token, labels, sequenceId=None, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token. If the list is empty, the
                             classifier will not be trained.
    @param sequenceId (int)  An integer ID associated with this token and its
                             sequence (document).
    @param reset      (int)  Should be 0 or 1. If 1, assumes we are at the
                             beginning of a new sequence.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", True)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token, labels, sequenceId, 0)
    self.network.run(1)

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if reset == 1:
      self.reset()


  def classifyText(self, token, reset=0):
    """
    Classify the token and return a list of the best classifications.

    @param token    (str)  The text token to train on
    @param reset    (int)  Should be 0 or 1. If 1, assumes we are at the
                           end of a sequence. A reset signal will be issued
                           after the model has been trained on this token.

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this sample belongs to the
                           i'th category. An array containing all zeros
                           implies no decision could be made.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", False)
      region.setParameter("inferenceMode", True)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token, [None], -1, 0)
    self.network.run(1)

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if reset == 1:
      self.reset()

    return self.classifierRegion.getOutputData("categoriesOut")[0:self.numLabels]


  def printRegionOutputs(self):
    """
    Print the outputs of regions to console for debugging, depending on
    verbosity level.
    """

    print "================== HTM Debugging output:"
    print "Sensor output:",
    print self.sensorRegion.getOutputData("dataOut").nonzero()
    print "Sensor categoryOut:",
    print self.sensorRegion.getOutputData("categoryOut")

    if self.verbosity >= 3:
      if self.tpRegion is not None:
        print "TP region input:",
        print self.tpRegion.getInputData("activeCells").nonzero()
        print "TP region output:",
        print self.tpRegion.getOutputData("mostActiveCells").nonzero()

      print "Classifier bottomUpIn: ",
      print self.classifierRegion.getInputData("bottomUpIn").nonzero()
      print "Classifier categoryIn: ",
      print self.classifierRegion.getInputData("categoryIn")[0:self.numLabels]

    print "Classifier categoriesOut: ",
    print self.classifierRegion.getOutputData("categoriesOut")[0:self.numLabels]
    print "Classifier categoryProbabilitiesOut",
    print self.classifierRegion.getOutputData("categoryProbabilitiesOut")[0:self.numLabels]


  def __getstate__(self):
    """
    Return serializable state.  This function will return a version of the
    __dict__ with data that shouldn't be pickled stripped out. For example,
    Network API instances are stripped out because they have their own
    serialization mechanism.

    See also: _serializeExtraData()
    """
    state = self.__dict__.copy()
    # Remove member variables that we can't pickle
    state.pop("network")
    state.pop("sensorRegion")
    state.pop("classifierRegion")
    state.pop("tpRegion")
    state.pop("learningRegions")
    state.pop("networkDataGen")

    return state


  def _serializeExtraData(self, extraDataDir):
    """
    Protected method that is called during serialization with an external
    directory path. We override it here to save the Network API instance.

    @param extraDataDir (string) Model's extra data directory path
    """
    self.network.save(os.path.join(extraDataDir, "network.nta"))


  def _deSerializeExtraData(self, extraDataDir):
    """
    Protected method that is called during deserialization (after __setstate__)
    with an external directory path. We override it here to load the Network API
    instance.

    @param extraDataDir (string) Model's extra data directory path
    """
    self.network = Network(os.path.join(extraDataDir, "network.nta"))
    self._initializeRegionHelpers()
    self.networkDataGen = NetworkDataGenerator()
