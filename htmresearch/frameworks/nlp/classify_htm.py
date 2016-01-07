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

from tabulate import tabulate

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
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               **kwargs):
    """
    @param networkConfig      (dict)    Network configuration dict with region
                                        parameters.
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.

    See ClassificationModel for remaining parameters.
    """
    super(ClassificationModelHTM, self).__init__(**kwargs)

    self.networkConfig = networkConfig
    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
    self.network = self.initModel()
    self._initializeRegionHelpers()


  def getClassifier(self):
    """
    Returns the classifier for the model.
    """
    return self.classifierRegion.getSelf().getAlgorithmInstance()


  def initModel(self):
    """
    Initialize the network; self.networdDataPath must already be set.
    """
    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         retina=self.retina,
                         apiKey=self.apiKey,
                         verbosity=self.verbosity-1)

    # This encoder specifies the LanguageSensor output width.
    return configureNetwork(None, self.networkConfig, encoder)


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

    self.network.enableProfiling()


  def tokenize(self, text):
    """
    Given a bunch of text (could be several sentences) return a single list
    containing individual tokens.  It currently uses the CIO tokenize function
    and ignores filterText.

    @param text         (str)     A bunch of text.
    @return             (list)    A list of text tokens.
    """
    encoder = self.sensorRegion.getSelf().encoder
    sentenceList = encoder.client.tokenize(text)
    tokenList = []
    for sentence in sentenceList:
      tokenList.extend(sentence.split(","))
    return tokenList


  def reset(self):
    """
    Issue a reset signal to the model. The assumption is that a sequence has
    just ended and a new sequence is about to begin.  The default behavior is
    to do nothing - not all subclasses may re-implement this.
    """
    # TODO: Introduce a consistent reset method name.
    for region in self.learningRegions:
      if region.type == "py.TemporalPoolerRegion":
        region.executeCommand(["reset"])
      elif region.type == "py.TPRegion":
        region.executeCommand(["resetSequenceStates"])


  def trainToken(self, token, labels, sampleId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param sampleId   (int)  An integer ID associated with this token and its
                             sequence (document).
    @param reset      (int)  Should be 0 or 1. If 1, assumes we are at the
                             end of the document.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", True)
      region.setParameter("inferenceMode", True)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token, labels, sequenceId=sampleId, reset=0)
    self.network.run(1)

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if reset == 1:
      self.reset()


  def inferToken(self, token, reset=0, sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and a list of sampleIds and distances.
    Repeated sampleIds are NOT removed from the results.

    @param token    (str)     The text token to train on
    @param reset    (int)     Should be 0 or 1. If 1, assumes we are at the
                              end of a sequence. A reset signal will be issued
                              after the model has been trained on this token.
    @param sortResults (bool) If true the list of sampleIds and distances
                              will be sorted in order of increasing distances.

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this token belongs to the
                           i'th category. An array containing all zeros
                           implies no decision could be made.
             (list)        A list of sampleIds
             (numpy array) An array of distances from each stored sample
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", False)
      region.setParameter("inferenceMode", True)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token, [None], sequenceId=-1, reset=0)
    self.network.run(1)

    dist = self.classifierRegion.getSelf().getLatestDistances()

    categoryVotes = self.classifierRegion.getOutputData(
        "categoriesOut")[0:self.numLabels]

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if reset == 1:
      self.reset()

    # Accumulate the ids. Sort results if requested
    classifier = self.getClassifier()
    if sortResults:
      idList = []
      sortedIndices = dist.argsort()
      for i in sortedIndices:
        idList.append(classifier.getPartitionId(i))
      sortedDistances = dist[sortedIndices]
      return categoryVotes, idList, sortedDistances

    else:
      idList = []
      for i in range(len(dist)):
        idList.append(classifier.getPartitionId(i))
      return categoryVotes, idList, dist


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


  def dumpProfile(self):
    """
    Print region profiling information in a nice format.
    """
    print "Profiling information for {}".format(type(self).__name__)
    totalTime = 0.000001
    for region in self.network.regions.values():
      timer = region.computeTimer
      totalTime += timer.getElapsed()

    profileInfo = []
    for region in self.network.regions.values():
      timer = region.computeTimer
      profileInfo.append([region.name,
                          timer.getStartCount(),
                          timer.getElapsed(),
                          100.0*timer.getElapsed()/totalTime])

    profileInfo.append(["Total time", "", totalTime, "100.0"])
    print tabulate(profileInfo, headers=["Region", "Count",
                   "Elapsed", "Percent of total"],
                   tablefmt = "grid")


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
