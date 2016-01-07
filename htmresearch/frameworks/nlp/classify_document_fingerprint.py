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

import os
import numpy

from tabulate import tabulate

from nupic.engine import Network

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork)
from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel

modelConfig = {
  "sensorRegionConfig": {
    "regionEnabled": True,
    "regionName": "sensor",
    "regionType": "py.LanguageSensor",
    "regionParams": {
      "verbosity": 0,
      "numCategories": 3
      },
    "encoders": {}
  },
  "classifierRegionConfig": {
    "regionEnabled": True,
    "regionName": "classifier",
    "regionType": "py.KNNClassifierRegion",
    "regionParams": {
      "k": None,  # To be filled in by constructor
      "distanceMethod": "rawOverlap",
      "maxCategoryCount": None,  # To be filled in by constructor
    }
  }
}


class ClassificationModelDocumentFingerprint(ClassificationModel):
  """
  Classify documents using a KNN classifier and CIO fingerprints created from
  a full document at time, rather than individual words/tokens.

  TODO: this class shares a lot of code with ClassificationModelHTM.  We should
  create a mid-level class for those models that use the network API. This class
  can keep all the common code.
  """

  def __init__(self,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               k=3,
               **kwargs):
    """
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.
    @param k                  (int)     The k for KNN classifier

    Note classifierMetric is not specified here as it is in other models. This
    is done in the network config file.
    """
    super(ClassificationModelDocumentFingerprint, self).__init__(**kwargs)

    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
    self.currentDocument = None
    self._initModel(k)
    self._initializeRegionHelpers()


  def getClassifier(self):
    """
    Returns the classifier for the model.
    """
    return self.classifierRegion.getSelf().getAlgorithmInstance()


  def _initModel(self, k):
    """
    Initialize the network
    """
    root = os.path.dirname(os.path.realpath(__file__))
    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         retina=self.retina,
                         fingerprintType=EncoderTypes.document,
                         apiKey=self.apiKey,
                         verbosity=self.verbosity-1)

    modelConfig["classifierRegionConfig"]["regionParams"]["k"] = k
    modelConfig["classifierRegionConfig"]["regionParams"][
                "maxCategoryCount"] = self.numLabels
    self.networkConfig = modelConfig
    self.network = configureNetwork(None, self.networkConfig, encoder)


  def _initializeRegionHelpers(self):
    """
    Set helper member variables once network has been initialized. This will
    also be called from _deSerializeExtraData()
    """
    learningRegions = []
    for region in self.network.regions.values():
      spec = region.getSpec()
      if spec.parameters.contains("learningMode"):
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


  def tokenize(self, text, preprocess=False):
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
    just ended and a new sequence is about to begin.
    """
    # TODO: Introduce a consistent reset method name in Regions
    for r in self.learningRegions:
      if r.type == "py.TemporalPoolerRegion":
        r.executeCommand(["reset"])
      elif r.type == "py.TPRegion":
        r.executeCommand(["resetSequenceStates"])


  def trainToken(self, token, labels, sampleId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID. This model buffers the tokens, etc. until reset=1 at which
    point the model is trained with the buffered tokens and the labels and
    sampleId sent in that call.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param sampleId   (int)  An integer ID associated with this token and its
                             sequence (document).
    @param reset      (int)  Should be 0 or 1. If 1, assumes we are at the
                             end of the document.
    """
    # Accumulate text
    if self.currentDocument is None:
      self.currentDocument = [token]
    else:
      self.currentDocument.append(token)

    # If reset issued, train on this document
    if reset == 1:
      document = " ".join(self.currentDocument)
      sensor = self.sensorRegion.getSelf()
      sensor.addDataToQueue(document, labels, sampleId, reset)

      for region in self.learningRegions:
        region.setParameter("learningMode", True)
      self.network.run(1)
      self.reset()
      self.currentDocument = None

      # Print the outputs of each region
      if self.verbosity >= 2:
        print "Training with document:",document
        print "SequenceId:",sampleId
        if self.verbosity >= 3:
          self.printRegionOutputs()


  def inferToken(self, token, reset=0, sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and a list of sampleIds and distances.
    Repeated sampleIds are NOT removed from the results.

    @param token    (str)  The text token to train on
    @param reset    (int)  Should be 0 or 1. If 1, assumes we are at the
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
    # Accumulate text
    if self.currentDocument is None:
      self.currentDocument = [token]
    else:
      self.currentDocument.append(token)

    # If reset issued, classify this document
    if reset == 1:

      for region in self.learningRegions:
        region.setParameter("learningMode", False)
        region.setParameter("inferenceMode", True)
      document = " ".join(self.currentDocument)
      sensor = self.sensorRegion.getSelf()
      sensor.addDataToQueue(token=document, categoryList=[None],
                            sequenceId=-1, reset=0)
      self.network.run(1)

      dist = self.classifierRegion.getSelf().getLatestDistances()

      if self.verbosity >= 2:
        print "Classifying document:",document
        self.printRegionOutputs()

      self.currentDocument = None
      categoryVotes = self.classifierRegion.getOutputData(
          "categoriesOut")[0:self.numLabels]

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

    else:

      return numpy.zeros(self.numLabels), [], numpy.zeros(0)


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
