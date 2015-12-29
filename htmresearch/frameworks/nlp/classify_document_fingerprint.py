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
      "k": 9,
      "distanceMethod": "rawOverlap",
      "maxCategoryCount": 100
    }
  }
}


class ClassificationModelDocumentFingerprint(ClassificationModel):
  """
  Classify documents using a KNN classifier and CIO fingerprints created from
  a full document at time, rather than individual words/tokens.
  """

  def __init__(self,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               k=3,
               verbosity=1,
               numLabels=3):
    """
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.
    @param k                  (int)     The k for KNN classifier
    @param numLabels          (int)     The maximum number of categories

    Note classifierMetric is not specified here as it is in other models. This
    is done in the network config file.
    """
    super(ClassificationModelDocumentFingerprint, self).__init__(
      verbosity=verbosity, numLabels=numLabels)

    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
    self.currentDocument = None
    self._initModel(k)
    self.learningRegions = self._getLearningRegions()



  def getClassifier(self):
    """
    Returns the classifier for the model.
    """
    return self.classifierRegion.getSelf().getAlgorithmInstance()


  def _initModel(self, k):
    """
    Initialize the network; self.networdDataPath must already be set.
    """
    root = os.path.dirname(os.path.realpath(__file__))
    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         cacheDir=os.path.join(root, "CioCache"),
                         retina=self.retina,
                         fingerprintType=EncoderTypes.document,
                         apiKey=self.apiKey)

    modelConfig["classifierRegionConfig"]["regionParams"]["k"] = k
    modelConfig["classifierRegionConfig"]["regionParams"][
                "maxCategoryCount"] = self.numLabels
    self.networkConfig = modelConfig
    self.network = configureNetwork(None, self.networkConfig, encoder)

    # This encoder specifies the LanguageSensor output width.
    # Always a sensor and classifier region.
    self.sensorRegion = self.network.regions[
      self.networkConfig["sensorRegionConfig"].get("regionName")]
    self.classifierRegion = self.network.regions[
      self.networkConfig["classifierRegionConfig"].get("regionName")]


  def _getLearningRegions(self):
    """Return tuple of the network's region objects that learn."""
    learningRegions = []
    for region in self.network.regions.values():
      spec = region.getSpec()
      if spec.parameters.contains('learningMode'):
        learningRegions.append(region)

    return learningRegions


  def reset(self):
    """
    Issue a reset signal to the model. The assumption is that a sequence has
    just ended and a new sequence is about to begin.  The default behavior is
    to do nothing - not all subclasses may re-implement this.
    """
    # TODO: Introduce a consistent reset method name
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

    # Accumulate text
    if self.currentDocument is None:
      self.currentDocument = [token]
    else:
      self.currentDocument.append(token)

    # If reset issued, train on this document
    if reset == 1:
      document = " ".join(self.currentDocument)
      sensor = self.sensorRegion.getSelf()
      sensor.addDataToQueue(document, labels, sequenceId, reset)

      for region in self.learningRegions:
        region.setParameter("learningMode", True)
      self.network.run(1)

      self.currentDocument = None

      # Print the outputs of each region
      if self.verbosity >= 2:
        print "Training with document:",document
        self.printRegionOutputs()


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
      sensor.addDataToQueue(document, [None], -1, 0)
      self.network.run(1)
      if reset == 1:
        self.reset()

      if self.verbosity >= 2:
        print "Classifying document:",document
        self.printRegionOutputs()

      self.currentDocument = None
      return self.classifierRegion.getOutputData("categoriesOut")[0:self.numLabels]

    else:
      return numpy.zeros(self.numLabels)


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
