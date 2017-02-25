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

from htmresearch.frameworks.classification.network_factory import (
  createAndConfigureNetwork)
from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classify_network_api import (
  ClassificationNetworkAPI
)

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


class ClassificationModelDocumentFingerprint(ClassificationNetworkAPI):
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
               k=1,
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


  def _initModel(self, k):
    """
    Initialize the network
    """
    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         retina=self.retina,
                         fingerprintType=EncoderTypes.document,
                         apiKey=self.apiKey,
                         verbosity=self.verbosity-1)

    modelConfig["classifierRegionConfig"]["regionParams"]["k"] = k
    modelConfig["classifierRegionConfig"]["regionParams"][
                "maxCategoryCount"] = self.numLabels
    self.networkConfig = modelConfig
    self.network = createAndConfigureNetwork(None, self.networkConfig, encoder)


  def trainToken(self, token, labels, sampleId, resetSequence=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID. This model buffers the tokens, labels, and IDs until
    resetSequence=1, at which point the model is trained with the buffered data.

    See base class for description of parameters.
    """
    # Accumulate text
    if self.currentDocument is None:
      self.currentDocument = [token]
    else:
      self.currentDocument.append(token)

    # If reset issued, train on this document
    if resetSequence == 1:
      document = " ".join(self.currentDocument)
      sensor = self.sensorRegion.getSelf()
      sensor.addDataToQueue(token=document, categoryList=labels,
                            sequenceId=sampleId, reset=resetSequence)

      for region in self.learningRegions:
        region.setParameter("learningMode", True)
      self.network.run(1)
      self.currentDocument = None

      # Print the outputs of each region
      if self.verbosity >= 2:
        print "Training with document:",document
        print "SequenceId:",sampleId
        if self.verbosity >= 3:
          self.printRegionOutputs()


  def inferToken(self, token, resetSequence=0, returnDetailedResults=False,
                 sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and a list of sampleIds and distances.
    Repeated sampleIds are NOT removed from the results.
    See base class for description of parameters.
    """
    # Accumulate text
    if self.currentDocument is None:
      self.currentDocument = [token]
    else:
      self.currentDocument.append(token)

    # If reset issued, classify this document
    if resetSequence == 1:

      for region in self.learningRegions:
        region.setParameter("learningMode", False)
      document = " ".join(self.currentDocument)
      sensor = self.sensorRegion.getSelf()
      sensor.addDataToQueue(token=document, categoryList=[None],
                            sequenceId=-1, reset=resetSequence)
      self.network.run(1)

      if self.verbosity >= 2:
        print "Classifying document:",document
        self.printRegionOutputs()

      self.currentDocument = None
      categoryVotes = self.classifierRegion.getOutputData(
          "categoriesOut")[0:self.numLabels]

      if returnDetailedResults:
        # Accumulate the ids. Sort results if requested
        dist = self.classifierRegion.getSelf().getLatestDistances()

        classifier = self.getClassifier()
        if sortResults:
          idList = []
          sortedIndices = dist.argsort()
          for i in sortedIndices:
            idList.append(classifier.getPartitionId(i))
          sortedDistances = dist[sortedIndices]
          return categoryVotes, idList, sortedDistances

        else:
          idList = [classifier.getPartitionId(i) for i in xrange(len(dist))]
          return categoryVotes, idList, dist

      else:
        # Non-detailed results
        return categoryVotes, None, None

    else:

      return numpy.zeros(self.numLabels), [], numpy.zeros(0)
