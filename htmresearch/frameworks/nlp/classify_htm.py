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

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork)
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classify_network_api import (
  ClassificationNetworkAPI
)


class ClassificationModelHTM(ClassificationNetworkAPI):
  """Classify text using generic network-API based models."""

  def __init__(self,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               **kwargs):
    """
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.

    See ClassificationModel for remaining parameters.
    """
    super(ClassificationModelHTM, self).__init__(**kwargs)

    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
    self.network = self.initModel()
    self._initializeRegionHelpers()


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


  def trainToken(self, token, labels, sampleId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sequence ID.

    See base class for description of parameters.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", True)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token,
                          categoryList=labels,
                          sequenceId=sampleId, reset=0)
    self.network.run(1)

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if reset == 1:
      self.reset()


  def inferToken(self, token, reset=0, returnDetailedResults=False,
                 sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and a list of sampleIds and distances.
    Repeated sampleIds are NOT removed from the results.

    See base class for description of parameters.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", False)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token,
                          categoryList=[None],
                          sequenceId=-1, reset=0)
    self.network.run(1)

    dist = self.classifierRegion.getSelf().getLatestDistances()

    categoryLikelihoods = self.classifierRegion.getOutputData(
        "categoriesOut")[0:self.numLabels]

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if reset == 1:
      self.reset()

    # If detailed results are not requested just return the category votes
    if not returnDetailedResults:
      return categoryLikelihoods, None, None

    # Unsorted detailed results are easy
    classifier = self.getClassifier()
    partitionIdList = classifier.getPartitionIdPerPattern()
    if not sortResults:
      return categoryLikelihoods, partitionIdList, dist

    # Sort results if requested
    sortedIndices = dist.argsort()
    sortedDistances = dist[sortedIndices]
    sortedSampleIdList = [partitionIdList[i] for i in sortedIndices]

    return categoryLikelihoods, sortedSampleIdList, sortedDistances