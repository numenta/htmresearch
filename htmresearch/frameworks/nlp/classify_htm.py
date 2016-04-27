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
               maxSparsity=1.0,
               cacheRoot=None,
               **kwargs):
    """
    @param retinaScaling      (float)   Scales the dimensions of the SDRs.
    @param retina             (str)     Name of Cio retina.
    @param apiKey             (str)     Key for Cio API.
    @param maxSparsity        (float)   The maximum sparsity of the CIO bitmap.
    @param cacheRoot          (str)     Directory for caching Cio encodings.

    See ClassificationModel for remaining parameters.
    """
    super(ClassificationModelHTM, self).__init__(**kwargs)

    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
    self.maxSparsity = maxSparsity

    self.network = self._initModel(cacheRoot)
    self._initializeRegionHelpers()


  def _initModel(self, cacheRoot):
    """
    Initialize the network; self.networdDataPath must already be set.
    """
    encoder = CioEncoder(retinaScaling=self.retinaScaling,
                         retina=self.retina,
                         apiKey=self.apiKey,
                         maxSparsity=self.maxSparsity,
                         verbosity=self.verbosity-1,
                         cacheDir=cacheRoot)

    # This encoder specifies the LanguageSensor output width.
    return configureNetwork(None, self.networkConfig, encoder)


  def trainToken(self, token, labels, tokenId, resetSequence=0):
    """
    Train the model with the given text token, associated labels, and ID
    associated with this token.

    See base class for description of parameters.
    """
    for region in self.learningRegions:
      region.setParameter("learningMode", True)
    sensor = self.sensorRegion.getSelf()
    sensor.addDataToQueue(token,
                          categoryList=labels,
                          sequenceId=tokenId,
                          reset=resetSequence)
    self.network.run(1)

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if resetSequence == 1:
      self.reset()


  def inferToken(self, token, resetSequence=0, returnDetailedResults=False,
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
                          sequenceId=-1, reset=resetSequence)
    self.network.run(1)

    dist = self.classifierRegion.getSelf().getLatestDistances()

    categoryLikelihoods = self.classifierRegion.getOutputData(
        "categoriesOut")[0:self.numLabels]

    # Print the outputs of each region
    if self.verbosity >= 2:
      self.printRegionOutputs()

    if resetSequence == 1:
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
