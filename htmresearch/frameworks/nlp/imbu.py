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

"""
Class for running models in Imbu app.

The script is runnable to demo Imbu functionality (from repo's base dir):

  python htmresearch/frameworks/nlp/imbu.py \
    --dataPath projects/nlp/data/sample_reviews/sample_reviews.csv \
    --modelName Keywords

  python htmresearch/frameworks/nlp/imbu.py \
    --dataPath projects/nlp/data/sample_reviews/sample_reviews_unlabeled.csv \
    --modelName HTMNetwork
"""

import argparse
import os

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.support.csv_helper import readCSV
from htmresearch.frameworks.nlp.model_factory import (
  ClassificationModelTypes,
  createModel,
  getNetworkConfig)



class ModelSimilarityMetrics(object):
  pctOverlapOfInput = "pctOverlapOfInput"
  rawOverlap = "rawOverlap"



class ImbuError(Exception):
  pass



class ImbuUnableToLoadModelError(ImbuError):
  def __init__(self, exc):
    self.exc = exc



def _loadNetworkConfig():
  """ Load network config by calculating path relative to this file, and load
  with htmresearch.frameworks.nlp.model_factory.getNetworkConfig()
  """
  root = (
    os.path.dirname(
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(
            os.path.realpath(__file__)
          )
        )
      )
    )
  )
  return getNetworkConfig(
    os.path.join(root, "projects/nlp/data/network_configs/imbu.json"))



class ImbuModels(object):

  defaultSimilarityMetric = ModelSimilarityMetrics.pctOverlapOfInput
  defaultModelType = ClassificationModelTypes.CioWordFingerprint
  defaultRetina = "en_associative"

  # Set of classification model types that accept CioEncoder kwargs
  requiresCIOKwargs = {
    ClassificationModelTypes.CioWordFingerprint,
    ClassificationModelTypes.HTMNetwork,
    ClassificationModelTypes.DocumentFingerPrint
  }

  def __init__(self, cacheRoot, dataPath,  modelSimilarityMetric=None,
      apiKey=None, retina=None):
    self.cacheRoot = cacheRoot
    self.modelSimilarityMetric = (
      modelSimilarityMetric or self.defaultSimilarityMetric
    )
    self.dataPath = dataPath
    self.dataDict = self._loadTrainingData()
    self.apiKey = apiKey
    self.retina = retina or self.defaultRetina


  def __repr__(self):
    return ("Imbu<cacheRoot={cacheRoot}, dataPath={dataPath}, "
            "modelSimilarityMetric={modelSimilarityMetric}, "
            "apiKey={apiKey}, retina={retina}>"
            .format(**self.__dict__))


  def _defaultModelFactoryKwargs(self):
    """ Default kwargs common to all model types.

    For Imbu to function unsupervised, numLabels is set to 1 in order to use
    unlabeled data for querying and still comply with models' inference logic.
    """
    return dict(
      numLabels=1,
      classifierMetric=self.modelSimilarityMetric)


  def _modelFactory(self, modelType, savePath, **kwargs):
    """ Imbu model factory.  Returns a concrete instance of a classification
    model given a model type name and kwargs.
    """

    modelType = modelType or self.defaultModelType

    kwargs.update(modelDir=savePath, **self._defaultModelFactoryKwargs())

    if modelType in self.requiresCIOKwargs:
      # Model type requires Cortical.io credentials
      kwargs.update(retina=self.retina, apiKey=self.apiKey)

    if modelType == "CioWordFingerprint":
      kwargs.update(fingerprintType=EncoderTypes.word,
                    cacheRoot=self.cacheRoot)

    elif modelType == "CioDocumentFingerprint":
      kwargs.update(fingerprintType=EncoderTypes.document,
                    cacheRoot=self.cacheRoot)

    elif modelType == "HTMNetwork":
      kwargs.update(networkConfig=_loadNetworkConfig(),
                    inputFilePath=None,
                    prepData=False,
                    retinaScaling=1.0)

    elif modelType == "Keywords":
      # k should be > the number of data samples because the Keywords model
      # looks for exact matching tokens, so we want to consider all data
      # samples in the search of k nearest neighbors.
      kwargs.update(k=10 * len(self.dataDict.keys()))

    model = createModel(modelType, **kwargs)

    model.verbosity = 0

    return model


  def createModel(self, modelType, loadPath, savePath, *modelFactoryArgs,
      **modelFactoryKwargs):
    """ Creates a new model and trains it, or loads a previously trained model
    from specified loadPath.
    """

    if loadPath:
      # User has explicitly specified a load path and expects a model to exist
      try:
        model = ClassificationModel.load(loadPath)

        if not isinstance(model, getattr(ClassificationModelTypes, modelType)):
          raise ImbuError("Model ({}) loaded from {} is not the same type as "
                          "requested ({})."
                          .format(repr(model), loadPath, modelType))

      except IOError as exc:
        # Model was not found, user may have specified incorrect path, DO NOT
        # attempt to create a new model and raise an exception
        raise ImbuUnableToLoadModelError(exc)
    else:
      # User has not specified a load path, defer to default case and
      # gracefully create a new model
      try:
        model = ClassificationModel.load(loadPath)
      except IOError as exc:
        model = self._modelFactory(modelType,
                                   savePath,
                                   *modelFactoryArgs,
                                   **modelFactoryKwargs)
        self.train(model, savePath)


    return model


  def _loadTrainingData(self):
    """ Load training data.
    """
    return readCSV(self.dataPath,
                   numLabels=0) # 0 to train models in unsupervised fashion


  def train(self, model, savePath=None):
    """ Train model, generically assigning category 0 to all documents.
    """
    for seqId, (text, _, _) in enumerate(self.dataDict.values()):
      model.trainDocument(text, [0], seqId)

    if savePath:
      self.save(model, savePath)


  @staticmethod
  def query(model, query, returnDetailedResults=True, sortResults=True):
    """ Query classification model.
    """
    return model.inferDocument(query,
                               returnDetailedResults=returnDetailedResults,
                               sortResults=sortResults)


  def save(self, model, savePath=None):
    """ Save classification model.
    """
    model.save(savePath)


  def formatResults(self, model, distanceArray, idList):
    """ Format distances to reflect the pctOverlapOfInput metric, return a list
    of results.
    """
    formattedDistances = (1.0 - distanceArray) * 100

    indexingFactor = getattr(model, "indexingFactor", 1)

    results = []
    for protoId, dist in zip(idList, formattedDistances):
      # get the sampleId from the protoId
      wordId = protoId % indexingFactor
      sampleId = (protoId - wordId) / indexingFactor
      results.append({"sampleId": sampleId,
                      "wordId": wordId,
                      "text": self.dataDict[sampleId][0],
                      "score": dist.item()})

    return results



def main():
  """ Main entry point for Imbu CLI utility to demonstration Imbu functionality.
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--cacheRoot",
                      help="Root directory in which to cache encodings")
  parser.add_argument("--modelSimilarityMetric",
                      default=ImbuModels.defaultSimilarityMetric,
                      help=("Classifier metric. Note: HTMNetwork model uses "
                            "the metric specified in the network config "
                            "file."))
  parser.add_argument("-d", "--dataPath",
                      help="Path to data CSV; samples must be in column w/ "
                           "header 'Sample'; see readCSV() for more.",
                      required=True)
  parser.add_argument("-m", "--modelName",
                      choices=list(ClassificationModelTypes.getTypes()),
                      default=ImbuModels.defaultModelType,
                      type=str,
                      help="Name of model class. Also used for model results "
                           "directory and pickle checkpoint.")
  parser.add_argument("--loadPath",
                      default="",
                      type=str,
                      help="Path from which to load a serialized model.")
  parser.add_argument("--savePath",
                      default="",
                      type=str,
                      help=("Directory path for saving the model. This "
                            "directory should only be used to store a saved "
                            "model. If the directory does not exist, it will "
                            "be created automatically and populated with model"
                            " data. A pre-existing directory will only be "
                            "accepted if it contains previously saved model "
                            "data. If such a directory is given, the full "
                            "contents of the directory will be deleted and "
                            "replaced with current"))
  parser.add_argument("--imbuRetinaId",
                      default=os.environ.get("IMBU_RETINA_ID"),
                      type=str)
  parser.add_argument("--corticalApiKey",
                      default=os.environ.get("CORTICAL_API_KEY"),
                      type=str)

  args = parser.parse_args()

  imbu = ImbuModels(
    cacheRoot=args.cacheRoot,
    modelSimilarityMetric=args.modelSimilarityMetric,
    dataPath=args.dataPath,
    retina=args.imbuRetinaId,
    apiKey=args.corticalApiKey
  )

  model = imbu.createModel(args.modelName,
                           loadPath=args.loadPath,
                           savePath=args.savePath)

  # Query the model.
  printTemplate = "{0:<10}|{1:<10}|{2:<10}"
  while True:
    print "Now we query the model for samples (quit with 'q')..."

    query = raw_input("Enter a query: ")

    if query == "q":
      break

    _, sortedIds, sortedDistances = imbu.query(model, query)

    results = imbu.formatResults(model, sortedDistances, sortedIds)

    # Display results.
    print printTemplate.format("Sample ID", "Word ID", "% Overlap With Query")
    for r in results:
      print printTemplate.format(r["sampleId"], r["wordId"], r["score"])



if __name__ == "__main__":
  main()
