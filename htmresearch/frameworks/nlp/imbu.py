import argparse
import os

import simplejson as json

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.support.csv_helper import readCSV
from htmresearch.frameworks.nlp.model_factory import (
  ClassificationModelTypes,
  createModel)



class ModelSimilarityMetrics(object):
  pctOverlapOfInput = "pctOverlapOfInput"



class ImbuError(Exception):
  pass



class ImbuUnableToLoadModelError(ImbuError):
  def __init__(self, exc):
    self.exc = exc



def _loadJSON(jsonPath):
  try:
    with open(jsonPath, "rb") as fin:
      return json.load(fin)
  except IOError as e:
    print "Could not find JSON at '{}'.".format(jsonPath)
    raise e



def _loadNetworkConfig():
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
  return _loadJSON(os.path.join(root,
                                "projects/nlp/data/network_configs/imbu.json"))



class ImbuModels(object):

  defaultSimilarityMetric = ModelSimilarityMetrics.pctOverlapOfInput
  defaultModelType = "HTMNetwork"

  def __init__(self, cacheRoot, dataPath, loadPath, savePath,
      modelSimilarityMetric):
    self.cacheRoot = cacheRoot
    self.modelSimilarityMetric = (
      modelSimilarityMetric or self.defaultSimilarityMetric
    )
    self.loadPath = loadPath
    self.savePath = savePath
    self.dataPath = dataPath
    self.dataDict = self._loadTrainingData()


  def __repr__(self):
    return ("Imbu<cacheRoot={cacheRoot}, dataPath={dataPath}, "
            "loadPath={loadPath}, savePath={savePath}, "
            "modelSimilarityMetric={modelSimilarityMetric}>"
            .format(**self.__dict__))


  def _defaultModelFactoryKwargs(self):
    return dict(
      numLabels=len(self.dataDict),
      modelDir=self.savePath,
      classifierMetric=self.modelSimilarityMetric)


  def modelFactory(self, modelType, **kwargs):

    modelType = modelType or self.defaultModelType

    kwargs.update(**self._defaultModelFactoryKwargs())

    if modelType == "CioWordFingerprint":
      kwargs.update(retina=kwargs.get("retina"),
                    apiKey=kwargs.get("apiKey"),
                    fingerprintType=EncoderTypes.word,
                    cacheRoot=self.cacheRoot)

    elif modelType == "CioDocumentFingerprint":
      kwargs.update(retina=kwargs.get("retina"),
                    apiKey=kwargs.get("apiKey"),
                    fingerprintType=EncoderTypes.document,
                    cacheRoot=self.cacheRoot)
    elif modelType == "HTMNetwork":
      kwargs.update(networkConfig=_loadNetworkConfig(),
                    inputFilePath=None,
                    prepData=False,
                    retinaScaling=1.0)

    model = createModel(modelType, **kwargs)


    model.verbosity = 0

    return model


  def loadModel(self, *modelFactoryArgs, **modelFactoryKwargs):

    try:
      model = ClassificationModel.load(self.loadPath)
    except IOError as exc:
      model = self.modelFactory(*modelFactoryArgs, **modelFactoryKwargs)
      self.train(model)

    return model


  def _loadTrainingData(self):
    return readCSV(self.dataPath, numLabels=0)


  def train(self, model):
    for seqId, (text, _, _) in enumerate(self.dataDict.values()):
      model.trainDocument(text, [seqId], seqId)

    self.save(model)


  @staticmethod
  def query(model, query, returnDetailedResults=True, sortResults=True):
    return model.inferDocument(query,
                               returnDetailedResults=returnDetailedResults,
                               sortResults=sortResults)


  def save(self, model):
    model.save(self.savePath)



def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--cacheRoot",
                      help="Root directory in which to cache encodings")
  parser.add_argument("--modelSimilarityMetric",
                      default=ImbuModels.defaultSimilarityMetric,
                      help="Classifier metric")
  parser.add_argument("--dataPath",
                      help="Path to data CSV; samples must be in column w/ "
                           "header 'Sample'; see readCSV() for more.",
                      required=True)
  parser.add_argument("--modelName",
                      choices=ClassificationModelTypes.getTypes(),
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
                      help="Path to save the serialized model.")
  args = parser.parse_args()

  imbu = ImbuModels(
    cacheRoot=args.cacheRoot,
    modelSimilarityMetric=args.modelSimilarityMetric,
    dataPath=args.dataPath,
    loadPath=args.loadPath,
    savePath=args.savePath
  )

  model = imbu.loadModel(args.modelName,
                         retina=os.environ.get("IMBU_RETINA_ID"),
                         apiKey=os.environ.get("CORTICAL_API_KEY"))

  # Query the model.
  printTemplate = "{0:<10}|{1:<30}"
  while True:
    print "Now we query the model for samples (quit with 'q')..."

    query = raw_input("Enter a query: ")

    if query == "q":
      break

    _, idList, sortedDistances = imbu.query(model, query)

    print printTemplate.format("Sample ID", "Distance from query")
    for sID, dist in zip(idList, sortedDistances):
      print printTemplate.format(sID, dist)



if __name__ == "__main__":
  main()
