import argparse
import os

from enum import Enum

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint)
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords)
from htmresearch.frameworks.nlp.classify_windows import (
  ClassificationModelWindows)
from htmresearch.support.csv_helper import readCSV
from htmresearch.support.text_preprocess import TextPreprocess



class ModelTypes(Enum):
  CioWordFingerprint = ClassificationModelFingerprint
  CioDocumentFingerprint = ClassificationModelFingerprint
  CioWindows = ClassificationModelWindows
  Keywords = ClassificationModelKeywords
  HTMNetwork = ClassificationModelHTM



class ModelSimilarityMetrics(Enum):
  pctOverlapOfInput = "pctOverlapOfInput"



class ImbuError(Exception):
  pass



class ImbuUnableToLoadModelError(ImbuError):
  def __init__(self, exc):
    self.exc = exc



class Imbu(object):

  defaultSimilarityMetric = ModelSimilarityMetrics.pctOverlapOfInput
  defaultModelType = ModelTypes.HTMNetwork

  def __init__(self, cacheRoot, dataPath, loadPath, savePath,
      modelSimilarityMetric):
    self.cacheRoot = cacheRoot
    self.modelSimilarityMetric = (
      modelSimilarityMetric or self.defaultSimilarityMetric
    )
    self.loadPath = loadPath
    self.savePath = savePath
    self.dataPath = dataPath


  def __repr__(self):
    return ("Imbu<cacheRoot={cacheRoot}, dataPath={dataPath}, "
            "loadPath={loadPath}, savePath={savePath}, "
            "modelSimilarityMetric={modelSimilarityMetric}>"
            .format(**self.__dict__))



  def modelFactory(self, modelType, **kwargs):

    modelType = modelType or self.defaultModelType

    if modelType is ModelTypes.CioWordFingerprint:
      model = modelType.value(retina=kwargs.get("retina"),
                              apiKey=kwargs.get("apiKey"),
                              fingerprintType=EncoderTypes.word,
                              modelDir=self.savePath,
                              cacheRoot=self.cacheRoot,
                              classifierMetric=self.modelSimilarityMetric.value)

    elif modelType is ModelTypes.CioDocumentFingerprint:
      model = modelType.value(retina=kwargs.get("retina"),
                              apiKey=kwargs.get("apiKey"),
                              fingerprintType=EncoderTypes.document,
                              modelDir=self.savePath,
                              cacheRoot=self.cacheRoot,
                              classifierMetric=self.modelSimilarityMetric.value)

    elif modelType is ModelTypes.HTMNetwork:
      raise NotImplementedError("HTMNetwork model type is not implemented.")

    else:
      if modelType not in ModelTypes:
        raise NotImplementedError()

      model = modelType.value(modelDir=self.savePath,
                              classifierMetric=self.modelSimilarityMetric.value)

    model.verbosity = 0
    model.numLabels = 0

    return model


  def loadModel(self):
    try:
      return ClassificationModel.load(self.loadPath)
    except IOError as exc:
      raise ImbuUnableToLoadModelError(exc)



  def _loadTrainingData(self):
    return readCSV(self.dataPath, numLabels=0)


  def train(self, model):
    dataDict = self._loadTrainingData()
    raise NotImplementedError() # Not sure what to do here any more



def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("--cacheRoot",
                      help="Root directory in which to cache encodings")
  parser.add_argument("--modelSimilarityMetric",
                      choices={metric.name
                               for metric in ModelSimilarityMetrics},
                      default=Imbu.defaultSimilarityMetric.name,
                      help="Classifier metric")
  parser.add_argument("--dataPath",
                      help="Path to data CSV; samples must be in column w/ "
                           "header 'Sample'; see readCSV() for more.",
                      required=True)
  parser.add_argument("--modelName",
                      choices={modelType.name for modelType in ModelTypes},
                      default=Imbu.defaultModelType.name,
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

  imbu = Imbu(
    cacheRoot=args.cacheRoot,
    modelSimilarityMetric=getattr(ModelSimilarityMetrics,
                                  args.modelSimilarityMetric),
    dataPath=args.dataPath,
    loadPath=args.loadPath,
    savePath=args.savePath
  )

  if args.loadPath:
    model = imbu.loadModel()
  else:
    model = imbu.modelFactory(getattr(ModelTypes, args.modelName),
                              retina=os.environ["IMBU_RETINA_ID"],
                              apiKey=os.environ["CORTICAL_API_KEY"])

  imbu.train(model)

  if args.savePath:
    model.save()

  # Query the model.
  printTemplate = "{0:<10}|{1:<30}"
  while True:
    print "Now we query the model for samples (quit with 'q')..."

    query = raw_input("Enter a query: ")

    if query == "q":
      break

    sortedDistances = model.queryModel(query)

    print printTemplate.format("Sample ID", "Distance from query")

    for sID, dist in sortedDistances:
      print printTemplate.format(sID, dist)



if __name__ == "__main__":
  main()
