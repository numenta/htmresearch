#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
A simple example showing how to use classification for detecting known
anomalies even if they occur again. There are two different methods shown here.
"""

import csv
import importlib

from nupic.data.datasethelpers import findDataset
from nupic.frameworks.opf.modelfactory import ModelFactory

from generate_data import generateData
from generate_model_params import createModelParams
from settings import \
  CLASS_RANGES, \
  RESULTS_DIR, \
  SIGNAL_TYPES, \
  TRAINING_SET_SIZE, \
  DATA_DIR, \
  MODEL_PARAMS_DIR, \
  WHITE_NOISE_AMPLITUDE, \
  SIGNAL_AMPLITUDE, \
  SIGNAL_MEAN


def createModel(metricName):
  modelParams = getModelParamsFromName(metricName)
  return ModelFactory.create(modelParams)


def trainAndClassify(trainingSetSize, model, data_path, results_path):
  """
  In this function we explicitly label specific portions of the data stream.
  Any later record that matches the pattern will get labeled the same way.
  """

  model.enableInference({'predictedField': 'y'})

  # Here we will get the classifier instance so we can add and query labels.
  classifierRegion = model._getAnomalyClassifier()
  classifierRegionPy = classifierRegion.getSelf()

  # We need to set this classification type. It is supposed to be the default
  # but is not for some reason.
  classifierRegionPy.classificationVectorType = 2

  with open (findDataset(data_path)) as fin:
    reader = csv.reader(fin)
    headers = reader.next()
    csvWriter = csv.writer(open(results_path,"wb"))
    csvWriter.writerow(["x", "y", "trueLabel", "anomalyScore", "predictedLabel"])
    for x, record in enumerate(reader):
      modelInput = dict(zip(headers, record))
      modelInput["y"] = float(modelInput["y"])
      trueLabel = modelInput["label"]
      result = model.run(modelInput)
      anomalyScore = result.inferences['anomalyScore']
      predictedLabel = result.inferences['anomalyLabel'][2:-2]

      # relabel prediction for all the records with indices withing the training set size range.
      if x < trainingSetSize:
        for label in CLASS_RANGES:
          for class_range in CLASS_RANGES[label]:
            start = class_range['start']
            end = class_range['end']

            if start <= x <= end:
              predictedLabel = label

            if x == end + 2:
              print "Adding labeled anomalies for record", x
              classifierRegion.executeCommand(["addLabel", str(start), str(end + 1), label])

      csvWriter.writerow([x, modelInput["y"], trueLabel, anomalyScore, predictedLabel])

  print "Results scores have been written to %s" % results_path


def computeClassificationAccuracy(result_file):
  num_errors = 0.0
  num_records = 0.0
  with open (findDataset(result_file)) as fin:
    reader = csv.reader(fin)
    headers = reader.next()
    for i, record in enumerate(reader):
      data = dict(zip(headers, record))

      if data['predictedLabel'] != data['trueLabel']:
          num_errors += 1.0
          print "=> Incorrectly predicted record at line #%s. True Label: %s. Predicted Label: %s" %\
                (i, data['trueLabel'], data['predictedLabel']) 
      
      num_records += 1.0
          
  print ""
  print "== Classification error for %s ==" % resultsPath
  print "* Number of errors: %s" % num_errors
  print "* Classification accuracy: %s %%" % (num_errors/num_records * 100)
  print ""


def getModelParamsFromName(metricName):
  
  importName = "model_params.%s_model_params" % metricName
    
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'" % importName)
  return importedModelParams

if __name__ == "__main__":

  # generate data
  generateData(whiteNoise=False, signal_mean=SIGNAL_MEAN, signal_amplitude=SIGNAL_AMPLITUDE)
  generateData(whiteNoise=True, signal_mean=SIGNAL_MEAN, signal_amplitude=SIGNAL_AMPLITUDE, noise_amplitude=WHITE_NOISE_AMPLITUDE)


  for signalType in SIGNAL_TYPES:
    # generate model params
    fileName = '%s/%s.csv' % (DATA_DIR, signalType)
    modelParamsName = '%s_model_params' % signalType
    createModelParams(MODEL_PARAMS_DIR, modelParamsName, fileName)

    dataPath = "%s.csv" % signalType
    resultsPath = "%s/%s.csv" % (RESULTS_DIR, signalType)

    model = createModel(signalType)
    trainAndClassify(TRAINING_SET_SIZE, model, dataPath, resultsPath)
    computeClassificationAccuracy(resultsPath)
