#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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
Groups together code used for creating a NuPIC model and dealing with IO.
"""
import importlib
import csv
import os
    
    
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.model_factory import ModelFactory

import nupic_anomaly_output

from settings import METRICS, PATIENT_IDS, SENSORS, CONVERTED_DATA_DIR, MODEL_PARAMS_DIR, MODEL_RESULTS_DIR


def createModel(modelParams):
  """
  Given a model params dictionary, create a CLA Model. Automatically enables
  inference for metric_value.
  :param modelParams: Model params dict
  :return: OPF Model object
  """
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": "metric_value"})
  return model


def getModelParamsFromName(csvName):
  """
  Given a csv name, assumes a matching model params python module exists within
  the model_params directory and attempts to import it.
  :param csvName: CSV name, used to guess the model params module name.
  :return: OPF Model params dictionary
  """

  print "Creating model from %s..." % csvName
  importName = "%s.%s" % (MODEL_PARAMS_DIR, csvName.replace(" ", "_"))
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. "
                    "Run trajectory_converter.py first!"
                    % csvName)
  return importedModelParams


def runIoThroughNupic(inputData, model, metric, sensor, patientId, plot):
  """
  Handles looping over the input data and passing each row into the given model
  object, as well as extracting the result object and passing it into an output
  handler.
  :param inputData: file path to input data CSV
  :param model: OPF Model object
  :param csvName: CSV name, used for output handler naming
  :param plot: Whether to use matplotlib or not. If false, uses file output.
  """
  inputFile = open(inputData, "rb")
  csvReader = csv.reader(inputFile.read().splitlines())
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  csvName = "%s_%s_%s" % (metric, sensor, patientId)
  print "running model with model_params '%s'" % csvName

  shifter = InferenceShifter()
  if plot:
    output = nupic_anomaly_output.NuPICPlotOutput(csvName)
  else:
    if not os.path.exists(MODEL_RESULTS_DIR):
      os.makedirs(MODEL_RESULTS_DIR)
    output = nupic_anomaly_output.NuPICFileOutput("%s/%s" % (MODEL_RESULTS_DIR, 
                                                             csvName))

  counter = 0
  for row in csvReader:
    counter += 1
    if (counter % 100 == 0):
      print "Read %i lines..." % counter

    metric_value = float(row[0])
    result = model.run({
      "metric_value": metric_value
    })

    if plot:
      result = shifter.shift(result)

    prediction = result.inferences["multiStepBestPredictions"][0]
    anomalyScore = result.inferences["anomalyScore"]
    output.write(counter, metric_value, prediction, anomalyScore)

  output.close()
  inputFile.close()


def runModel(metric, sensor, patientId, plot=False):
  """
  Assumes the CSV Name corresponds to both a like-named model_params file in the
  model_params directory, and that the data exists in a like-named CSV file in
  the current directory.
  :param csvName: Important for finding model params and input CSV file
  :param plot: Plot in matplotlib? Don't use this unless matplotlib is
  installed.
  """

  csvName = "%s_%s_%s" % (metric, sensor, patientId)
  model = createModel(getModelParamsFromName(csvName))

  inputData = "%s/%s.csv" % (CONVERTED_DATA_DIR, csvName)
  runIoThroughNupic(inputData, model, metric, sensor, patientId, plot)


if __name__ == "__main__":
  for sensor in SENSORS:
    for patientId in PATIENT_IDS:
      for metric in METRICS:
        runModel(metric, sensor, patientId)
