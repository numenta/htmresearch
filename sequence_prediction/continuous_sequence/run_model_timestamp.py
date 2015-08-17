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
(This is a component of the One Hot Gym Prediction Tutorial.)
"""
import importlib
import sys
import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager

import nupic_output
from optparse import OptionParser

DESCRIPTION = (
  "Starts a NuPIC model from the model params returned by the swarm\n"
  "and pushes each line of input from the gym into the model. Results\n"
  "are written to an output file (default) or plotted dynamically if\n"
  "the --plot option is specified.\n"
  "NOTE: You must run ./swarm.py before this, because model parameters\n"
  "are required to run NuPIC.\n"
)


DATA_DIR = "./data"
MODEL_PARAMS_DIR = "./model_params"



def getMetricSpecs(predictedField):
  _METRIC_SPECS = (
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'nrmse', 'window': 1000, 'steps': 1}),
  )
  return _METRIC_SPECS


def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": predictedField})
  return model


def getModelParamsFromName(dataSet):
  importName = "model_params.%s_model_params" % (
    dataSet.replace(" ", "_").replace("-", "_")
  )
  print "Importing model params from %s" % importName
  try:
    importedModelParams = importlib.import_module(importName).MODEL_PARAMS
  except ImportError:
    raise Exception("No model params exist for '%s'. Run swarm first!"
                    % dataSet)
  return importedModelParams



def runIoThroughNupic(inputData, model, dataSet, plot, savePrediction=True):
  print dataSet
  inputFile = open(inputData, "rb")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  shifter = InferenceShifter()
  if plot:
    output = nupic_output.NuPICPlotOutput([dataSet])
  else:
    output = nupic_output.NuPICFileOutput([dataSet])

  _METRIC_SPECS = getMetricSpecs(predictedField)
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  counter = 0
  for row in csvReader:
    counter += 1
    timestamp = datetime.datetime.strptime(row[0], DATE_FORMAT)
    data = float(row[1])
    if dataSet == 'rec-center-hourly':
      result = model.run({
        "timestamp": timestamp,
        "kw_energy_consumption": data
      })
    elif dataSet == 'nyc_taxi':
      result = model.run({
        "timestamp": timestamp,
        "passenger_count": data
      })

    result.metrics = metricsManager.update(result)

    if counter % 100 == 0:
      print "Read %i lines..." % counter
      print ("After %i records, 1-step NRMSE=%f", counter,
              result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=1:window=1000:"
                             "field="+predictedField])
 
    if plot:
      result = shifter.shift(result)

    prediction_1step = result.inferences["multiStepBestPredictions"][1]
    prediction_5step = result.inferences["multiStepBestPredictions"][5]
    output.write([timestamp], [data], [prediction_1step], [prediction_5step])

  inputFile.close()
  output.close()


def runModel(dataSet, plot=False):
  print "Creating model from %s..." % dataSet
  model = createModel(getModelParamsFromName(dataSet))
  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))
  runIoThroughNupic(inputData, model, dataSet, plot)


def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default=0,
                    dest="dataSet",
                    help="DataSet Name, choose from rec-center-hourly, nyc_taxi")

  parser.add_option("-p",
                    "--plot",
                    default=False,
                    dest="plot",
                    help="Set to True to plot result")


  (options, remainder) = parser.parse_args()
  print options

  return options, remainder

if __name__ == "__main__":
  print DESCRIPTION
  plot = False
  args = sys.argv[1:]
  if "--plot" in args:
    plot = True

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot

  if dataSet == "rec-center-hourly":
    DATE_FORMAT = "%m/%d/%y %H:%M" # '7/2/10 0:00'
    predictedField = "kw_energy_consumption"
  elif dataSet == "nyc_taxi":
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    predictedField = "passenger_count"

  runModel(dataSet, plot=plot)
