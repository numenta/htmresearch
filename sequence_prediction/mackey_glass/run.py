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
import importlib
import sys
import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager

import nupic_output

from model_params.model_params import MODEL_PARAMS


DATA_TRAIN = "data_train.csv"
DATA_TEST = "data_test.csv"
DATA_DIR = "."
# '7/2/10 0:00'
DATE_FORMAT = "%m/%d/%y %H:%M"

_METRIC_SPECS = (
    MetricSpec(field='y', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(field='y', metric='trivial',
               inferenceElement='prediction',
               params={'errorMetric': 'aae', 'window': 1000, 'steps': 1}),
    MetricSpec(field='y', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'nrmse', 'window': 1000000, 'steps': 1}),
    MetricSpec(field='y', metric='trivial',
               inferenceElement='prediction',
               params={'errorMetric': 'nrmse', 'window': 1000000, 'steps': 1}),
)

def createModel(modelParams):
  model = ModelFactory.create(modelParams)
  model.enableInference({"predictedField": "y"})
  return model



def runIoThroughNupic(inputData, model, dataName, plot):
  inputFile = open(inputData, "rU")
  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  shifter = InferenceShifter()
  if plot:
    output = nupic_output.NuPICPlotOutput([dataName])
  else:
    output = nupic_output.NuPICFileOutput([dataName])

  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  counter = 0
  for row in csvReader:
    counter += 1
    y = float(row[0])
    result = model.run({
      "y": y
    })
    result.metrics = metricsManager.update(result)

    if counter % 100 == 0:
      print "Read %i lines..." % counter
      print ("After %i records, 1-step nrmse=%f", counter,
              result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=1:window=1000000:"
                             "field=y"])
 
    if plot:
      result = shifter.shift(result)

    prediction = result.inferences["multiStepBestPredictions"][1]
    output.write([0], [y], [prediction])

  inputFile.close()
  output.close()



def runModel(plot=False):
  model = createModel(MODEL_PARAMS)
  print "Training..."
  runIoThroughNupic(DATA_TRAIN, model, DATA_TRAIN, plot)
  model.resetSequenceStates()
  model.disableLearning()
  print "Testing..."
  runIoThroughNupic(DATA_TEST, model, DATA_TEST, plot)



if __name__ == "__main__":
  plot = False
  args = sys.argv[1:]
  if "--plot" in args:
    plot = True
  runModel(plot=plot)