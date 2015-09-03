## ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

import importlib
import sys
import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager
from nupic.frameworks.opf import metrics
import nupic_output
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nrmse import *
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plt.ion()
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
                 params={'errorMetric': 'negativeLogLikelihood', 'window': 1000, 'steps': 5}),
      MetricSpec(field=predictedField, metric='multiStep',
                 inferenceElement='multiStepBestPredictions',
                 params={'errorMetric': 'nrmse', 'window': 1000, 'steps': 5}),
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

  i = 0
  for row in csvReader:
    i += 1
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
        "passenger_count": float(row[1])
      })

    result.metrics = metricsManager.update(result)

    if i % 100 == 0:
      print "Read %i lines..." % i
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='negativeLogLikelihood':steps=5:window=1000:"
                             "field="+predictedField]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=5:window=1000:"
                             "field="+predictedField]
      print "After %i records, 5-step negLL=%f nrmse=%f" % (i, negLL, nrmse)
 
    if plot:
      result = shifter.shift(result)

    # prediction_1step = result.inferences["multiStepBestPredictions"][1]
    prediction_5step = result.inferences["multiStepBestPredictions"][5]
    output.write([timestamp], [data], [prediction_5step])
    # output.write([timestamp], [data], [prediction_1step], [prediction_5step])

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
                    default='nyc_taxi',
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


def getInputRecord(df, predictedField, i):
  inputRecord = {
    predictedField: float(df[predictedField][i]),
    "timeofday": float(df["timeofday"][i]),
    # "dayofweek": row[3],
  }
  return inputRecord

def runMultiplePass(df, model, predictedField, nMultiplePass, nTrain):
  """
  run CLA model through data record 0:nTrain nMultiplePass passes
  """
  # nMultiplePass = 5
  # nTrain = 5000
  print "run TM through the train data multiple times"
  for nPass in xrange(nMultiplePass):
    for j in xrange(nTrain):
      inputRecord = getInputRecord(df, predictedField, j)
      result = model.run(inputRecord)
      if j % 100 == 0:
        print " pass %i, record %i" % (nPass, j)
    # reset temporal memory
    model._getTPRegion().getSelf()._tfdr.reset()

  return model


if __name__ == "__main__":
  print DESCRIPTION

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  plot = _options.plot

  if dataSet == "rec-center-hourly":
    DATE_FORMAT = "%m/%d/%y %H:%M" # '7/2/10 0:00'
    predictedField = "kw_energy_consumption"
  elif dataSet == "nyc_taxi":
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    predictedField = "passenger_count"
  else:
    raise RuntimeError("un recognized dataset")
  # runModel(dataSet, plot=plot)

  modelParams = getModelParamsFromName(dataSet)

  print "Creating model from %s..." % dataSet
  # model = createModel(modelParams)

  from clamodel_custom import CLAModel_custom
  # from nupic.frameworks.opf.clamodel import CLAModel
  model = CLAModel_custom(**modelParams['modelParams'])
  model.enableInference({"predictedField": predictedField})
  model.enableLearning()
  model._spLearningEnabled = True
  model._tpLearningEnabled = True

  inputData = "%s/%s.csv" % (DATA_DIR, dataSet.replace(" ", "_"))

  sensor = model._getSensorRegion()
  encoderList = sensor.getSelf().encoder.getEncoderList()
  if sensor.getSelf().disabledEncoder is not None:
    classifier_encoder = sensor.getSelf().disabledEncoder.getEncoderList()
    classifier_encoder = classifier_encoder[0]
  else:
    classifier_encoder = None

  # encoder = model._classifierInputEncoder
  maxBucket = classifier_encoder.n - classifier_encoder.w + 1


  shifter = InferenceShifter()
  output = nupic_output.NuPICFileOutput([dataSet])

  _METRIC_SPECS = getMetricSpecs(predictedField)
  metric = metrics.getModule(_METRIC_SPECS[0])
  metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                  model.getInferenceType())

  if plot:
    import matplotlib.gridspec as gridspec
    plotCount = 1
    plotHeight = max(plotCount * 3, 6)
    fig = plt.figure(figsize=(14, plotHeight))
    gs = gridspec.GridSpec(plotCount, 1)
    plt.title(predictedField)
    plt.ylabel('Data')
    plt.xlabel('Timed')
    plt.tight_layout()
    plt.ion()

  print "Load dataset: ", dataSet
  df = pd.read_csv(inputData, header=0, skiprows=[1,2])

  nMultiplePass = 3
  nTrain = 5000
  print " run TM through the first %i samples %i passes " %(nMultiplePass, nTrain)
  model = runMultiplePass(df, model, predictedField, nMultiplePass, nTrain)

  likelihoodsVecAll = np.zeros((maxBucket, len(df)))

  prediction_5step = None
  time_step = []
  actual_data = []
  patternNZ_track = []
  nPredictionSteps = 5
  predict_data = np.zeros((nPredictionSteps, 0))
  predict_data_ML = []
  negLL_track = []

  activeCellNum = []
  predCellNum = []
  predictedActiveColumnsNum = []

  sp = model._getSPRegion().getSelf()._sfdr
  spActiveCellsCount = np.zeros(sp.getColumnDimensions())

  for i in xrange(len(df)):
    inputRecord = getInputRecord(df, predictedField, i)

    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    prePredictiveCells = tm.predictiveCells
    prePredictiveColumn = np.array(list(prePredictiveCells))/ tm.cellsPerColumn

    result = model.run(inputRecord)
    # spRegion = model._getSPRegion().getSelf()
    # sensorOutput = spRegion._spatialPoolerInput

    sp = model._getSPRegion().getSelf()._sfdr
    spOutput = model._getSPRegion().getOutputData('bottomUpOut')
    spActiveCellsCount[spOutput.nonzero()[0]] += 1

    activeDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
    sp.getActiveDutyCycles(activeDutyCycle)
    overlapDutyCycle = np.zeros(sp.getColumnDimensions(), dtype=np.float32)
    sp.getOverlapDutyCycles(overlapDutyCycle)

    if i % 100 == 0 and i > 0:
      plt.figure(1)
      plt.clf()
      plt.subplot(2, 2, 1)
      plt.hist(overlapDutyCycle)
      plt.xlabel('overlapDutyCycle')

      plt.subplot(2, 2, 2)
      plt.hist(activeDutyCycle)
      plt.xlabel('activeDutyCycle-1000')

      plt.subplot(2, 2, 3)
      plt.hist(spActiveCellsCount)
      plt.xlabel('activeDutyCycle-Total')
      plt.draw()

    tp = model._getTPRegion()
    tm = tp.getSelf()._tfdr
    tpOutput = tm.infActiveState['t']

    predictiveCells = tm.predictiveCells
    predCellNum.append(len(predictiveCells))
    predColumn = np.array(list(predictiveCells))/ tm.cellsPerColumn

    patternNZ = tpOutput.reshape(-1).nonzero()[0]
    activeColumn = patternNZ / tm.cellsPerColumn
    activeCellNum.append(len(patternNZ))

    predictedActiveColumns = np.intersect1d(prePredictiveColumn, activeColumn)
    predictedActiveColumnsNum.append(len(predictedActiveColumns))

    result.metrics = metricsManager.update(result)

    negLL = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='negativeLogLikelihood':steps=5:window=1000:"
                             "field="+predictedField]
    if i % 100 == 0 and i>0:
      negLL = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='negativeLogLikelihood':steps=5:window=1000:"
                             "field="+predictedField]
      nrmse = result.metrics["multiStepBestPredictions:multiStep:"
                             "errorMetric='nrmse':steps=5:window=1000:"
                             "field="+predictedField]

      numActiveCell = np.mean(activeCellNum[-100:])
      numPredictiveCells = np.mean(predCellNum[-100:])
      numCorrectPredicted = np.mean(predictedActiveColumnsNum[-100:])

      print "After %i records, 5-step negLL=%f nrmse=%f ActiveCell %f PredCol %f CorrectPredCol %f" % \
            (i, negLL, nrmse, numActiveCell, numPredictiveCells, numCorrectPredicted)    

    # prediction_1step = result.inferences["multiStepBestPredictions"][1]
    last_prediction = prediction_5step
    prediction_5step = result.inferences["multiStepBestPredictions"][5]
    output.write([i], [inputRecord[predictedField]], [float(prediction_5step)])


    bucketLL = result.inferences['multiStepBucketLikelihoods'][5]
    likelihoodsVec = np.zeros((maxBucket,))
    if bucketLL is not None:
      for (k, v) in bucketLL.items():
        likelihoodsVec[k] = v

    time_step.append(i)
    actual_data.append(inputRecord[predictedField])
    predict_data_ML.append(result.inferences['multiStepBestPredictions'][5])
    negLL_track.append(negLL)

    likelihoodsVecAll[0:len(likelihoodsVec), i] = likelihoodsVec


    startPlotFrom = 500
    if plot and i > startPlotFrom:
      # prepare data for display
      if i > 100:
        time_step_display = time_step[-100:]
        actual_data_display = actual_data[-100:]
        predict_data_ML_display = predict_data_ML[-100:]
        likelihood_display = likelihoodsVecAll[:, i-100:i]
        xl = [(i)-100, (i)]
      else:
        time_step_display = time_step
        actual_data_display = actual_data
        predict_data_ML_display = predict_data_ML
        likelihood_display = likelihoodsVecAll[:, :i]
        xl = [0, (i)]

      plt.figure(2)
      plt.clf()
      plt.imshow(likelihood_display, extent=(time_step_display[0], time_step_display[-1], 0, 40000),
                 interpolation='nearest', aspect='auto', origin='lower', cmap='Reds')
      plt.plot(time_step_display, actual_data_display, 'k', label='Data')
      plt.plot(time_step_display, predict_data_ML_display, 'b', label='Best Prediction')
      plt.xlim(xl)
      plt.draw()

  predData_TM_five_step = np.roll(np.array(predict_data_ML), 5)
  nTest = len(actual_data) - nTrain - 5
  NRMSE_TM = NRMSE(actual_data[nTrain:nTrain+nTest], predData_TM_five_step[nTrain:nTrain+nTest])
  print "NRMSE on test data: ", NRMSE_TM
  output.close()
