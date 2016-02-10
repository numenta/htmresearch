import json
import os
from matplotlib import pyplot
import matplotlib as mpl


import numpy
from plot import plotAccuracy
from plot import computeAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
pyplot.ion()




if __name__ == '__main__':

  outdir = 'tm/result/'
  experiment = os.path.join(outdir, "num_predictions{}".format(
    1), "noise_at{}".format(6000)) + '/0.log'

  experiment = os.path.join("lstm/results",
                            "high-order-noise",
                            "0.log")
  (predictions, truths, iterations,
   resets, randoms, trains, killCell) = readExperiment(experiment)

  (accuracy, x) = computeAccuracy(predictions,
                                  truths,
                                  iterations,
                                  resets=resets,
                                  randoms=randoms)

  plotAccuracy((accuracy, x),
               trains,
               window=100,
               type=type,
               label='NoiseExperiment',
               hideTraining=True,
               lineSize=1.0)
