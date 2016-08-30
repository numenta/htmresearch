#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

from optparse import OptionParser

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from htmresearch.frameworks.classification.utils.traces import loadTraces


"""
Script to visualize the TM states
Example Usage:

python plot_experiment_result.py --xlim "1700,1750"

"""

def _getArgs():
  parser = OptionParser(usage=" plot trace files of the sequence classification"
                              " experiment ")
  parser.add_option("-f",
                    "--fileName",
                    type=str,
                    default='results/traces_sp-True_tm-True_tp-False.csv',
                    dest="fileName",
                    help="fileName of the csv trace file")

  parser.add_option("--xlim",
                    type=str,
                    default=None,
                    dest="xl",
                    help="x-axis range")

  (options, remainder) = parser.parse_args()
  return options, remainder



if __name__ == "__main__":
  (_options, _args) = _getArgs()
  fileName = _options.fileName

  xl = _options.xl

  traces = loadTraces(fileName)

  numTMcells = 2048 * 32

  t = np.array(traces['step'])
  classLabels = np.array(traces['categoryTrace'])
  sensorValue = np.array(traces['sensorValueTrace'])
  if xl is None:
    xl = [t[-200], t[-1]]
  else:
    xl = np.array([float(x) for x in xl.split(',')])
    if np.max(xl) > len(t):
      print
      print "xlim needs to be smaller than {} for this trace file".format(len(t))
      print
  selectRange = np.where(np.logical_and(t > xl[0],  t < xl[1]))[0]

  np.random.seed(21)
  randomCellOrder = np.random.permutation(np.arange(numTMcells))

  for traceName in ['tpActiveCellsTrace',
                    'tmPredictiveActiveCellsTrace',
                    'tmActiveCellsTrace']:

    f, ax = plt.subplots(2, sharex=True)

    # plot sensor value and class labels
    ax[0].plot(t, sensorValue)
    ax[0].set_xlim(xl)

    yl = ax[0].get_ylim()
    height = yl[1] - yl[0]

    # plot class labels as transparent colored rectangles
    classColor = {0: 'red', 1: 'blue'}
    xStartLabel = xl[0]
    while xStartLabel < xl[1]:
      currentClassLabel = classLabels[xStartLabel]
      if len(np.where(classLabels[xStartLabel:] != currentClassLabel)[0]) == 0:
        width = len(classLabels[xStartLabel:])
      else:
        width = np.where(classLabels[xStartLabel:] != currentClassLabel)[0][0]

      ax[0].add_patch(
          patches.Rectangle((t[0]+xStartLabel, yl[0]), width, height,
                            facecolor=classColor[currentClassLabel], alpha=0.6)
      )
      xStartLabel += width
    ax[0].set_ylabel('Sensor Value')

    # plot TM activations
    ax[1].set_axis_bgcolor('black')

    tmActiveCellsTrace = traces[traceName]
    for i in range(len(selectRange)):
      tmActiveCells = tmActiveCellsTrace[selectRange[i]]
      sdrT = t[selectRange[i]] * np.ones((len(tmActiveCells, )))
      ax[1].plot(sdrT, randomCellOrder[tmActiveCells], 's', color='white', ms=1)

    ax[1].set_ylabel(traceName)
    ax[1].set_ylim([0, numTMcells])
    ax[1].set_xlabel('Time')

  plt.show()