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

import csv

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


from htmresearch.frameworks.classification.utils.traces import loadTraces

plt.ion()

if __name__ == "__main__":
  fileName = 'results/traces_sp-True_tm-True_tp-True.csv'
  traces = loadTraces(fileName)

  numTMcells = 2048 * 32

  t = np.array(traces['step'])
  classLabels = np.array(traces['categoryTrace'])
  sensorValue = np.array(traces['sensorValueTrace'])

  xl = [500, 550]
  selectRange = np.where(np.logical_and(t > xl[0],  t < xl[1]))[0]

  plt.figure()
  f, ax = plt.subplots(4, sharex=True)

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
  ax[2].set_axis_bgcolor('black')
  ax[3].set_axis_bgcolor('black')

  tmPredictiveActiveCellsTrace = traces['tmPredictiveActiveCellsTrace']
  tmActiveCellsTrace = traces['tmActiveCellsTrace']
  tpActiveCellsTrace = traces['tpActiveCellsTrace']

  tmPredictiveActiveCells = np.zeros((len(selectRange), numTMcells))
  for i in range(len(selectRange)):
    tmPredictiveActiveCells = tmPredictiveActiveCellsTrace[selectRange[i]]
    sdrT = t[selectRange[i]] * np.ones((len(tmPredictiveActiveCells, )))
    ax[1].plot(sdrT, tmPredictiveActiveCells, 's', color='white', ms=3)

    tmActiveCells = tmActiveCellsTrace[selectRange[i]]
    sdrT = t[selectRange[i]] * np.ones((len(tmActiveCells, )))
    ax[2].plot(sdrT, tmActiveCells, 's', color='white', ms=3)

    tpActiveCells = tpActiveCellsTrace[selectRange[i]]
    sdrT = t[selectRange[i]] * np.ones((len(tpActiveCells, )))
    ax[3].plot(sdrT, tpActiveCells, 's', color='white', ms=3)

  ax[1].set_ylabel('Predicted Active TM Cells')
  ax[1].set_ylim([0, numTMcells])

  ax[2].set_ylabel('Active TM Cells')
  ax[2].set_ylim([0, numTMcells])

  ax[3].set_ylabel('TP Cells')
  ax[3].set_ylim([0, numTMcells])

  ax[3].set_xlabel('Time')
