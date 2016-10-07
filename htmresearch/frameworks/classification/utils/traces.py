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
import json
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plotTraces(numTmCells, xlim, traces):
  """
  Plot network traces
  :param numTmCells: (int) number of cells in the TM
  :param xlim: (list) min and max values used for the x-axis range
  :param traces: (list of dict) network traces to plot
  :return: 
  """

  t = np.array(traces['recordNumber'])
  classLabels = np.array(traces['actualCategory'])
  sensorValue = np.array(traces['sensorValue'])
  if xlim is None:
    xlim = [t[0], t[-1]]
  else:
    xlim = np.array(xlim)
    if np.max(xlim) > len(t):
      print
      print "xlim needs to be smaller than {} for this trace file".format(
        len(t))
      print

  selectRange = np.where(np.logical_and(t > xlim[0], t < xlim[1]))[0]

  np.random.seed(21)
  randomCellOrder = np.random.permutation(np.arange(numTmCells))

  for traceName in ['tpActiveCells',
                    'tmPredictedActiveCells',
                    'tmActiveCells']:

    f, ax = plt.subplots(4, sharex=True)

    # plot sensor value and class labels
    ax[0].set_title('Sensor data')
    ax[0].plot(t, sensorValue)
    ax[0].set_xlim(xlim)

    yl = ax[0].get_ylim()
    height = yl[1] - yl[0]

    # plot class labels as transparent colored rectangles
    classColor = {0: 'grey', 1: 'b', 2: 'r', 3: 'y'}
    xStartLabel = xlim[0]
    while xStartLabel < xlim[1]:
      currentClassLabel = classLabels[xStartLabel]
      if len(np.where(classLabels[xStartLabel:] != currentClassLabel)[0]) == 0:
        width = len(classLabels[xStartLabel:])
      else:
        width = np.where(classLabels[xStartLabel:] != currentClassLabel)[0][0]

      ax[0].add_patch(
        patches.Rectangle((t[0] + xStartLabel, yl[0]), width, height,
                          facecolor=classColor[currentClassLabel], alpha=0.6)
      )
      xStartLabel += width
    ax[0].set_ylabel('Sensor Value')

    # plot classification accuracy
    ax[1].set_title('Classification accuracy rolling average')
    ax[1].plot(traces['classificationAccuracy'])

    # plot clustering accuracy
    ax[2].set_title('Clustering accuracy rolling average')
    if 'clusteringAccuracy' in traces:
      ax[2].plot(traces['clusteringAccuracy'])

    # plot cell activations
    ax[3].set_axis_bgcolor('black')

    cellTrace = traces[traceName]
    if len(selectRange) <= len(cellTrace):
      for i in range(len(selectRange)):
        cells = cellTrace[selectRange[i]]
        if cells is not None:
          sdrT = t[selectRange[i]] * np.ones((len(cells, )))
          ax[3].plot(sdrT, randomCellOrder[cells], 's', color='white', ms=1)

    ax[3].set_title('Cell activation')
    ax[3].set_ylabel(traceName)
    ax[3].set_ylim([0, numTmCells])
    ax[3].set_xlabel('Time')

  plt.show()



def saveTraces(traces, fileName):
  """
  Save netwrok traces to CSV
  :param traces: (dict) network traces. E.g: activeCells, sensorValues, etc.
  :param fileName: (str) name of the file
  """
  with open(fileName, 'wb') as fw:
    writer = csv.writer(fw)
    headers = ['step'] + traces.keys()
    writer.writerow(headers)
    for i in range(len(traces['sensorValueTrace'])):
      row = [i]
      for t in traces.keys():
        if len(traces[t]) > i:
          if type(traces[t][i]) == np.ndarray:
            traces[t][i] = list(traces[t][i])
          if type(traces[t][i]) != list:
            row.append(traces[t][i])
          else:
            row.append(json.dumps(traces[t][i]))
        else:
          row.append(None)
      writer.writerow(row)



def loadTraces(fileName):
  """
  Load netwrok traces from CSV
  :param fileName: (str) name of the file
  :return traces: (dict) network traces. E.g: activeCells, sensorValues, etc.
  """

  csv.field_size_limit(sys.maxsize)

  with open(fileName, 'rb') as fr:
    reader = csv.reader(fr)
    headers = reader.next()

    traces = dict()
    for field in headers:
      traces[field] = []

    for row in reader:
      for i in range(len(row)):
        if len(row[i]) == 0:
          data = []
        else:
          if headers[i] in ['tmPredictedActiveCells',
                            'tpActiveCells',
                            'tmActiveCells']:
            if row[i] == '[]':
              data = []
            else:
              data = map(int, row[i][1:-1].split(','))
          else:
            data = float(row[i])
        traces[headers[i]].append(data)

  return traces
