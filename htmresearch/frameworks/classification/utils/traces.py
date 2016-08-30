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
import numpy as np



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
          if headers[i] in ['step',
                            'classificationAccuracyTrace',
                            'sensorValueTrace',
                            'categoryTrace',
                            'testClassificationAccuracyTrace']:
            data = float(row[i])
          elif headers[i] in ['tmPredictiveActiveCellsTrace',
                              'tpActiveCellsTrace',
                              'tmActiveCellsTrace']:
            if row[i] == '[]':
              data = []
            else:
              data = map(int, row[i][1:-1].split(','))
          else:
            raise ValueError('Unknown header name')
        traces[headers[i]].append(data)

  return traces