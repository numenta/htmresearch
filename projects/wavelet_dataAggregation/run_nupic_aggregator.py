# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
Run Nupic aggregator on good day bad day data

(1) Perform aggregation using nupic.data.aggregator
(2) Plot aggregated and raw data using matplotlib/plotly

"""

from nupic.data import aggregator
from nupic.data import fieldmeta

from unicorn_backend.utils import date_time_utils

import matplotlib.pyplot as plt
import plotly.plotly as py
plt.ion()
plt.close('all')


def initializeAggregator(aggSpec, modelSpec):
  inputRecordSchema = (
    fieldmeta.FieldMetaInfo(modelSpec["timestampFieldName"],
                            fieldmeta.FieldMetaType.datetime,
                            fieldmeta.FieldMetaSpecial.timestamp),
    fieldmeta.FieldMetaInfo(modelSpec["valueFieldName"],
                            fieldmeta.FieldMetaType.float,
                            fieldmeta.FieldMetaSpecial.none),
  )

  dataAggregator = aggregator.Aggregator(
    aggregationInfo=dict(
      fields=([(modelSpec["valueFieldName"], aggSpec["func"])]
              if aggSpec is not None else []),
      seconds=aggSpec["windowSize"] if aggSpec is not None else 0
    ),
    inputFields=inputRecordSchema)
  return dataAggregator


if __name__ == "__main__":
  inputFile = open('example_data/JAO_Apple_Heart Rate_raw_20160404.csv')
  # skip header lines
  inputFile.readline()

  aggSpec = {"func": "mean",
             "windowSize": 3000}

  modelSpec = {"timestampFieldName": "timestamp",
               "valueFieldName": "value"}

  dataAggregator = initializeAggregator(aggSpec, modelSpec)

  timeStampRaw = []
  timeStampAgg = []
  valueRaw = []
  valueAgg = []
  sliceEndTime = []
  for inputRow in inputFile.readlines():
    inputRow = inputRow.split(',')
    fields = [
      date_time_utils.parseDatetime(inputRow[0],
                                    '%m/%d/%y %H:%M'),
      float(inputRow[1])
    ]
    aggRow, _ = dataAggregator.next(fields, None)

    timeStampRaw.append(fields[0])
    valueRaw.append(fields[1])
    if aggRow is not None:
      sliceEndTime.append(dataAggregator._endTime)
      timeStampAgg.append(aggRow[0])
      valueAgg.append(aggRow[1])

  fig = plt.figure()
  plt.plot(timeStampRaw, valueRaw, '.')
  plt.plot(timeStampAgg, valueAgg, 'r+')
  yl = plt.ylim()
  # for timestamp in sliceEndTime:
  #   plt.vlines(timestamp, yl[0], yl[1])
  plt.legend(['Raw', 'Aggregate'])
  plt.xlabel('Timestamp')
  plt.ylabel('Value')
  plt.xlim([timeStampRaw[100], timeStampRaw[300]])

  # plot_url = py.plot_mpl(fig, filename='GDBD_HeartRate_VisualizeAggregation',
  #                        fileopt='overwrite', sharing='private')
