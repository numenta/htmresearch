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

import csv
import numpy as np
from scipy import signal

"""
Automatically find data aggregation window and suggest whether to use TimeOfDay and DayOfWeek encoder

Example usage:

(timestamp, value) = readCSVfiles('example_data/art_daily_flatmiddle.csv')
(new_sampling_interval, useTimeOfDay, useDayOfWeek) = get_suggested_timescale_and_encoder(timestamp, value)

"""

def readCSVfiles(fileName):
  """
  Read csv data file, the data file must have two columns
  with header "timestamp", and "value"
  """
  fileReader = csv.reader(open(fileName, 'r'))
  fileReader.next() # skip header line
  timestamps = []
  values = []

  for row in fileReader:
    timestamps.append(row[0])
    values.append(row[1])

  timestamps = np.array(timestamps, dtype='datetime64')
  values = np.array(values, dtype='float32')

  return (timestamps, values)



def resample_data(timestamp, sig, new_sampling_interval):
  """
  Resample time series data at new sampling interval using linear interpolation
  Note: the resampling function is using interpolation, it may not be appropriate for aggregation purpose
  :param timestamp: timestamp in numpy datetime64 type
  :param sig: value of the time series
  :param new_sampling_interval: new sampling interrval
  """
  nSampleNew = np.floor((timestamp[-1] - timestamp[0])/new_sampling_interval).astype('int') + 1

  timestamp_new = np.empty(nSampleNew, dtype='datetime64[s]')
  for sampleI in xrange(nSampleNew):
    timestamp_new[sampleI] = timestamp[0] + sampleI * new_sampling_interval

  sig_new = np.interp((timestamp_new-timestamp[0]).astype('float32'),
                      (timestamp-timestamp[0]).astype('float32'), sig)

  return timestamp_new, sig_new


def calculate_cwt(sampling_interval, value):
  """
  Calculate continuous wavelet transformation (CWT)
  Return variance of the cwt coefficients overtime and its cumulative
  distribution

  :param sampling_interval: sampling interval of the time series
  :param value: value of the time series
  """

  t = np.array(range(len(value)))*sampling_interval
  widths = np.logspace(0, np.log10(len(value)/20), 50)

  T = int(widths[-1])

  # continulus wavelet transformation with ricker wavelet
  cwtmatr = signal.cwt(value, signal.ricker, widths)
  cwtmatr = cwtmatr[:, 4*T:-4*T]
  value = value[4*T:-4*T]
  t = t[4*T:-4*T]

  freq = 1/widths.astype('float') / sampling_interval / 4
  time_scale = widths * sampling_interval * 4

  # variance of wavelet power
  cwt_var = np.var(np.abs(cwtmatr), axis=1)
  cwt_var = cwt_var/np.sum(cwt_var)

  return cwtmatr, cwt_var, time_scale


def get_local_maxima(cwt_var, time_scale):
  """
  Find local maxima from the wavelet coefficient variance spectrum
  A strong maxima is defined as
  (1) At least 10% higher than the nearest local minima
  (2) Above the baseline value

  The algorithm will suggest an encoder if its corresponding
  perodicity is close to a strong maxima:
  (1) horizontally must within the nearest local minimum
  (2) vertically must within 50% of the peak of the strong maxima
  """

  # peak & valley detection
  local_min = (np.diff(np.sign(np.diff(cwt_var))) > 0).nonzero()[0] + 1
  local_max = (np.diff(np.sign(np.diff(cwt_var))) < 0).nonzero()[0] + 1

  baseline_value = 1.0/len(cwt_var)

  dayPeriod = 86400.0
  weekPeriod = 604800.0

  cwt_var_at_dayPeriod = np.interp(dayPeriod, time_scale, cwt_var)
  cwt_var_at_weekPeriod = np.interp(weekPeriod, time_scale, cwt_var)

  useTimeOfDay = False
  useDayOfWeek = False

  strong_local_max = []
  for i in xrange(len(local_max)):
    left_local_min = np.where(np.less(local_min, local_max[i]))[0]
    if len(left_local_min) == 0:
      left_local_min = 0
      left_local_min_value = cwt_var[0]
    else:
      left_local_min = local_min[left_local_min[-1]]
      left_local_min_value = cwt_var[left_local_min]

    right_local_min = np.where(np.greater(local_min, local_max[i]))[0]
    if len(right_local_min) == 0:
      right_local_min = len(cwt_var)-1
      right_local_min_value = cwt_var[-1]
    else:
      right_local_min = local_min[right_local_min[0]]
      right_local_min_value = cwt_var[right_local_min]

    local_max_value = cwt_var[local_max[i]]
    nearest_local_min_value = np.max(left_local_min_value, right_local_min_value)
    if ((local_max_value - nearest_local_min_value)/nearest_local_min_value > 0.1 and
           local_max_value > baseline_value ):
      strong_local_max.append(local_max[i])

      if (time_scale[left_local_min] < dayPeriod < time_scale[right_local_min] and
              cwt_var_at_dayPeriod > local_max_value*0.5):
        useTimeOfDay = True

      if (time_scale[left_local_min] < weekPeriod < time_scale[right_local_min] and
              cwt_var_at_weekPeriod > local_max_value*0.5):
        useDayOfWeek = True

  return useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max


def determine_aggregation_window(time_scale, cum_cwt_var, thresh, dt_sec, data_length):
  cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]
  aggregation_time_scale = cutoff_time_scale/10.0
  if aggregation_time_scale < dt_sec*4:
    aggregation_time_scale = dt_sec*4

  if data_length < 1000:
    aggregation_time_scale = dt_sec
  else:
    # make sure there is > 1000 records after aggregation
    dt_max = float(data_length)/1000.0 * dt_sec
    if aggregation_time_scale > dt_max > dt_sec:
        aggregation_time_scale = dt_max

  return aggregation_time_scale


def get_suggested_timescale_and_encoder(timestamp, value, thresh=0.2):
  """
  Recommend aggregation timescales and encoder types for time series data

  :param timestamp: sampling times of the time series
  :param value: value of the time series
  :param thresh: aggregation threshold (default value based on experiments with NAB data)
  :return: new_sampling_interval, a string for suggested sampling interval (e.g., 300000ms)
  :return: useTimeOfDay, a bool variable for whether to use time of day encoder
  :return: useDayOfWeek, a bool variable for whether to use day of week encoder
  """

  # The data may have inhomogeneous sampling rate, here we take the median
  # of the sampling intervals and resample the data with the same sampling intervals
  dt = np.median(np.diff(timestamp))
  dt_sec = dt.astype('float32')
  (timestamp, value) = resample_data(timestamp, value, dt)

  (cwtmatr, cwt_var, time_scale) = calculate_cwt(dt_sec, value)
  cum_cwt_var = np.cumsum(cwt_var)

  # decide aggregation window
  new_sampling_interval_sec = determine_aggregation_window(time_scale, cum_cwt_var, thresh, dt_sec, data_length=len(value))
  new_sampling_interval = str(int(new_sampling_interval_sec * 1000))+'ms'

  # decide whether to use TimeOfDay and DayOfWeek encoders
  (useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max) = get_local_maxima(cwt_var, time_scale)

  print "original sampling interval (sec) ", dt_sec
  print "suggested sampling interval (sec) ", new_sampling_interval_sec
  print "use TimeOfDay encoder? ", useTimeOfDay
  print "use DayOfWeek encoder? ", useDayOfWeek
  return (new_sampling_interval, useTimeOfDay, useDayOfWeek)