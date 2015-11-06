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
Automatically find data aggregation window and suggest whether to use TimeOfDay 
and DayOfWeek encoder.

Example usage:

(timestamps, values) = read_csv_files('example_data/art_daily_flatmiddle.csv')
(new_sampling_interval, useTimeOfDay, 
useDayOfWeek) = get_suggested_timescale_and_encoder(timestamps, values)

"""

import csv
import numpy as np

_mode_from_name_dict = {
  'v': 0,
  's': 1,
  'f': 2
  }



def _convolve(a, v, mode='full'):
  """
  Returns the discrete, linear convolution of two one-dimensional sequences.

  The convolution operator is often seen in signal processing, where it
  models the effect of a linear time-invariant system on a signal [1]_.  In
  probability theory, the sum of two independent random variables is
  distributed according to the convolution of their individual
  distributions.

  If `v` is longer than `a`, the arrays are swapped before computation.

  Parameters
  ----------
  a : (N,) array_like
      First one-dimensional input array.
  v : (M,) array_like
      Second one-dimensional input array.
  mode : {'full', 'valid', 'same'}, optional
      'full':
        By default, mode is 'full'.  This returns the convolution
        at each point of overlap, with an output shape of (N+M-1,). At
        the end-points of the convolution, the signals do not overlap
        completely, and boundary effects may be seen.

      'same':
        Mode `same` returns output of length ``max(M, N)``.  Boundary
        effects are still visible.

      'valid':
        Mode `valid` returns output of length
        ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
        for points where the signals overlap completely.  Values outside
        the signal boundary have no effect.

  Returns
  -------
  out : ndarray
      Discrete, linear convolution of `a` and `v`.

  References
  ----------
  .. [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.

  """
  a, v = np.array(a, ndmin=1), np.array(v, ndmin=1)
  if len(v) > len(a):
    a, v = v, a
  if len(a) == 0:
    raise ValueError('a cannot be empty')
  if len(v) == 0:
    raise ValueError('v cannot be empty')
  mode = _mode_from_name(mode)
  return np.core.multiarray.correlate(a, v[::-1], mode)



def _mode_from_name(mode):
  if isinstance(mode, basestring):
    return _mode_from_name_dict[mode.lower()[0]]
  return mode



def _ricker_wavelet(points, a):
  """
  Return a Ricker wavelet, also known as the "Mexican hat wavelet".

  It models the function:

      ``A (1 - x^2/a^2) exp(-t^2/a^2)``,

  where ``A = 2/sqrt(3a)pi^1/3``.

  Parameters
  ----------
  points : int
      Number of points in `vector`.
      Will be centered around 0.
  a : scalar
      Width parameter of the wavelet.

  Returns
  -------
  vector : (N,) ndarray
      Array of length `points` in shape of ricker curve.

  """
  A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
  wsq = a ** 2
  vec = np.arange(0, points) - (points - 1.0) / 2
  tsq = vec ** 2
  mod = (1 - tsq / wsq)
  gauss = np.exp(-tsq / (2 * wsq))
  total = A * mod * gauss
  return total



def _cwt(data, wavelet, widths):
  """
  Continuous wavelet transform.

  Performs a continuous wavelet transform on `data`,
  using the `wavelet` function. A CWT performs a convolution
  with `data` using the `wavelet` function, which is characterized
  by a width parameter and length parameter.

  Parameters
  ----------
  data : (N,) ndarray
      data on which to perform the transform.
  wavelet : function
      Wavelet function, which should take 2 arguments.
      The first argument is the number of points that the returned vector
      will have (len(wavelet(width,length)) == length).
      The second is a width parameter, defining the size of the wavelet
      (e.g. standard deviation of a gaussian). See `ricker`, which
      satisfies these requirements.
  widths : (M,) sequence
      Widths to use for transform.

  Returns
  -------
  cwt: (M, N) ndarray
      Will have shape of (len(data), len(widths)).

  """
  output = np.zeros([len(widths), len(data)])
  for ind, width in enumerate(widths):
    wavelet_data = wavelet(min(10 * width, len(data)), width)
    output[ind, :] = _convolve(data, wavelet_data,
                               mode='same')
  return output



def read_csv_files(fileName):
  """
  Read csv data file, the data file must have two columns
  with header "timestamp", and "value"
  """
  fileReader = csv.reader(open(fileName, 'r'))
  fileReader.next()  # skip header line
  timestamps = []
  values = []

  for row in fileReader:
    timestamps.append(row[0])
    values.append(row[1])

  timestamps = np.array(timestamps, dtype='datetime64')
  values = np.array(values, dtype='float32')

  return timestamps, values



def resample_data(timestamp, sig, new_sampling_interval):
  """
  Resample time series data at new sampling interval using linear interpolation.
  Note: the resampling function is using interpolation, 
  it may not be appropriate for aggregation purpose
  :param timestamp: timestamp in numpy datetime64 type
  :param sig: value of the time series.
  :param new_sampling_interval: new sampling interval.
  """
  nSampleNew = np.floor((timestamp[-1] - timestamp[0])
                        / new_sampling_interval).astype('int') + 1

  timestamp_new = np.empty(nSampleNew, dtype='datetime64[s]')
  for sampleI in xrange(nSampleNew):
    timestamp_new[sampleI] = timestamp[0] + sampleI * new_sampling_interval

  sig_new = np.interp((timestamp_new - timestamp[0]).astype('float32'),
                      (timestamp - timestamp[0]).astype('float32'), sig)

  return timestamp_new, sig_new



def calculate_cwt(sampling_interval, value):
  """
  Calculate continuous wavelet transformation (CWT)
  Return variance of the cwt coefficients overtime and its cumulative
  distribution

  :param sampling_interval: sampling interval of the time series
  :param value: value of the time series
  """

  #t = np.array(range(len(value))) * sampling_interval
  widths = np.logspace(0, np.log10(len(value) / 20), 50)

  T = int(widths[-1])

  # continuous wavelet transformation with ricker wavelet
  cwtmatr = _cwt(value, _ricker_wavelet, widths)
  cwtmatr = cwtmatr[:, 4 * T:-4 * T]
  #value = value[4 * T:-4 * T]
  #t = t[4 * T:-4 * T]

  #freq = 1 / widths.astype('float') / sampling_interval / 4
  time_scale = widths * sampling_interval * 4

  # variance of wavelet power
  cwt_var = np.var(np.abs(cwtmatr), axis=1)
  cwt_var = cwt_var / np.sum(cwt_var)

  return cwtmatr, cwt_var, time_scale



def get_local_maxima(cwt_var, time_scale):
  """
  Find local maxima from the wavelet coefficient variance spectrum
  A strong maxima is defined as
  (1) At least 10% higher than the nearest local minima
  (2) Above the baseline value

  The algorithm will suggest an encoder if its corresponding
  periodicity is close to a strong maxima:
  (1) horizontally must within the nearest local minimum
  (2) vertically must within 50% of the peak of the strong maxima
  """

  # peak & valley detection
  local_min = (np.diff(np.sign(np.diff(cwt_var))) > 0).nonzero()[0] + 1
  local_max = (np.diff(np.sign(np.diff(cwt_var))) < 0).nonzero()[0] + 1

  baseline_value = 1.0 / len(cwt_var)

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
      right_local_min = len(cwt_var) - 1
      right_local_min_value = cwt_var[-1]
    else:
      right_local_min = local_min[right_local_min[0]]
      right_local_min_value = cwt_var[right_local_min]

    local_max_value = cwt_var[local_max[i]]
    nearest_local_min_value = np.max(left_local_min_value,
                                     right_local_min_value)
    if ((local_max_value - nearest_local_min_value) / nearest_local_min_value
          > 0.1 and local_max_value > baseline_value):
      strong_local_max.append(local_max[i])

      if (time_scale[left_local_min] < dayPeriod < time_scale[right_local_min]
          and cwt_var_at_dayPeriod > local_max_value * 0.5):
        useTimeOfDay = True

      if (time_scale[left_local_min] < weekPeriod < time_scale[right_local_min]
          and cwt_var_at_weekPeriod > local_max_value * 0.5):
        useDayOfWeek = True

  return useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max



def determine_aggregation_window(time_scale, cum_cwt_var, thresh,
                                 dt_sec, data_length):
  cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]
  aggregation_time_scale = cutoff_time_scale / 10.0
  if aggregation_time_scale < dt_sec * 4:
    aggregation_time_scale = dt_sec * 4

  if data_length < 1000:
    aggregation_time_scale = dt_sec
  else:
    # make sure there is > 1000 records after aggregation
    dt_max = float(data_length) / 1000.0 * dt_sec
    if aggregation_time_scale > dt_max > dt_sec:
      aggregation_time_scale = dt_max

  return aggregation_time_scale



def get_suggested_timescale_and_encoder(timestamp, value, thresh=0.2):
  """
  Recommend aggregation timescales and encoder types for time series data

  :param timestamp: sampling times of the time series
  :param value: value of the time series
  :param thresh: aggregation threshold 
    (default value based on experiments with NAB data)
  :return: new_sampling_interval, a string for suggested sampling interval 
    (e.g., 300000ms)
  :return: useTimeOfDay, a bool variable for whether to use time of day encoder
  :return: useDayOfWeek, a bool variable for whether to use day of week encoder
  """

  # The data may have inhomogeneous sampling rate, here we take the median
  # of the sampling intervals and resample the data with the same sampling 
  # intervals
  dt = np.median(np.diff(timestamp))
  dt_sec = dt.astype('float32')
  (timestamp, value) = resample_data(timestamp, value, dt)

  (cwtmatr, cwt_var, time_scale) = calculate_cwt(dt_sec, value)
  cum_cwt_var = np.cumsum(cwt_var)

  # decide aggregation window
  new_sampling_interval_sec = determine_aggregation_window(time_scale,
                                                           cum_cwt_var,
                                                           thresh,
                                                           dt_sec,
                                                           len(value))
  new_sampling_interval = str(int(new_sampling_interval_sec * 1000)) + 'ms'

  # decide whether to use TimeOfDay and DayOfWeek encoders
  (useTimeOfDay, useDayOfWeek, local_min, local_max,
   strong_local_max) = get_local_maxima(cwt_var, time_scale)

  print "original sampling interval (sec) ", dt_sec
  print "suggested sampling interval (sec) ", new_sampling_interval_sec
  print "use TimeOfDay encoder? ", useTimeOfDay
  print "use DayOfWeek encoder? ", useDayOfWeek
  return new_sampling_interval, useTimeOfDay, useDayOfWeek
