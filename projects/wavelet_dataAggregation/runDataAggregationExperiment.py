from os.path import isfile, join, exists
import pandas as pd
import numpy as np
from scipy import signal
import numpy.matlib
import time
import csv

import os
import time
os.environ['TZ'] = 'GMT'
time.tzset()

display = True
if display:
  import matplotlib.pyplot as plt
  plt.close('all')
  plt.ion()


def plotWaveletPower(sig, cwtmatr, time_scale, x_range=None, title=''):
  """
  Display wavelet transformations along with the original data
  :param sig: original sigal
  :param cwtmatr: cwt coefficients
  :param time_scale: time scales of wavelets
  :param x_range: x range of the plot
  :param title: title of the plot
  """
  if x_range is None:
    x_range = range(0, cwtmatr.shape[1])

  fig, ax = plt.subplots(nrows=2, ncols=1)

  y_time_scale_tick = ['1-sec', '1mins', '5mins', '30mins', '60mins', '2hrs', '4hrs', '12hrs', '1day', '1week']
  y_time_scale = [1, 60, 300, 1800, 3600, 7200, 14400, 43200, 86400, 604800]

  y_tick = (np.log10(y_time_scale) - np.log10(time_scale[0]) ) / \
           (np.log10(time_scale[-1]) - np.log10(time_scale[0])) * (len(time_scale)-1)
  good_tick = np.where(np.logical_and(y_tick >= 0, y_tick < len(time_scale)))[0]
  y_tick = y_tick[good_tick]
  y_time_scale_tick = [y_time_scale_tick[i] for i in good_tick]

  ax[0].imshow(np.abs(cwtmatr[:, x_range]), aspect='auto')
  ax[0].set_yticks(y_tick)
  ax[0].set_yticklabels(y_time_scale_tick)
  ax[0].set_xlabel(' Time ')
  ax[0].set_title(title)

  ax[1].plot(sig[x_range])
  ax[1].set_xlabel(' Time ')
  ax[1].autoscale(tight=True)
  plt.show()


def calculate_cwt(sampling_interval, sig, figDir='./', fileName='./', display=True):
  """
  Calculate continuous wavelet transformation (CWT)
  Return variance of the cwt coefficients overtime and its cumulative
  distribution

  :param sampling_interval: sampling interval of the time series
  :param sig: value of the time series
  :param figDir: directory of cwt plots
  :param fileName: name of the dataset, used for determining figDir
  :param display: whether to create the cwt plot
  """

  t = np.array(range(len(sig)))*sampling_interval
  widths = np.logspace(0, np.log10(len(sig)/20), 50)

  T = int(widths[-1])

  # continulus wavelet transformation with ricker wavelet
  cwtmatr = signal.cwt(sig, signal.ricker, widths)
  cwtmatr = cwtmatr[:, 4*T:-4*T]
  sig = sig[4*T:-4*T]
  t = t[4*T:-4*T]

  freq = 1/widths.astype('float') / sampling_interval / 4
  time_scale = widths * sampling_interval * 4

  # variance of wavelet power
  cwt_var = np.var(np.abs(cwtmatr), axis=1)
  cwt_var = cwt_var/np.sum(cwt_var)
  cum_cwt_var = np.cumsum(cwt_var)

  (useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max) = get_local_maxima(cwt_var, time_scale)

  if not exists(figDir):
      os.makedirs(figDir)

  if display:
    # plot wavelet coefficients along with the raw signal
    plt.close('all')
    plotWaveletPower(sig, cwtmatr, time_scale)
    plt.savefig(join(figDir, fileName + 'wavelet_transform.pdf'))

    fig, axs = plt.subplots(nrows=2, ncols=1)
    ax = axs[0]
    ax.plot(time_scale, cwt_var, '-o')
    ax.axvline(x=86400, color='c')
    ax.axvline(x=604800, color='c')

    for _ in xrange(len(local_max)):
      ax.axvline(x=time_scale[local_max[_]], color='r')
    for _ in xrange(len(strong_local_max)):
      ax.axvline(x=time_scale[strong_local_max[_]], color='k')
    for _ in xrange(len(local_min)):
      ax.axvline(x=time_scale[local_min[_]], color='b')

    ax.set_xscale('log')
    ax.set_xlabel(' Time Scale (sec) ')
    ax.set_ylabel(' Variance of Power')
    ax.autoscale(tight='True')
    ax.set_title(fileName)

    ax = axs[1]
    ax.plot(time_scale, cum_cwt_var, '-o')
    ax.set_xscale('log')
    ax.set_xlabel(' Time Scale (sec) ')
    ax.set_ylabel(' Accumulated Variance of Power')
    ax.autoscale(tight='True')
    plt.title(['useTimeOfDay: '+str(useTimeOfDay)+' useDayOfWeek: '+str(useDayOfWeek)])
    plt.savefig(join(figDir, fileName + 'aggregation_time_scale.pdf'))

  return cum_cwt_var, cwt_var, time_scale


def get_local_maxima(cwt_var, time_scale):
  """
  Find local maxima from the wavelet coefficient variance spectrum
  A strong maxima is defined as
  (1) At least 10% higher than the nearest local minima
  (2) Above the baseline value
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
    if ( (local_max_value - nearest_local_min_value)/nearest_local_min_value > 0.1 and
           local_max_value > baseline_value):
      strong_local_max.append(local_max[i])

      if (time_scale[left_local_min] < dayPeriod and
              dayPeriod < time_scale[right_local_min] and
              cwt_var_at_dayPeriod > local_max_value/2.0):
        if np.abs(dayPeriod - time_scale[local_max[i]])/dayPeriod < 0.5:
          useTimeOfDay = True

      if (time_scale[left_local_min] < weekPeriod and
              weekPeriod < time_scale[right_local_min] and
              cwt_var_at_weekPeriod > local_max_value/2.0):
        if np.abs(weekPeriod - time_scale[local_max[i]])/weekPeriod < 0.5:
          useDayOfWeek = True

  return useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max


def get_suggested_timescale_and_encoder(timestamp, sig, thresh=0.2):
  dt = np.median(np.diff(timestamp))
  dt_sec = dt.astype('float32')
  # resample the data with homogeneous sampling intervals
  (timestamp, sig) = resample_data(timestamp, sig, dt, display=True)

  (cum_cwt_var, cwt_var, time_scale) = calculate_cwt(dt_sec, sig)

  (useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max) = get_local_maxima(cwt_var, time_scale)

  cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]
  aggregation_time_scale = cutoff_time_scale/10.0
  if aggregation_time_scale < dt_sec*4:
    aggregation_time_scale = dt_sec*4

  new_sampling_interval = str(int(aggregation_time_scale/4))+'S'

  return (new_sampling_interval, useTimeOfDay, useDayOfWeek)


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


def writeCSVfiles(fileName, timestamp, value):
  """
  write data to csv file,
  the data file will have two columns with header "timestamp", and "value"
  """
  fileWriter = csv.writer(open(fileName, 'w'))
  fileWriter.writerow(['timestamp', 'value'])
  for i in xrange(len(timestamp)):
    fileWriter.writerow([timestamp[i].astype('O').strftime("%Y-%m-%d %H:%M:%S"),
                         value[i]])


def resample_data(timestamp, sig, new_sampling_interval, display=False):
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

  if display:
    plt.figure(3)
    plt.plot(timestamp, sig)
    plt.plot(timestamp_new, sig_new)
    plt.legend(['before resampling', 'after resampling'])

  return (timestamp_new, sig_new)


def aggregate_data(thresh_list, dataFile, aggregatedDataPath, waveletDir='./wavelet/', display=False, verbose=0):
  """
  Aggregate individual dataset, the aggregated data will be saved at aggregatedDataFile
  :param thresh: aggregation threshold
  :param dataFile: path of the original datafile
  :param aggregatedDataFile: path of the aggregated datafile
  :param waveletDir: path of wavelet transformations (for visual inspection)
  """
  data_file_dir = dataFile.split('/')

  (timestamp, sig) = readCSVfiles(dataFile)

  # dt = (timestamp[len(sig)-1] - timestamp[0])/(len(sig)-1)
  dt = np.median(np.diff(timestamp))
  dt_sec = dt.astype('float32')
  # resample the data with homogeneous sampling intervals
  (timestamp, sig) = resample_data(timestamp, sig, dt, display=True)

  (cum_cwt_var, cwt_var, time_scale) = calculate_cwt(dt_sec, sig,
                                                     display=display,
                                                     figDir=join(waveletDir, data_file_dir[-2]),
                                                     fileName=data_file_dir[-1])

  for thresh in thresh_list:
    new_data_dir = join(aggregatedDataPath, 'thresh='+str(thresh), data_file_dir[-2])
    if not exists(new_data_dir):
        os.makedirs(new_data_dir)

    new_data_file = join(new_data_dir, data_file_dir[-1])

    # determine aggregation time scale
    cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]
    aggregation_time_scale = cutoff_time_scale/10.0
    if aggregation_time_scale < dt_sec*4:
      aggregation_time_scale = dt_sec*4

    new_sampling_interval = np.timedelta64(int(aggregation_time_scale/4 * 1000), 'ms')
    nSampleNew = np.floor((timestamp[-1] - timestamp[0])/new_sampling_interval).astype('int') + 1

    timestamp_new = np.empty(nSampleNew, dtype='datetime64[s]')
    value_new = np.empty(nSampleNew, dtype='float32')

    left_sampleI = 0
    new_sampleI = 0

    for sampleI in xrange(len(sig)):
      if timestamp[sampleI] >= timestamp[0] + new_sampleI * new_sampling_interval:
        timestamp_new[new_sampleI] = timestamp[0] + new_sampleI * new_sampling_interval
        value_new[new_sampleI] = (np.mean(sig[left_sampleI:sampleI+1]))
        left_sampleI = sampleI+1
        new_sampleI += 1

    writeCSVfiles(new_data_file, timestamp_new, value_new)

    if verbose > 0:
      print " original length: ", len(sig), "\t file: ", dataFile
      print "\t\tthreshold: ", thresh, "\t new length: ", len(value_new)


def aggregate_nab_data(thresh_list, dataPath='data/',
                       aggregatedDataPath='data_aggregate/',
                       waveletDir='wavelet/',
                       verbose=0):
  """
  Aggregate all NAB data using the wavelet transformation based algorithm

  :param thresh_list: threshold of the aggregation, a number in [0, 1)
  :param dataPath: path of the original NAB data
  :param aggregatedDataPath: path of the aggregated NAB data
  :param waveletDir: path of wavelet transformations (for visual inspection)
  """

  if not exists(aggregatedDataPath):
      os.makedirs(aggregatedDataPath)

  dataDirs = [join(dataPath, f) for f in os.listdir(dataPath) if not isfile(join(dataPath, f))]

  for dir in dataDirs:
    datafiles = [join(dir, f) for f in os.listdir(dir) if isfile(join(dir, f))]

    for i in range(len(datafiles)):
      aggregate_data(thresh_list, datafiles[i], aggregatedDataPath, waveletDir, verbose=verbose)

        
def get_pre_aggregated_anomaly_score(data_path, result_folder, result_folder_pre_aggregate):
  """
  This function transforms anomaly scores on the aggregated data file (in result_folder)
  to the original sampling rate of the data (in data_path) before aggregation. The new anomaly
  score will be saved to result_folder_pre_aggregate
  """

  dataDirs = [join(result_folder, f) for f in os.listdir(result_folder) if not isfile(join(result_folder, f))]

  for dir in dataDirs:
    resultfiles = [join(dir, f) for f in os.listdir(dir) if isfile(join(dir, f))]

    for i in range(len(resultfiles)):
      result_file_dir = resultfiles[i].split('/')

      original_data_file = join(data_path, result_file_dir[-2], result_file_dir[-1][8:])
      dat = pd.read_csv(original_data_file, header=0, names=['timestamp', 'value'])
      result = pd.read_csv(resultfiles[i], header=0,
                           names=['timestamp', 'value', 'anomaly_score', 'raw_score', 'label'])

      time_stamp_pre_aggregation = pd.to_datetime(dat.timestamp)
      time_stamp_after_aggregation = pd.to_datetime(result.timestamp)

      binary_anomaly_score_pre_aggregation = np.zeros(shape=(len(dat),))
      binary_anomaly_score_after_aggregation = np.zeros(shape=(len(result),))
      for j in range(len(result)):
        if result.anomaly_score[j] > .5:
          binary_anomaly_score_after_aggregation[j] = 1

          idx_original = np.argmin(abs(time_stamp_pre_aggregation - time_stamp_after_aggregation[j]))
          binary_anomaly_score_pre_aggregation[idx_original] = 1

      value_pre_aggregation = dat.value.values
      raw_score_pre_aggregation = np.zeros(shape=(len(dat),))
      label_pre_aggregation = np.zeros(shape=(len(dat),))

      # raw_score_pre_aggregation = np.interp(time_stamp_original, time_stamp_after_aggregation, result.raw_score.values)
      result_pre_aggregate = pd.DataFrame(np.transpose(np.array([time_stamp_pre_aggregation,
                                                    value_pre_aggregation,
                                                    binary_anomaly_score_pre_aggregation,
                                                    raw_score_pre_aggregation,
                                                    label_pre_aggregation])),
                                          columns=['timestamp', 'value', 'anomaly_score', 'raw_score', 'label'])

      result_file_dir_pre_aggregate = join(result_folder_pre_aggregate, result_file_dir[-2])
      if not exists(result_file_dir_pre_aggregate):
          os.makedirs(result_file_dir_pre_aggregate)
      result_file_pre_aggregate = join(result_file_dir_pre_aggregate, result_file_dir[-1])
      result_pre_aggregate.to_csv(result_file_pre_aggregate, index=False)
      print " write pre-aggregated file to ", result_file_pre_aggregate

      # compare anomaly scores before and after aggregations for individual files
      # plt.figure(2)
      # plt.plot(time_stamp_after_aggregation, binary_anomaly_score_after_aggregation)
      # plt.plot(time_stamp_pre_aggregation, binary_anomaly_score_pre_aggregation)


def runTimeVsDataLength(dataPath):
  """
  Plot Data Aggregation Algorithm Runtime vs length of the data
  """

  dataDirs = [join(dataPath, f) for f in os.listdir(dataPath) if not isfile(join(dataPath, f))]

  thresh = 0.2

  dataLength = []
  runTime = []
  for dir in dataDirs:
    datafiles = [join(dir, f) for f in os.listdir(dir) if isfile(join(dir, f))]

    for i in range(len(datafiles)):
      (timestamp, sig) = readCSVfiles(datafiles[i])

      dataLength.append(len(sig))

      start_time = time.time()

      aggregate_data([thresh], datafiles[i], aggregatedDataPath='data_aggregate/', display=False)

      end_time = time.time()

      print " length: ", len(sig), " file: ", datafiles[i], " Time: ", (end_time - start_time)

      runTime.append(end_time - start_time)

  plt.figure()
  plt.plot(dataLength, runTime, '*')
  plt.xlabel(' Dataset Size (# Record)')
  plt.ylabel(' Runtime (seconds) ')
  plt.savefig('RuntimeVsDatasetSize.pdf')

  return (dataLength, runTime)


if __name__ == "__main__":

  NABPath = '/Users/ycui/nta/NAB/'
  currentPath = os.getcwd()

  thresh_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.40]

  # step 1: aggregate NAB data with different threshold
  print " aggregating NAB data ..."
  aggregate_nab_data(thresh_list, dataPath=NABPath+'data/', verbose=2)

  # step 2: run HTM on aggregated NAB data
  for thresh in thresh_list:
    resultsAggregatePath = currentPath + "/results_aggregate/thresh=" + str(thresh) + "/numenta"
    if not os.path.exists(resultsAggregatePath):
      os.os.makedirs(resultsAggregatePath)
    print " run HTM on aggregated data with threshold " + str(thresh)
    os.system("python " + NABPath + "run.py -d numenta --detect --dataDir " + currentPath + "/data_aggregate/thresh=" + str(thresh) + \
              "/ --resultsDir "+ currentPath + "/results_aggregate/thresh=" + str(thresh) + " --skipConfirmation")

  # step 3: get pre-aggregated anomaly score
  for thresh in thresh_list:
    preresultAggregatePath = currentPath + "/results_pre_aggregate/thresh=" + str(thresh) + "/numenta"
    if not os.path.exists(preresultAggregatePath):
      os.os.makedirs(preresultAggregatePath)
    get_pre_aggregated_anomaly_score(data_path=NABPath+'data/',
                                     result_folder='results_aggregate/thresh=' + str(thresh) + '/numenta',
                                     result_folder_pre_aggregate='results_pre_aggregate/thresh=' + str(thresh) + '/numenta')

  # step 4: run NAB scoring
  for thresh in thresh_list:
    print " run scoring on aggregated data with threshold " + str(thresh)
    os.system("python " + NABPath + "run.py -d numenta --score --skipConfirmation " +
           "--thresholdsFile " + NABPath + "config/thresholds.json " +
           "--resultsDir " + currentPath + "/results_pre_aggregate/thresh="+str(thresh)+"/")

  # step 5: read & compare scores
  standard_score = []
  data_length_all = []
  for thresh in thresh_list:
    scorefile = "./results_pre_aggregate/thresh=" + str(thresh) + "/numenta/numenta_standard_scores.csv"
    scoredf = pd.read_csv(scorefile, header=0)
    scoredf = scoredf.sort('File')
    scoredf.index = range(len(scoredf))
    standard_score.append(scoredf.Score.values[:-1])
    data_length = []
    for i in xrange(len(scoredf.File)-1):
      datafile = './data_aggregate/thresh=' + str(thresh) + '/' + scoredf.File[i]
      dat = pd.read_csv(datafile, header=0, names=['timestamp', 'value'])
      data_length.append(len(dat))
    data_length_all.append(data_length)

  data_length_all = np.array(data_length_all)
  standard_score = np.array(standard_score)

  short_dat = np.where(data_length_all[0, :] < 1000)[0]
  long_dat = np.where(data_length_all[0, :] > 1000)[0]
  use_dat = np.array(range(data_length_all.shape[1]))
  use_dat = long_dat

  # plt.imshow(data_length_all, interpolation='nearest', aspect='auto')

  # plot anomaly score vs aggregation threshold
  anomaly_score_diff = standard_score[:, long_dat] - numpy.matlib.repmat(standard_score[0, long_dat], len(thresh_list), 1)

  shortFileName = []
  for i in range(len(scoredf.File.values[:-1])):
    file = scoredf.File.values[i]
    fileName = file.split('/')[-1]
    fileName = fileName[:-4]
    shortFileName.append(fileName)
  fig=plt.figure()
  plt.imshow(anomaly_score_diff, interpolation='nearest', aspect='auto')
  ytickLoc = range(len(thresh_list))
  plt.yticks(ytickLoc, thresh_list)
  plt.xticks(range(len(scoredf.File)-1), shortFileName, rotation='vertical')
  plt.subplots_adjust(bottom=0.6)
  plt.ylabel(' Threshold')
  plt.title(' Anomaly Score Relative to BaseLine')
  plt.colorbar()
  plt.clim(-2, 2)
  plt.savefig('AnomalyScore_Vs_AggregationThreshold_EachFile.pdf')

  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(np.array(thresh_list)*100, np.median(standard_score[:, use_dat], 1), '-o')
  plt.plot(np.array(thresh_list)*100, np.mean(standard_score[:, use_dat], 1), '-o')
  plt.legend(['Median', 'Mean'])
  plt.xlabel(' Threshold (%)')
  plt.ylabel(' Median Anomaly Score ')

  plt.subplot(2, 1, 2)
  plt.plot(np.array(thresh_list)*100, np.median(data_length_all[:, use_dat], 1), '-o')
  plt.plot(np.array(thresh_list)*100, np.mean(data_length_all[:, use_dat], 1), '-o')
  plt.xlabel(' Threshold (%)')
  plt.ylabel(' Data Length ')
  plt.legend(['Median', 'Mean'])
  plt.savefig('AnomalyScore_Vs_AggregationThreshold.pdf')
  num_better_anomaly_score = []
  for i in xrange(len(thresh_list)-1):
    num_better_anomaly_score.append(len(np.where(standard_score[i+1, :] > standard_score[0, :])[0]))

  (dataLength, runTime) = runTimeVsDataLength(dataPath=NABPath+'data/')
