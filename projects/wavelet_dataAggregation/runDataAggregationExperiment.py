from os import listdir, makedirs, system, getcwd
from os.path import isfile, join, exists
import pandas as pd
import numpy as np
from scipy import signal
import numpy.matlib
import time

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


def calculate_cwt(dat, figDir='./', fileName='./', display=True):
  """
  Calculate continuous wavelet transformation (CWT)
  Return variance of the cwt coefficients overtime and its cumulative
  distribution

  :param dat: pandas dataframe with two columns, 'timestamp' and 'value'
  :param figDir: directory of cwt plots
  :param fileName: name of the dataset, used for determining figDir
  :param display: whether to create the cwt plot
  """
  sig = dat['value'].values
  timestamp = pd.to_datetime(dat['timestamp'])
  sampling_interval = timestamp[len(sig)-1] - timestamp[0]
  dt = sampling_interval.total_seconds()/(len(sig)-1)

  t = np.array(range(len(sig)))*dt
  widths = np.logspace(0, np.log10(len(sig)/20), 50)

  T = int(widths[-1])

  # continulus wavelet transformation with ricker wavelet
  cwtmatr = signal.cwt(sig, signal.ricker, widths)
  cwtmatr = cwtmatr[:, 4*T:-4*T]
  sig = sig[4*T:-4*T]
  t = t[4*T:-4*T]

  freq = 1/widths.astype('float') / dt / 4
  time_scale = widths * dt * 4

  # variance of wavelet power
  cwt_var = np.var(np.abs(cwtmatr), axis=1)
  cwt_var = cwt_var/np.sum(cwt_var)
  cum_cwt_var = np.cumsum(cwt_var)

  (useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max) = get_local_maxima(cwt_var, time_scale)

  if not exists(figDir):
      makedirs(figDir)

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

  dayPeriod = 86400
  weekPeriod = 604800

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
        useTimeOfDay = True

      if (time_scale[left_local_min] < weekPeriod and
              weekPeriod < time_scale[right_local_min] and
              cwt_var_at_weekPeriod > local_max_value/2.0):
        useDayOfWeek = True

  return useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max


def get_suggested_timescale_and_encoder(dat, thresh=0.2):

  sig = dat['value'].values
  timestamp = pd.to_datetime(dat.timestamp)
  sampling_interval = timestamp[len(sig)-1] - timestamp[0]
  dt = sampling_interval.total_seconds()/(len(sig)-1)
  (cum_cwt_var, cwt_var, time_scale) = calculate_cwt(dat, display=False)

  (useTimeOfDay, useDayOfWeek, local_min, local_max, strong_local_max) = get_local_maxima(cwt_var, time_scale)

  cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]
  aggregation_time_scale = cutoff_time_scale/10.0
  if aggregation_time_scale < dt*4:
    aggregation_time_scale = dt*4

  new_sampling_interval = str(int(aggregation_time_scale/4))+'S'

  return (new_sampling_interval, useTimeOfDay, useDayOfWeek)


def aggregate_nab_data(thresh_list, dataPath='data/',
                       aggregatedDataPath='data_aggregate', waveletDir='wavelet/'):
  """
  Aggregate NAB data using the wavelet transformation based algorithm

  :param thresh_list: threshold of the aggregation, a number in [0, 1)
  :param dataPath: path of the original NAB data
  :param aggregatedDataPath: path of the aggregated NAB data
  :param waveletDir: path of wavelet transformations (for visual inspection)
  """
  if not exists(aggregatedDataPath):
      makedirs(aggregatedDataPath)

  dataDirs = [join(dataPath, f) for f in listdir(dataPath) if not isfile(join(dataPath, f))]

  data_length = []
  for dir in dataDirs:
    datafiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

    for i in range(len(datafiles)):
      dat = pd.read_csv(datafiles[i], header=0, names=['timestamp', 'value'])
      data_file_dir = datafiles[i].split('/')
      data_length.append(len(dat))

      print " length: ", len(dat), " file: ", datafiles[i]

      sig = dat['value'].values
      timestamp = pd.to_datetime(dat.timestamp)
      sampling_interval = timestamp[len(sig)-1] - timestamp[0]
      dt = sampling_interval.total_seconds()/(len(sig)-1)

      (cum_cwt_var, cwt_var, time_scale) = calculate_cwt(dat,
                                                  figDir=join(waveletDir, data_file_dir[-2]),
                                                  fileName=data_file_dir[-1])

      ts = pd.Series(np.array(dat.value).astype('float32'), index=pd.to_datetime(dat.timestamp))
      # apply different threshold to the cumulative power variance
      for thresh in thresh_list:
        cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]
        aggregation_time_scale = cutoff_time_scale/10.0
        if aggregation_time_scale < dt*4:
          aggregation_time_scale = dt*4

        new_sampling_interval = str(int(aggregation_time_scale/4))+'S'
        ts_aggregate = ts.resample(new_sampling_interval, how='mean')
        ts_aggregate = ts_aggregate.interpolate(method='linear')

        timestamp = ts_aggregate.index
        value = np.array(ts_aggregate.values)
        dat_aggregated = pd.DataFrame(np.transpose(np.array([timestamp, value])), columns=['timestamp', 'value'])

        print "thresh: ", thresh, " original dt ", dt, " new dt: ", new_sampling_interval, \
              "original length: ", len(ts), " new length: ", len(ts_aggregate)

        new_data_dir = join(aggregatedDataPath, 'thresh='+str(thresh), data_file_dir[-2])
        if not exists(new_data_dir):
            makedirs(new_data_dir)

        new_data_file = join(new_data_dir, data_file_dir[-1])

        dat_aggregated.to_csv(new_data_file, index=False)

        
def get_pre_aggregated_anomaly_score(data_path, result_folder, result_folder_pre_aggregate):
  """
  This function transforms anomaly scores on the aggregated data file (in result_folder)
  to the original sampling rate of the data (in data_path) before aggregation. The new anomaly
  score will be saved to result_folder_pre_aggregate
  """

  dataDirs = [join(result_folder, f) for f in listdir(result_folder) if not isfile(join(result_folder, f))]

  for dir in dataDirs:
    resultfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

    for i in range(len(resultfiles)):
      result_file_dir = resultfiles[i].split('/')

      original_data_file = join(data_path, result_file_dir[-2], result_file_dir[-1][8:])
      dat = pd.read_csv(original_data_file, header=0, names=['timestamp', 'value'])
      result = pd.read_csv(resultfiles[i], header=0,
                           names=['timestamp', 'value', 'anomaly_score', 'raw_score', 'label'])

      time_stamp_pre_aggregation =pd.to_datetime(dat.timestamp)
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
          makedirs(result_file_dir_pre_aggregate)
      result_file_pre_aggregate = join(result_file_dir_pre_aggregate, result_file_dir[-1])
      result_pre_aggregate.to_csv(result_file_pre_aggregate, index=False)
      print " write pre-aggregated file to ", result_file_pre_aggregate

      # compare anomaly scores before and after aggregations
      # plt.figure(2)
      # plt.plot(time_stamp_after_aggregation, binary_anomaly_score_after_aggregation)
      # plt.plot(time_stamp_pre_aggregation, binary_anomaly_score_pre_aggregation)


def runTimeVsDataLength(dataPath):
  """
  Plot Data Aggregation Algorithm Runtime vs length of the data
  """

  dataDirs = [join(dataPath, f) for f in listdir(dataPath) if not isfile(join(dataPath, f))]

  thresh = 0.2

  dataLength = []
  runTime = []
  for dir in dataDirs:
    datafiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]

    for i in range(len(datafiles)):
      dat = pd.read_csv(datafiles[i], header=0, names=['timestamp', 'value'])
      data_file_dir = datafiles[i].split('/')
      dataLength.append(len(dat))

      sig = dat['value'].values
      timestamp = pd.to_datetime(dat.timestamp)
      sampling_interval = timestamp[len(sig)-1] - timestamp[0]
      dt = sampling_interval.total_seconds()/(len(sig)-1)
      ts = pd.Series(np.array(dat.value).astype('float32'), index=pd.to_datetime(dat.timestamp))

      start_time = time.time()
      (cum_cwt_var, time_scale) = calculate_cwt(dat, display=False)
      # apply different threshold to the cumulative power variance

      cutoff_time_scale = time_scale[np.where(cum_cwt_var >= thresh)[0][0]]

      end_time = time.time()

      print " length: ", len(dat), " file: ", datafiles[i], " Time: ", end_time - start_time

      runTime.append(end_time - start_time)

  plt.figure()
  plt.plot(dataLength, runTime, '*')
  plt.xlabel(' Dataset Size (# Record)')
  plt.ylabel(' Runtime (seconds) ')
  plt.savefig('RuntimeVsDatasetSize.pdf')

  return (dataLength, runTime)


if __name__ == "__main__":

  NABPath = '/Users/ycui/nta/NAB/'
  currentPath = getcwd()

  thresh_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.40]

  # step 1: aggregate NAB data with different threshold
  print " aggregating NAB data ..."
  aggregate_nab_data(thresh_list, dataPath=NABPath+'data/')

  # step 2: run HTM on aggregated NAB data
  for thresh in thresh_list:
    print " run HTM on aggregated data with threshold " + str(thresh)
    system("python " + NABPath + "run.py -d numenta --detect --dataDir data_aggregate/thresh=" + str(thresh) +
              "/ --resultsDir "+ currentPath + "/results_aggregate/thresh=" + str(thresh) + " --skipConfirmation")

  # step 3: get pre-aggregated anomaly score
  for thresh in thresh_list:
    get_pre_aggregated_anomaly_score(data_path=NABPath+'data/',
                                     result_folder='results_aggregate/thresh=' + str(thresh) + '/numenta',
                                     result_folder_pre_aggregate='results_pre_aggregate/thresh=' + str(thresh) + '/numenta')

  # step 4: run NAB scoring
  for thresh in thresh_list:
    print " run scoring on aggregated data with threshold " + str(thresh)
    system("python " + NABPath + "run.py -d numenta --score --skipConfirmation " +
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
