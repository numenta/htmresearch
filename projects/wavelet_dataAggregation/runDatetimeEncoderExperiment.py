import pandas as pd
import numpy as np
from runDataAggregationExperiment import get_suggested_timescale_and_encoder
from os.path import isfile, join, exists

import matplotlib.pyplot as plt
plt.close('all')
plt.ion()

"""
You need to run NAB with four different sets of model parameters before using this script


# go to the NAB directory, create a new result directory named results_encoder/

# the model parameters are located in ./nab_modelparams

# first use model_params_value_only
python run.py -d numenta --detect --dataDir data_processed/ --resultsDir results_encoder/value_only --score

# second use model_params_time_of_day
python run.py -d numenta --detect --dataDir data_processed/ --resultsDir results_encoder/time_of_day/ --score

# third use model_params_day_of_week
python run.py -d numenta --detect --dataDir data_processed/ --resultsDir results_encoder/day_of_week/ --score


"""

NABPath = '/Users/ycui/nta/NAB/'

def loadNABscore(encoderType):
  scorefile = NABPath + "results_encoder/" + encoderType + "/numenta/numenta_standard_scores.csv"
  score = pd.read_csv(scorefile, header=0)
  score = score[:-1]
  score = score.sort('File')
  score.index = range(len(score))
  return score


score_value = loadNABscore("value_only")
score_time_of_day = loadNABscore("time_of_day")
score_day_of_week = loadNABscore("day_of_week")
score_both = loadNABscore("time_of_day_and_day_of_week")

fileList = score_value['File'].values
better_with_time_of_day = (score_time_of_day['Score'] > score_value['Score'])
better_with_day_of_week = (score_day_of_week['Score'] > score_value['Score'])


dataPath = NABPath+'/data'
useTimeOfDayEncoder = []
useDayOfWeekEncoder = []
for i in xrange(len(score_value)):
  filename = join(dataPath, score_value['File'][i])
  dat = pd.read_csv(filename, header=0, names=['timestamp', 'value'])
  (new_sampling_interval, useTimeOfDay, useDayOfWeek) = get_suggested_timescale_and_encoder((dat))

  useTimeOfDayEncoder.append(useTimeOfDay)
  useDayOfWeekEncoder.append(useDayOfWeek)

  print " file: ", filename, " useTimeOfDay: ", useTimeOfDay, " useDayOfWeek: ", useDayOfWeek

useTimeOfDayEncoder = np.array(useTimeOfDayEncoder)
useDayOfWeekEncoder = np.array(useDayOfWeekEncoder)
resultMat = np.array([score_time_of_day['Score'] > score_value['Score'], useTimeOfDayEncoder])


result = pd.DataFrame(np.transpose([score_day_of_week.File.values,
                                    useTimeOfDayEncoder,
                                    useDayOfWeekEncoder,
                                    score_value['Score'].values,
                                    score_time_of_day['Score'].values,
                                    score_day_of_week['Score'].values,
                                    score_value['FP'].values,
                                    score_time_of_day['FP'].values,
                                    score_value['TP'].values,
                                    score_time_of_day['TP'].values]),
             columns=['FileName', 'useTimeOfDay', 'useDayOfWeek',
                      'scoreWithValue', 'scoreWithTimeOfDay', 'scoreWithDayOfWeek',
                      'FPWithValue', 'FPWithTimeOfDay',
                      'TPWithValue', 'TPWithTimeOfDay'])
result.to_csv('DatetimeEncoderExperiment.csv')

# NAB score according to the algorithm's suggestion
scoreSelect = []
for i in xrange(len(useTimeOfDayEncoder)):
  if useTimeOfDayEncoder[i]:
    scoreSelect.append(score_time_of_day['Score'].values[i])
  else:
    scoreSelect.append(score_value['Score'].values[i])
print " mean score without timeOfDay, ", np.mean(result.scoreWithValue)
print " mean score with timeOfDay, ", np.mean(result.scoreWithTimeOfDay)
print " mean score with algorithm, ", np.mean(scoreSelect)

# plot experiment results
idx = np.where(useTimeOfDayEncoder)[0]
result_true = [np.sum(result.ix[idx].scoreWithValue < result.ix[idx].scoreWithTimeOfDay),
           np.sum(result.ix[idx].scoreWithValue == result.ix[idx].scoreWithTimeOfDay),
           np.sum(result.ix[idx].scoreWithValue > result.ix[idx].scoreWithTimeOfDay)]
idx = np.where(useTimeOfDayEncoder == False)[0]
result_false = [np.sum(result.ix[idx].scoreWithValue < result.ix[idx].scoreWithTimeOfDay),
           np.sum(result.ix[idx].scoreWithValue == result.ix[idx].scoreWithTimeOfDay),
           np.sum(result.ix[idx].scoreWithValue > result.ix[idx].scoreWithTimeOfDay)]

fig, ax = plt.subplots()
rec1 = ax.bar([0, 4], [result_true[0], result_false[0]], color='b')
rec2 = ax.bar([1, 5], [result_true[1], result_false[1]], color='y')
rec3 = ax.bar([2, 6], [result_true[2], result_false[2]], color='r')
ax.set_xticks([1.5, 5.5])
ax.set_xticklabels(('Suggest to use TimeOfDay', 'Suggest not to use TimeOfDay'))
ax.set_ylabel('Number of Datasets')
plt.legend((rec1[0], rec2[0], rec3[0]),
           ('Better NAB score with TimeOfDay',
            'Same NAB score with TimeOfDay',
            'Worse NAB score with TimeOfDay'))
plt.savefig('experimentWithTimeOfDayEncoder.pdf')


idx = np.where(useDayOfWeekEncoder)[0]
result_true = [np.sum(result.ix[idx].scoreWithValue < result.ix[idx].scoreWithDayOfWeek),
           np.sum(result.ix[idx].scoreWithValue == result.ix[idx].scoreWithDayOfWeek),
           np.sum(result.ix[idx].scoreWithValue > result.ix[idx].scoreWithDayOfWeek)]
idx = np.where(useDayOfWeekEncoder == False)[0]
result_false = [np.sum(result.ix[idx].scoreWithValue < result.ix[idx].scoreWithDayOfWeek),
           np.sum(result.ix[idx].scoreWithValue == result.ix[idx].scoreWithDayOfWeek),
           np.sum(result.ix[idx].scoreWithValue > result.ix[idx].scoreWithDayOfWeek)]

fig, ax = plt.subplots()
rec1 = ax.bar([0, 4], [result_true[0], result_false[0]], color='b')
rec2 = ax.bar([1, 5], [result_true[1], result_false[1]], color='y')
rec3 = ax.bar([2, 6], [result_true[2], result_false[2]], color='r')
ax.set_xticks([1.5, 5.5])
ax.set_xticklabels(('Suggest to use DayOfWeek', 'Suggest not to use DayOfWeek'))
ax.set_ylabel('Number of Datasets')
plt.legend((rec1[0], rec2[0], rec3[0]),
           ('Better NAB score with DayOfWeek',
            'Same NAB score with DayOfWeek',
            'Worse NAB score with DayOfWeek'))
plt.savefig('experimentWithDayOfWeek.pdf')

#
# fig=plt.figure()
# plt.imshow(resultMat, interpolation='nearest', aspect='auto')
#
# resultMat2 = np.array([score_day_of_week['Score'] > score_value['Score'], useDayOfWeekEncoder])
#
# fig=plt.figure()
# plt.imshow(resultMat2, interpolation='nearest', aspect='auto')
#
#
# result.ix[idx]
# result.ix[idx].scoreWithValue
# idx = np.where(useDayOfWeekEncoder)[0]
# np.mean(score_day_of_week.Score.values[idx])
# np.mean(score_value.Score.values[idx])