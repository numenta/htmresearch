import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

plt.ion()

year = 2015
record_num = []

aggregation_rule = {'Sum': sum}

ts_all = None
for month in xrange(1, 12):
  datafileName =  'yellow_tripdata_' + str(year) + '-' + "{:0>2d}".format(month) + '.csv'

  if os.path.isfile(datafileName):
    print " Load Datafile: ", datafileName
    df = pd.read_csv(datafileName, header=0, usecols=[1, 3])

    record_num.append(len(df))

    ts = pd.Series(np.array(df.passenger_count), index=pd.to_datetime(df.tpep_pickup_datetime))
    del df

    print " aggregate data at 30min resolution"
    ts_aggregate = ts.resample("30min", how=aggregation_rule)

    if ts_all is not None:
      print " concat ts_all"
      ts_all = pd.concat([ts_all, ts_aggregate])
    else:
      print " initialize ts_all"
      ts_all = ts_aggregate
  else:
    print datafileName, " not exist"

plt.close('all')
plt.figure(1)
plt.plot(ts_aggregate.index, ts_aggregate.Sum)
plt.figure(2)
plt.plot(ts_all.index, ts_all.Sum)

import csv
outputFile = open('nyc_taxi.csv',"w")
csvWriter = csv.writer(outputFile)
csvWriter.writerow(['timestamp', 'passenger_count'])
csvWriter.writerow(['datetime', 'int'])
csvWriter.writerow(['T', ''])
for i in range(len(ts_all)):
  csvWriter.writerow([ts_all.index[i], ts_all.values[i][0]])
outputFile.close()