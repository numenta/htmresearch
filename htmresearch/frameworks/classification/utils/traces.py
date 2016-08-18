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
