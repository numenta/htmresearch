import csv
import matplotlib.pyplot as plt
from settings import RESULTS_DIR, DATA_DIR, SIGNAL_TYPES

plt.figure()
for signal_type in SIGNAL_TYPES:
  filePath = "%s/%s.csv" %(DATA_DIR, signal_type)
  with open(filePath, 'rb') as f:
    reader = csv.reader(f)
    headers = reader.next()
    x = []
    data = []
    labels = []
    for i, values in enumerate(reader):
      record = dict(zip(headers, values))
      x.append(i)
      data.append(record['y'])
      labels.append(record['label'])

    plt.subplot(2, 1, SIGNAL_TYPES.index(signal_type) + 1)
    plt.plot(x, data)

plt.show()
