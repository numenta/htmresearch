import csv
import matplotlib.pyplot as plt
from prettytable import PrettyTable



def plotSensorData(expSetups):
  """
  Plot several sensor data CSV recordings and highlights the sequences.
  @param expSetups: (list of dict) list of setup for each experiment
  """

  plt.figure()

  numExps = len(expSetups)
  t_headers = ['expId'] + expSetups[0].keys()
  t = PrettyTable(t_headers)
  for expSetup in expSetups:
    expId = expSetups.index(expSetup)
    expSetup['expId'] = expId
    row = [expSetup[th] for th in t_headers]
    t.add_row(row)

    filePath = expSetup['filePath']
    timesteps = []
    data = []
    labels = []
    categoriesLabelled = []
    categoryColors = ['r', 'y', 'g']
    with open(filePath, 'rb') as f:
      reader = csv.reader(f)
      headers = reader.next()

      # skip the 2 first rows
      reader.next()
      reader.next()

      for i, values in enumerate(reader):
        record = dict(zip(headers, values))
        timesteps.append(i)
        data.append(record['y'])
        labels.append(int(record['label']))

      plt.subplot(numExps, 1, expId + 1)
      plt.plot(timesteps, data, 'xb-', label='signal')

      previousLabel = labels[0]
      start = 0
      labelCount = 0
      numPoints = len(labels)
      for label in labels:

        if previousLabel != label or labelCount == numPoints - 1:

          categoryColor = categoryColors[previousLabel]
          if categoryColor not in categoriesLabelled:
            labelLegend = 'sequence %s' % previousLabel
            categoriesLabelled.append(categoryColor)
          else:
            labelLegend = None

          end = labelCount
          plt.axvspan(start, end, facecolor=categoryColor, alpha=0.5,
                      label=labelLegend)
          start = end
          previousLabel = label

        labelCount += 1

      plt.xlim(xmin=0, xmax=len(timesteps))

      csvName = expSetup['filePath'].split('/')[-1][:-4]
      title = 'signal=%s' % ',  '.join(csvName.split('_'))
      plt.title(title)

      plt.legend()

  print t
  plt.show()
