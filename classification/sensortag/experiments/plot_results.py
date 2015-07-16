__author__ = 'mleborgne'

import matplotlib.pyplot as plt
import csv
import os

from settings import METRICS, SENSORS, PATIENT_IDS, ANOMALY_LIKELIHOOD_THRESHOLD, MODEL_RESULTS_DIR, PLOT_RESULTS_DIR

for patient in PATIENT_IDS:
  for sensor in SENSORS:
    fig = plt.figure(figsize=(40, 10))
    plot_index = 1
    for metric in METRICS:

      fileName = "%s/%s_%s_%s_out.csv" % (MODEL_RESULTS_DIR, metric, sensor, patient)
      inputFile = open(fileName, "rb")
      csvReader = csv.reader(inputFile)
      csvReader.next()  # skip header row

      t = []
      metric_values = []
      for row in csvReader:
        timestep = int(row[0])
        t.append(timestep)
        metric_values.append(row[1])
        anomaly_likelyhood = row[4]
        if float(anomaly_likelyhood) > ANOMALY_LIKELIHOOD_THRESHOLD:
          # print "Anomalous datapoint for %s(%s). Anomaly Likelihood = %s" %(sensor, metric, anomaly_likelyhood)
          fig.add_subplot(3, 1, plot_index)
          plt.axvspan(timestep, timestep, color='red', alpha=0.5)

      fig.add_subplot(3, 1, plot_index)
      plt.plot(t, metric_values)
      plt.title("Sensor: %s  |  Metric: %s  |  Patient ID: %s   |   Anomaly Likelyhood Threshold: %s" %
                (sensor, metric, patient, ANOMALY_LIKELIHOOD_THRESHOLD))
      plot_index += 1

    plt.savefig('%s/%s_%s_%s.png' % (PLOT_RESULTS_DIR,  patient, sensor, ANOMALY_LIKELIHOOD_THRESHOLD))
