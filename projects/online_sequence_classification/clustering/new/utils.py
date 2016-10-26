import csv
import os
import scipy
import numpy as np



def cluster_category_frequencies(cluster):
  """
  Returns frequency of each category in this cluster. E.g:
  [
    {
      'actual_category': 1.0,
      'num_points': 20
    },
       ...
    {
      'actual_category': 5.0,
      'num_points': 30
    }   
  ]
  """
  labels = []
  for point in cluster.points:
    labels.append(point['label'])

  unique, counts = np.unique(labels, return_counts=True)
  category_frequencies = []
  for actualCategory, numberOfPoints in np.asarray((unique, counts)).T:
    category_frequencies.append({
      'actual_category': actualCategory,
      'num_points': numberOfPoints
    })

  return category_frequencies



def moving_average(last_ma, new_point_value, rolling_window_size):
  """
  Online computation of moving average.
  From: http://www.daycounter.com/LabBook/Moving-Average.phtml
  """

  ma = last_ma + (new_point_value - last_ma) / float(rolling_window_size)
  return ma



def clustering_stats(record_number,
                     clusters,
                     closest,
                     actual_label,
                     num_correct,
                     accuracy_ma,
                     rolling_window):
  if closest:
    # info about predicted cluster
    category_frequencies = cluster_category_frequencies(closest)
    cluster_category = category_frequencies[0]['actual_category']

    # compute accuracy      
    if cluster_category == actual_label:
      accuracy = 1
    else:
      accuracy = 0
    num_correct += accuracy
    accuracy_ma = moving_average(accuracy_ma, accuracy, rolling_window)
    cluster_id = closest.id
    cluster_size = closest.size

    print("Record: %s | Accuracy MA: %s | Total clusters: %s | "
          "Closest: {id=%s, size=%s, category=%s} | Actual category: %s"
          % (record_number, accuracy_ma, len(clusters), cluster_id,
             cluster_size, cluster_category, actual_label))

  return accuracy_ma



def get_file_name(exp_name, network_config):
  trace_csv = 'traces_%s_%s.csv' % (exp_name, network_config)
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      os.pardir, os.pardir, 'classification', 'results',
                      trace_csv)



def convert_to_sdrs(patterNZs, input_width):
  sdrs = []
  for i in range(len(patterNZs)):
    patternNZ = patterNZs[i]
    sdr = np.zeros(input_width)
    sdr[patternNZ] = 1
    sdrs.append(sdr)
  return sdrs



def load_csv(input_file):
  with open(input_file, 'r') as f:
    reader = csv.reader(f)
    headers = reader.next()
    points = []
    labels = []
    for row in reader:
      dict_row = dict(zip(headers, row))
      points.append(scipy.array([float(dict_row['x']),
                                 float(dict_row['y'])]))
      labels.append(int(dict_row['label']))

    return points, labels
