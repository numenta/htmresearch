import csv
import os
import scipy
import numpy as np
from sklearn import manifold
from distances import cluster_distance_factory



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



def convert_to_sdr(patternNZ, input_width):
  sdr = np.zeros(input_width)
  sdr[np.array(patternNZ, dtype='int')] = 1
  return sdr



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



def get_num_clusters(cluster_ids):
  """
  Return the number of actual clusters.
  Note that noise is labelled as category 0, but that there might not be 
  noise in this dataset.

  :param cluster_ids: (list) cluster IDs
  :return num_clusters: (int) number of unique clusters 
  """
  unique_clusters = np.unique(cluster_ids)
  num_categories = len(unique_clusters)
  if 0 not in unique_clusters:
    num_categories += 1

  return num_categories



def find_cluster_repetitions(sdrs, cluster_ids):
  """
  Find how many times a cluster IDs is assigned to a consecutive sequence of 
  points.
  :param sdrs: (list of np.array)
  :param cluster_ids: (list) cluster IDs
  :return cluster_repetitions: (list) repetition count of consecutive sequence 
    of points with the same cluster ID.
  :return sdr_clusters: (dict of list) keys are the cluster IDs. Values are 
    the SDRs in the cluster.
  """
  num_clusters = get_num_clusters(cluster_ids)
  repetition_counter = np.zeros((num_clusters,))
  last_category = None
  cluster_repetitions = []
  sdr_clusters = {i: [] for i in range(num_clusters)}
  for i in range(len(cluster_ids)):
    category = int(cluster_ids[i])
    if category != last_category:
      repetition_counter[category] += 1
    last_category = category
    cluster_repetitions.append(repetition_counter[category] - 1)
    sdr_clusters[category].append(sdrs[i])

  assert len(cluster_ids) == sum([len(sdr_clusters[i])
                                  for i in range(num_clusters)])

  return cluster_repetitions, sdr_clusters



def find_cluster_assignments(sdrs, cluster_ids, ignore_noise):
  """
  Compare inter-cluster distance over time.
  :param sdrs: (list of np.array) list of sdr
  :param cluster_ids: (list) name of the cluster field in the traces dict
  :param ignore_noise: (bool) whether to take noise (label=0) into account
  :return: 
  """

  cluster_reps, sdr_clusters = find_cluster_repetitions(sdrs, cluster_ids)

  num_reps_per_category = {}
  categories = list(np.unique(cluster_ids))
  repetition = np.array(cluster_reps)
  for category in categories:
    num_reps_per_category[category] = np.max(
      repetition[np.array(cluster_ids) == category])

  sdr_slices = []
  cluster_assignments = []
  min_num_reps = np.min(num_reps_per_category.values()).astype('int32')
  for rpt in range(min_num_reps + 1):
    idx0 = np.logical_and(np.array(cluster_ids) == 0, repetition == rpt)
    idx1 = np.logical_and(np.array(cluster_ids) == 1, repetition == rpt)
    idx2 = np.logical_and(np.array(cluster_ids) == 2, repetition == rpt)

    c0slice = [sdrs[i] for i in range(len(idx0)) if idx0[i]]
    c1slice = [sdrs[i] for i in range(len(idx1)) if idx1[i]]
    c2slice = [sdrs[i] for i in range(len(idx2)) if idx2[i]]

    if not ignore_noise:
      sdr_slices.append(c0slice)
      cluster_assignments.append(0)
    sdr_slices.append(c1slice)
    cluster_assignments.append(1)
    sdr_slices.append(c2slice)
    cluster_assignments.append(2)

  return cluster_assignments, sdr_slices



def cluster_distance_matrix(sdr_clusters, distance_func):
  """
  Compute distance matrix between clusters of SDRs
  :param sdr_clusters: list of sdr clusters. Each cluster is a list of SDRs.
  :return: distance matrix
  """

  cluster_dist = cluster_distance_factory(distance_func)

  num_clusters = len(sdr_clusters)
  distance_mat = np.zeros((num_clusters, num_clusters), dtype=np.float64)

  for i in range(num_clusters):
    for j in range(i, num_clusters):
      distance_mat[i, j] = cluster_dist(sdr_clusters[i],
                                        sdr_clusters[j])
      distance_mat[j, i] = distance_mat[i, j]

  return distance_mat



def project_clusters_2D(distance_mat):
  seed = np.random.RandomState(seed=3)

  mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                     random_state=seed,
                     dissimilarity="precomputed", n_jobs=1)

  pos = mds.fit(distance_mat).embedding_

  nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                      dissimilarity="precomputed", random_state=seed, n_jobs=1,
                      n_init=1)

  npos = nmds.fit_transform(distance_mat, init=pos)

  return npos
