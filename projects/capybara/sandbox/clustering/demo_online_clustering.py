import random
import time

import numpy as np
import scipy
from htmresearch.frameworks.clustering.distances import kernel_dist
from htmresearch.frameworks.clustering.online_agglomerative_clustering \
  import OnlineAgglomerativeClustering
from matplotlib import pyplot as plt

from htmresearch.frameworks.capybara.unsupervised.kernels \
  import normalized_gaussian_kernel



def generate_points(num_classes, points_per_class, noise, dim):
  points = []
  labels = []
  # create three random 2D gaussian clusters
  for i in range(num_classes):
    center = [i for _ in range(dim)]
    for _ in range(points_per_class):
      point = scipy.array([center[k] + random.normalvariate(0, noise)
                           for k in range(dim)])
      points.append(point)
      labels.append(i)

  # shuffle
  shuf_indices = range(len(points))
  random.shuffle(shuf_indices)
  points = [points[i] for i in shuf_indices]
  labels = [labels[i] for i in shuf_indices]

  return points, labels



def cluster_category_frequencies(cluster):
  labels = []
  for point in cluster.points:
    labels.append(point['label'])

  unique, counts = np.unique(labels, return_counts=True)
  frequencies = []
  for actualCategory, numberOfPoints in np.asarray((unique, counts)).T:
    frequencies.append({
      'actual_category': actualCategory,
      'num_points': numberOfPoints
    })

  return frequencies



def moving_average(last_ma, new_point_value, rolling_window_size):
  """
  Online computation of moving average.
  From: http://www.daycounter.com/LabBook/Moving-Average.phtml
  """

  ma = last_ma + (new_point_value - last_ma) / float(rolling_window_size)
  return ma



def run(max_num_clusters,
        cluster_size_cutoff,
        trim_clusters,
        plot,
        rolling_window,
        distance_func,
        points,
        labels):
  num_points = len(points)
  start = time.time()
  model = OnlineAgglomerativeClustering(max_num_clusters, distance_func,
                                        cluster_size_cutoff)
  if plot:
    plt.ion()  # interactive mode on
    last_cx = []
    last_cy = []
    last_winning_cx = []
    last_winning_cy = []

  num_correct = 0
  tot_num_points = 0
  category_predictions = {i: 0 for i in range(max_num_clusters)}
  accuracy_ma = 0
  accuracy_mas = []
  for i in range(num_points):
    point = points[i]
    actual_label = labels[i]
    winning_clusters, closest = model.cluster(point, trim_clusters,
                                              actual_label)
    if closest:
      # info about predicted cluster
      cluster_id = closest.id
      cluster_size = closest.size
      freqs = cluster_category_frequencies(closest)
      cluster_category = freqs[0]['actual_category']
      category_predictions[cluster_category] += 1

      # compute accuracy      
      if cluster_category == actual_label:
        accuracy = 1
      else:
        accuracy = 0
      tot_num_points += 1
      num_correct += accuracy
      accuracy_ma = moving_average(accuracy_ma, accuracy, rolling_window)
      accuracy_mas.append(accuracy_ma)

    else:
      cluster_id = 'NA'
      cluster_size = None
      cluster_category = None
    clusters = model._clusters

    winning_cluster_ids = [c.id for c in winning_clusters]
    cluster_ids = [c.id for c in clusters]

    print("%s winning clusters: %s | %s clusters: %s | "
          "closest: {id=%s, size=%s, category=%s} | Actual category: %s"
          % (len(winning_clusters), winning_cluster_ids, len(clusters),
             cluster_ids, cluster_id, cluster_size, cluster_category,
             actual_label))

    if plot:
      plt.plot(point[0], point[1], 'bo')

      winning_cx = [x.center[0] for x in winning_clusters]
      winning_cy = [y.center[1] for y in winning_clusters]

      cx = [x.center[0] for x in clusters]
      cy = [y.center[1] for y in clusters]

      plt.plot(last_cx, last_cy, "bo")
      plt.plot(cx, cy, "yo")
      plt.plot(last_winning_cx, last_winning_cy, "bo")
      plt.plot(winning_cx, winning_cy, "ro")
      plt.pause(0.0001)

      last_cx = cx
      last_cy = cy

  print 'clustering accuracy = %s' % (num_correct / float(tot_num_points))
  print('category predictions: %s' % category_predictions)
  print ("%d points clustered in %.2f s." % (num_points, time.time() - start))

  # plot results
  fig, ax = plt.subplots(nrows=2, figsize=(8, 6))
  # clustering accuracy
  ax[0].plot(accuracy_mas)
  ax[0].set_title('Clustering Accuracy Moving Average (Window = %s)'
                  % rolling_window)
  ax[0].set_xlabel('Time step')
  ax[0].set_ylabel('Accuracy MA')
  # prediction frequency
  ax[1].bar(category_predictions.keys(), category_predictions.values())
  ax[1].set_title('Clustering Prediction Frequencies')
  ax[1].set_xlabel('Category')
  ax[1].set_ylabel('Frequency of prediction')
  plt.tight_layout()
  plt.savefig('clustering_accuracy.png')



def demo_gaussian_noise():
  # the value of N is generally quite forgiving, i.e.
  # giving 6 will still only find the 3 clusters.
  # around 10 it will start finding more
  max_num_clusters = 6
  cluster_size_cutoff = 0.5
  trim_clusters = True
  plot = True
  rolling_window = 10
  distance_func = kernel_dist(normalized_gaussian_kernel)

  # 2D data
  num_classes = 4
  dim = 2

  num_points_per_class = 200
  noise_level = 0.1
  points, labels = generate_points(num_classes,
                                   num_points_per_class,
                                   noise_level,
                                   dim)
  # We can't plot in 2D for now if dim > 2
  if dim > 2:
    plot = False
  run(max_num_clusters,
      cluster_size_cutoff,
      trim_clusters,
      plot,
      rolling_window,
      distance_func,
      points,
      labels)



if __name__ == "__main__":
  demo_gaussian_noise()
