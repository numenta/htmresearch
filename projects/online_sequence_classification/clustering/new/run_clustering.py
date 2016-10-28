import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

from htmresearch.frameworks.classification.utils.traces import loadTraces

from clustering import PerfectClustering
from online_clustering import OnlineClustering
from distances import euclidian
from utils import (clustering_stats, moving_average, get_file_name,
                   convert_to_sdrs, copy_clusters)
from plot import (plot_accuracy, plot_cluster_assignments,
                  plot_inter_sequence_distances)



def run(points,
        categories,
        distance_func,
        moving_average_window,
        max_num_clusters,
        ClusteringClass,
        cluster_snapshot_indices):
  num_points = len(points)
  model = ClusteringClass(max_num_clusters, distance_func)

  clusters_snapshots = []
  closest_cluster_history = []
  clustering_accuracies = []
  num_correct = 0
  clustering_accuracy = 0
  for i in range(num_points):
    point = points[i]
    actual_category = categories[i]
    closest = model.cluster(point, actual_category)
    closest_cluster_history.append(closest)
    if i in cluster_snapshot_indices:
      clusters_copy = copy_clusters(model.clusters)
      clusters_snapshots.append(clusters_copy)

    clustering_accuracy = clustering_stats(i,
                                           model.clusters,
                                           closest,
                                           actual_category,
                                           num_correct,
                                           clustering_accuracy,
                                           moving_average_window)
    clustering_accuracies.append(clustering_accuracy)

  return clustering_accuracies, clusters_snapshots, closest_cluster_history



def main():
  distance_functions = [euclidian]
  clustering_classes = [PerfectClustering, OnlineClustering]
  network_config = 'sp=True_tm=True_tp=False_SDRClassifier'
  exp_names = ['binary_ampl=10.0_mean=0.0_noise=0.0',
               'binary_ampl=10.0_mean=0.0_noise=1.0',
               'sensortag_z']

  # Exp params
  moving_average_window = 10  # for all moving averages of the experiment
  ClusteringClass = clustering_classes[0]
  distance_func = distance_functions[0]
  exp_name = exp_names[0]
  start_idx = 0
  end_idx = -1
  input_width = 2048 * 32
  active_cells_weight = 0
  predicted_active_cells_weight = 1
  ignore_noise = True  # ignore noise when plotting results
  max_num_clusters = 3
  num_cluster_snapshots = 2

  # Clean an create output directory
  output_dir = exp_name
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  # load traces
  file_name = get_file_name(exp_name, network_config)
  traces = loadTraces(file_name)
  sensor_values = traces['sensorValue'][start_idx:end_idx]
  categories = traces['actualCategory'][start_idx:end_idx]
  raw_anomaly_scores = traces['rawAnomalyScore'][start_idx:end_idx]
  anomaly_scores = []
  anomaly_score_ma = 0.0
  for raw_anomaly_score in raw_anomaly_scores:
    anomaly_score_ma = moving_average(anomaly_score_ma,
                                      raw_anomaly_score,
                                      moving_average_window)
    anomaly_scores.append(anomaly_score_ma)

  active_cells = traces['tmActiveCells'][start_idx:end_idx]
  predicted_active_cells = traces['tmPredictedActiveCells'][start_idx:end_idx]

  # generate sdrs to cluster
  active_cells_sdrs = convert_to_sdrs(active_cells, input_width)
  predicted_activeCells_sdrs = np.array(convert_to_sdrs(predicted_active_cells,
                                                        input_width))
  sdrs = (active_cells_weight * np.array(active_cells_sdrs) +
          predicted_active_cells_weight * predicted_activeCells_sdrs)

  # start and end for the x axis of the graphs
  start = start_idx
  if end_idx < 0:
    end = len(sdrs) - end_idx - 1
  else:
    end = end_idx
  xlim = [start, end]

  # list of timesteps specifying when a snapshot of the clusters will be taken
  step = (end - start) / num_cluster_snapshots - 1
  cluster_snapshot_indices = range(start + step, end, step)

  # run clustering
  (clustering_accuracies,
   cluster_snapshots,
   closest_cluster_history) = run(sdrs,
                                  categories,
                                  distance_func,
                                  moving_average_window,
                                  max_num_clusters,
                                  ClusteringClass,
                                  cluster_snapshot_indices)

  # plot cluster assignments over time
  for i in range(num_cluster_snapshots):
    clusters = cluster_snapshots[i]
    plot_cluster_assignments(output_dir, clusters, cluster_snapshot_indices[i])

    # plot inter-cluster distance matrix
    cluster_ids = [c.id for c in closest_cluster_history if c is not None]
    plot_id = 'inter-cluster_t=%s' % cluster_snapshot_indices[i]
    plot_inter_sequence_distances(output_dir, plot_id, distance_func, sdrs,
                                  cluster_ids, ignore_noise)

    # plot inter-category distance matrix
    plot_id = 'inter-category_t=%s ' % cluster_snapshot_indices[i]
    plot_inter_sequence_distances(output_dir,
                                  plot_id,
                                  distance_func,
                                  sdrs[:cluster_snapshot_indices[i]],
                                  categories[:cluster_snapshot_indices[i]],
                                  ignore_noise)

  # plot clustering accuracy over time
  plot_id = 'file=%s | moving_average_window=%s' % (exp_name,
                                                    moving_average_window)
  plot_accuracy(output_dir,
                plot_id,
                sensor_values,
                categories,
                anomaly_scores,
                clustering_accuracies,
                xlim)

  plt.show()



if __name__ == "__main__":
  main()
