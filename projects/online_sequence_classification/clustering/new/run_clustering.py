import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

from htmresearch.frameworks.classification.utils.traces import loadTraces
from utils import (clustering_stats, moving_average, get_file_name,
                   convert_to_sdrs)
from plot import (plot_accuracy, plot_cluster_assignments,
                  plot_inter_sequence_distances)
from clustering import PerfectClustering
from online_clustering import OnlineClustering
from distances import euclidian



def run(points,
        labels,
        distance_func,
        rolling_window,
        max_num_clusters,
        clustering_class):
  num_points = len(points)
  model = clustering_class(max_num_clusters, distance_func)

  clusters_history = []
  closest_cluster_history = []
  accuracy_ma_history = []
  num_correct = 0
  accuracy_ma = 0
  for i in range(num_points):
    point = points[i]
    actual_label = labels[i]
    closest = model.cluster(point, actual_label)
    closest_cluster_history.append(closest)
    clusters_history.append(model.clusters)

    accuracy_ma = clustering_stats(i,
                                   model.clusters,
                                   closest,
                                   actual_label,
                                   num_correct,
                                   accuracy_ma,
                                   rolling_window)
    accuracy_ma_history.append(accuracy_ma)

  return accuracy_ma_history, clusters_history, closest_cluster_history



def main():
  clustering_classes = [PerfectClustering, OnlineClustering]
  network_config = 'sp=True_tm=True_tp=False_SDRClassifier'
  exp_names = ['binary_ampl=10.0_mean=0.0_noise=0.0',
               'binary_ampl=10.0_mean=0.0_noise=1.0',
               'sensortag_z']

  # Exp params
  clustering_class = clustering_classes[0]
  ignore_noise = True
  distance_func = euclidian
  exp_name = exp_names[0]
  anomaly_score_type = 'rawAnomalyScore'
  start_idx = 0
  end_idx = -1
  rolling_window = 10
  input_width = 2048 * 32
  active_cells_weight = 0
  predicted_active_cells_weight = 1
  max_num_clusters = 3
  cluster_assignments_timestep_slices = 1

  # Clean an create output directory
  output_dir = exp_name
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  # load traces
  file_name = get_file_name(exp_name, network_config)
  traces = loadTraces(file_name)
  input_data = traces['sensorValue'][start_idx:end_idx]
  categories = traces['actualCategory'][start_idx:end_idx]
  anomaly_scores = traces[anomaly_score_type][start_idx:end_idx]
  anomaly_scores_mas = []
  anomaly_scores_ma = 0.0
  for anomaly_score in anomaly_scores:
    anomaly_scores_ma = moving_average(anomaly_scores_ma,
                                       anomaly_score,
                                       rolling_window)
    anomaly_scores_mas.append(anomaly_scores_ma)

  active_cells = traces['tmActiveCells'][start_idx:end_idx]
  predicted_active_cells = traces['tmPredictedActiveCells'][start_idx:end_idx]

  # generate sdrs to cluster
  active_cells_sdrs = convert_to_sdrs(active_cells, input_width)
  predicted_activeCells_sdrs = np.array(convert_to_sdrs(predicted_active_cells,
                                                        input_width))
  sdrs = (active_cells_weight * np.array(active_cells_sdrs) +
          predicted_active_cells_weight * predicted_activeCells_sdrs)
  start = start_idx
  if end_idx < 0:
    end = len(sdrs) - end_idx - 1
  else:
    end = end_idx
  xlim = [start, end]

  # run clustering
  (accuracy_ma_history,
   clusters_history,
   closest_cluster_history) = run(sdrs,
                                  categories,
                                  distance_func,
                                  rolling_window,
                                  max_num_clusters,
                                  clustering_class)

  # plot cluster assignments over time
  step = len(sdrs) / cluster_assignments_timestep_slices - 1
  timesteps = range(step, len(sdrs), step)
  for timestep in timesteps:
    clusters = clusters_history[timestep]
    plot_cluster_assignments(output_dir, clusters, timestep)

  # plot inter-cluster distance matrix
  cluster_ids = [c.id for c in closest_cluster_history]
  plot_id = 'inter-cluster'
  plot_inter_sequence_distances(output_dir, plot_id, distance_func, sdrs,
                                cluster_ids, ignore_noise)

  # plot inter-category distance matrix
  plot_id = 'inter-category'
  plot_inter_sequence_distances(output_dir, plot_id, distance_func, sdrs,
                                categories, ignore_noise)

  # plot clustering accuracy over time
  plot_accuracy(output_dir,
                accuracy_ma_history,
                rolling_window,
                input_data,
                categories,
                anomaly_scores_mas,
                exp_name,
                anomaly_score_type,
                xlim)

  plt.show()



if __name__ == "__main__":
  main()
