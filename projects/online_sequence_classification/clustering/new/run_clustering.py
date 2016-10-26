import numpy as np
from matplotlib import pyplot as plt

from htmresearch.frameworks.classification.utils.traces import loadTraces
from utils import (clustering_stats, moving_average, get_file_name,
                   convert_to_sdrs)
from plot import plot_accuracy, plot_clustering_results
from clustering import PerfectClustering as Clustering
from online_clustering import OnlineClustering as Clustering
from distances import euclidian as distance



def run(points,
        labels,
        rolling_window,
        max_num_clusters):
  num_points = len(points)
  model = Clustering(max_num_clusters, distance)

  clusters_history = []
  accuracy_ma_history = []
  num_correct = 0
  accuracy_ma = 0
  for i in range(num_points):
    point = points[i]
    actual_label = labels[i]
    closest = model.cluster(point, actual_label)
    clusters_history.append(model.clusters)

    accuracy_ma = clustering_stats(i,
                                   model.clusters,
                                   closest,
                                   actual_label,
                                   num_correct,
                                   accuracy_ma,
                                   rolling_window)
    accuracy_ma_history.append(accuracy_ma)

  return accuracy_ma_history, clusters_history



def main():
  network_config = 'sp=True_tm=True_tp=False_SDRClassifier'
  exp_names = ['binary_ampl=10.0_mean=0.0_noise=0.0',
               'binary_ampl=10.0_mean=0.0_noise=1.0',
               'sensortag_z']

  # exp params
  exp_name = exp_names[0]
  anomaly_score_type = 'rawAnomalyScore'
  start_idx = 0
  end_idx = -1
  rolling_window = 10
  input_width = 2048 * 32
  active_cells_weight = 0
  predicted_active_cells_weight = 1
  max_num_clusters = 3
  cluster_assignments_slices = 4

  # load traces
  file_name = get_file_name(exp_name, network_config)
  traces = loadTraces(file_name)
  input_data = traces['sensorValue'][start_idx:end_idx]
  labels = traces['actualCategory'][start_idx:end_idx]
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

  # generate points to cluster
  active_cells_sdrs = convert_to_sdrs(active_cells, input_width)
  predicted_activeCells_sdrs = np.array(convert_to_sdrs(predicted_active_cells,
                                                        input_width))
  points = (active_cells_weight * np.array(active_cells_sdrs) +
            predicted_active_cells_weight * predicted_activeCells_sdrs)

  # run clustering
  (accuracy_ma_history, clusters_history) = run(points,
                                                labels,
                                                rolling_window,
                                                max_num_clusters)

  # plot cluster assignments over time
  timesteps = range(0, len(points), len(points) / cluster_assignments_slices)
  for timestep in timesteps:
    last_clusters = clusters_history[timestep]
    plot_clustering_results(last_clusters, timestep)

  # plot clustering accuracy over time
  start = start_idx
  if end_idx < 0:
    end = len(points) - end_idx - 1
  else:
    end = end_idx
  xlim = [start, end]
  plot_accuracy(accuracy_ma_history,
                rolling_window,
                input_data,
                labels,
                anomaly_scores_mas,
                exp_name,
                anomaly_score_type,
                xlim)
  plt.show()



if __name__ == "__main__":
  main()
