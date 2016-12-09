# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import copy
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

from htmresearch.frameworks.classification.utils.traces import loadTraces

from clustering import PerfectClustering, OnlineClusteringV2
from clustering_interface import Point
from distances import euclidian_distance
from utils import (clustering_stats, moving_average, get_file_name,
                   convert_to_sdrs)
from plot import (plot_accuracy, plot_cluster_assignments,
                  plot_inter_sequence_distances)



def run(sdrs,
        categories,
        anomaly_scores,
        distance_func,
        moving_average_window,
        max_num_clusters,
        merge_threshold,
        ClusteringClass,
        cluster_snapshot_indices):
  num_sdrs = len(sdrs)
  model = ClusteringClass(distance_func)

  clusters_snapshots = []
  closest_cluster_history = []
  clustering_accuracies = []
  num_correct = 0
  clustering_accuracy = 0
  last_category = int(categories[0])
  new_cluster = model.create_cluster()
  for i in range(num_sdrs):
    sdr = sdrs[i]
    actual_category = int(categories[i])
    point = Point(sdr, actual_category)

    # Create a new cluster.
    # Note: the anomaly score could be used to create new clusters, but for 
    # now we'll use an artificial signal (i.e. new category)
    # anomaly_score = anomaly_scores[i]
    if last_category != actual_category:
      if merge_threshold is None:
        merge_threshold = model.average_cluster_distance() / 2.0
      model.add_or_merge_cluster(new_cluster, merge_threshold)
      new_cluster = model.create_cluster()

    # Inference.
    new_cluster.add(point)
    distance_to_closest, closest_cluster = model.infer(new_cluster.center)
    closest_cluster_history.append(closest_cluster)
    if i in cluster_snapshot_indices:
      clusters_snapshots.append([copy.deepcopy(c)
                                 for c in model.clusters.values()])

    clustering_accuracy = clustering_stats(i,
                                           model.clusters,
                                           closest_cluster,
                                           actual_category,
                                           num_correct,
                                           clustering_accuracy,
                                           moving_average_window)
    clustering_accuracies.append(clustering_accuracy)
    last_category = actual_category

  return clustering_accuracies, clusters_snapshots, closest_cluster_history



def main():
  distance_functions = [euclidian_distance]
  clustering_classes = [PerfectClustering, OnlineClusteringV2]
  network_config = 'sp=True_tm=True_tp=False_SDRClassifier'
  exp_names = ['binary_ampl=10.0_mean=0.0_noise=0.0',
               'binary_ampl=10.0_mean=0.0_noise=1.0',
               'sensortag_z']

  # Exp params
  moving_average_window = 2  # for all moving averages of the experiment
  ClusteringClass = clustering_classes[1]
  distance_func = distance_functions[0]
  exp_name = exp_names[2]
  start_idx = -700
  end_idx = -200
  input_width = 2048 * 32
  active_cells_weight = 0
  predicted_active_cells_weight = 1
  max_num_clusters = 3
  merge_threshold = 3.15  # 3.0 or 3.15
  num_cluster_snapshots = 5
  show_plots = True
  distance_matrix_ignore_noise = True  # whether to ignore label 0 (noise)

  # Clean an create output directory for the graphs
  plots_output_dir = 'plots/%s' % exp_name
  if os.path.exists(plots_output_dir):
    shutil.rmtree(plots_output_dir)
  os.makedirs(plots_output_dir)

  # load traces
  file_name = get_file_name(exp_name, network_config)
  traces = loadTraces(file_name)
  num_records = len(traces['sensorValue'])

  # start and end for the x axis of the graphs
  if start_idx < 0:
    start = num_records + start_idx
  else:
    start = start_idx
  if end_idx < 0:
    end = num_records + end_idx
  else:
    end = end_idx
  xlim = [0, end - start]

  # input data
  sensor_values = traces['sensorValue'][start:end]
  categories = traces['actualCategory'][start:end]
  active_cells = traces['tmActiveCells'][start:end]
  predicted_active_cells = traces['tmPredictedActiveCells'][start:end]
  raw_anomaly_scores = traces['rawAnomalyScore'][start:end]
  anomaly_scores = []
  anomaly_score_ma = 0.0
  for raw_anomaly_score in raw_anomaly_scores:
    anomaly_score_ma = moving_average(anomaly_score_ma,
                                      raw_anomaly_score,
                                      moving_average_window)
    anomaly_scores.append(anomaly_score_ma)

  # generate sdrs to cluster
  active_cells_sdrs = convert_to_sdrs(active_cells, input_width)
  predicted_active_cells_sdrs = np.array(
    convert_to_sdrs(predicted_active_cells, input_width))
  sdrs = (float(active_cells_weight) * np.array(active_cells_sdrs) +
          float(predicted_active_cells_weight) * predicted_active_cells_sdrs)

  # list of timesteps specifying when a snapshot of the clusters will be taken
  step = (end - start) / num_cluster_snapshots - 1
  cluster_snapshot_indices = range(step, end - start, step)

  # run clustering
  (clustering_accuracies,
   cluster_snapshots,
   closest_cluster_history) = run(sdrs,
                                  categories,
                                  anomaly_scores,
                                  distance_func,
                                  moving_average_window,
                                  max_num_clusters,
                                  merge_threshold,
                                  ClusteringClass,
                                  cluster_snapshot_indices)
  # cluster_categories = []
  # for c in closest_cluster_history:
  #   if c is not None:
  #     cluster_categories.append(c.label_distribution()[0]['label'])

  # plot cluster assignments over time
  for i in range(num_cluster_snapshots):
    clusters = cluster_snapshots[i]
    snapshot_index = cluster_snapshot_indices[i]
    plot_cluster_assignments(plots_output_dir, clusters, snapshot_index)

    # plot inter-cluster distance matrix
    # plot_id = 'inter-cluster_t=%s' % snapshot_index
    # plot_inter_sequence_distances(plots_output_dir,
    #                               plot_id,
    #                               distance_func,
    #                               sdrs[:snapshot_index],
    #                               cluster_categories[:snapshot_index],
    #                               distance_matrix_ignore_noise)

    # plot inter-category distance matrix
    plot_id = 'inter-category_t=%s ' % snapshot_index
    plot_inter_sequence_distances(plots_output_dir,
                                  plot_id,
                                  distance_func,
                                  sdrs[:snapshot_index],
                                  categories[:snapshot_index],
                                  distance_matrix_ignore_noise)

  # plot clustering accuracy over time
  plot_id = 'file=%s | moving_average_window=%s' % (exp_name,
                                                    moving_average_window)
  plot_accuracy(plots_output_dir,
                plot_id,
                sensor_values,
                categories,
                anomaly_scores,
                clustering_accuracies,
                xlim)

  if show_plots:
    plt.show()



if __name__ == "__main__":
  main()
