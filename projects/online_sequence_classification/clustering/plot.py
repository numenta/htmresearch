import numpy as np
from matplotlib import pyplot as plt

from utils import (find_cluster_assignments,
                   cluster_distance_matrix)



def plot_inter_sequence_distances(output_dir,
                                  plot_id,
                                  distance_func,
                                  sdrs,
                                  cluster_ids,
                                  ignore_noise):
  cluster_assignments, sdr_slices = find_cluster_assignments(sdrs, cluster_ids,
                                                             ignore_noise)

  distance_mat = cluster_distance_matrix(sdr_slices, distance_func)

  title = 'distance_matrix_%s' % plot_id
  outputFile = '%s/%s' % (output_dir, '%s.png' % title)
  plot_distance_mat(distance_mat, title, outputFile)



def plot_distance_mat(distance_mat, title, output_file):
  plt.figure()
  plt.imshow(distance_mat, interpolation="nearest")
  plt.colorbar()
  plt.title(title)
  plt.xlabel('Sequence category')
  plt.ylabel('Sequence category')
  plt.savefig(output_file)
  print('==> saved: %s' % output_file)
  plt.draw()



def plot_accuracy(output_dir,
                  plot_id,
                  sensor_values,
                  categories,
                  anomaly_scores,
                  clustering_accuracies,
                  xlim):
  fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(15, 7))

  # plot sensor data and categories
  t = range(xlim[0], xlim[1])
  ax[0].plot(t, sensor_values)
  ax[0].set_xlabel('Time step')
  ax[0].set_ylabel('Signal amplitude')
  ax[0].set_xlim(xmin=xlim[0], xmax=xlim[1])
  category_colors = ['grey', 'blue', 'yellow', 'red', 'green', 'orange']
  previous_category = categories[0]
  start = 0
  category_count = 0
  num_points = len(categories)
  categories_labelled = []
  for category in categories:
    if previous_category != category or category_count == num_points - 1:

      category_color = category_colors[int(previous_category)]
      if category_color not in categories_labelled:
        labelLegend = 'class=%s' % int(previous_category)
        categories_labelled.append(category_color)
      else:
        labelLegend = None

      end = category_count
      ax[0].axvspan(start, end, facecolor=category_color, alpha=0.4,
                    label=labelLegend)
      ax[1].axvspan(start, end, facecolor=category_color, alpha=0.4)
      ax[2].axvspan(start, end, facecolor=category_color, alpha=0.4)
      start = end
      previous_category = category

    category_count += 1

  title = 'Sensor data (%s)' % plot_id.split('|')[0]
  ax[0].set_title(title)
  ax[0].set_ylim([-1, 11])
  ax[0].legend(ncol=10)

  # plot anomaly score
  title = 'Anomaly score (%s)' % plot_id
  ax[1].set_title(title)
  ax[1].set_ylim([-0.1, 1.1])
  ax[1].plot(anomaly_scores)
  ax[1].set_ylabel('Anomaly score')

  # clustering accuracy
  title = 'Clustering accuracy (%s)' % plot_id
  ax[2].plot(clustering_accuracies)
  ax[2].set_title(title)
  ax[2].set_ylim([-0.1, 1.1])
  ax[2].set_xlabel('Time step')
  ax[2].set_ylabel('Clustering accuracy')


  plt.tight_layout(pad=0.5)
  fig_name = 'clustering_accuracy.png'
  plt.savefig('%s/%s' % (output_dir, fig_name))
  print('==> saved: %s/%s' % (output_dir, fig_name))
  plt.draw()



def plot_cluster_assignments(output_dir, clusters, timestep):
  fig, ax = plt.subplots(figsize=(15, 7))
  # cluster sizes
  num_clusters = len(clusters)
  categories_to_num_points = {}
  for i in range(num_clusters):
    cluster = clusters[i]
    cluster_id = cluster.id
    freqs = cluster.label_distribution()
    for freq in freqs:
      num_points = int(freq['num_points'])
      category = int(freq['label'])
      if category not in categories_to_num_points:
        categories_to_num_points[category] = {}
      categories_to_num_points[category][cluster_id] = num_points

  cluster_ids = []
  for clusters_to_num_points in categories_to_num_points.values():
    cluster_ids.extend(clusters_to_num_points.keys())
  cluster_ids = list(set(cluster_ids))

  # Get some pastel shades for the colors. Note: category index start at 0 
  num_bars = len(cluster_ids)
  num_categories = max(categories_to_num_points.keys()) + 1
  colors = plt.cm.BuPu(np.linspace(0, 0.5, num_categories))
  bottom = np.array([0 for _ in range(num_bars)])
  # Plot bars and create text labels for the table
  cell_text = []
  categories = []
  for category, clusters_to_num_points in categories_to_num_points.items():
    categories.append(category)
    bars = []
    for cid in cluster_ids:
      if cid in clusters_to_num_points:
        bars.append(clusters_to_num_points[cid])
      else:
        bars.append(0)

    # draw the bars for this category
    x = np.array([i for i in range(num_bars)])
    ax.bar(x,
           bars,
           align='center',
           bottom=bottom,
           color=colors[category])
    bottom += np.array(bars)
    ax.set_xticks(x)
    cell_text.append([x for x in bars])

  ax.set_title('Number of points per category by cluster ID (Timestep: %s)'
               % timestep)
  ax.set_ylabel('Number of points')

  # Reverse colors and text labels to display the last value at the top.
  colors = colors[::-1]
  cell_text.reverse()

  # Add a table at the bottom of the axes
  rowLabels = ['category %s' % c for c in categories]
  colLabels = ['c%s' % c for c in cluster_ids]
  the_table = plt.table(cellText=cell_text,
                        cellLoc='center',
                        rowLabels=rowLabels,
                        rowColours=colors,
                        colLabels=colLabels,
                        loc='bottom')
  the_table.auto_set_font_size(False)
  the_table.set_fontsize(9)
  the_table.scale(1, 2)
  ax.set_xticks([])
  plt.tight_layout(pad=7)
  fig_name = 'cluster_assignments_t=%s.png' % timestep
  plt.savefig('%s/%s' % (output_dir, fig_name))
  print('==> saved: %s/%s' % (output_dir, fig_name))
  plt.draw()
