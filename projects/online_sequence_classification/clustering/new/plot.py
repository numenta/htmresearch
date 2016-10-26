import numpy as np
from matplotlib import pyplot as plt

from utils import cluster_category_frequencies



def plot_accuracy(accuracy_moving_averages,
                  rolling_window,
                  points,
                  labels,
                  anomalyScores,
                  title,
                  anomalyScoreType,
                  xlim):
  fig, ax = plt.subplots(nrows=3, sharex=True)
  t = range(xlim[0], xlim[1])
  ax[0].plot(t, points, label='signal')
  ax[0].set_xlabel('Time step')
  ax[0].set_ylabel('Signal amplitude')
  ax[0].set_xlim(xmin=xlim[0], xmax=xlim[1])
  categoryColors = ['grey', 'blue', 'yellow', 'red', 'green', 'orange']
  previousLabel = labels[0]
  start = 0
  labelCount = 0
  numPoints = len(labels)
  categoriesLabelled = []
  for label in labels:
    if previousLabel != label or labelCount == numPoints - 1:

      categoryColor = categoryColors[int(previousLabel)]
      if categoryColor not in categoriesLabelled:
        labelLegend = 'Cat. %s' % int(previousLabel)
        categoriesLabelled.append(categoryColor)
      else:
        labelLegend = None

      end = labelCount
      ax[0].axvspan(start, end, facecolor=categoryColor, alpha=0.4,
                    label=labelLegend)
      start = end
      previousLabel = label

    labelCount += 1

  ax[0].set_title(title)
  ax[0].legend(ncol=4)

  # clustering accuracy
  ax[1].plot(accuracy_moving_averages)
  ax[1].set_title('Clustering Accuracy Moving Average (Window = %s)'
                  % rolling_window)
  ax[1].set_xlabel('Time step')
  ax[1].set_ylabel('Accuracy MA')

  # plot anomaly score
  ax[2].set_title(anomalyScoreType)
  ax[2].plot(anomalyScores)

  plt.tight_layout(pad=0.5)
  fig_name = 'clustering_accuracy.png'
  plt.savefig(fig_name)
  print('==> saved: %s' % fig_name)
  plt.draw()



def plot_clustering_results(clusters, timestep):
  fig, ax = plt.subplots()
  # cluster sizes
  num_clusters = len(clusters)
  categories_to_num_points = {}
  for i in range(num_clusters):
    cluster = clusters[i]
    cluster_id = cluster.id
    freqs = cluster_category_frequencies(cluster)
    for freq in freqs:
      num_points = int(freq['num_points'])
      category = int(freq['actual_category'])
      if category not in categories_to_num_points:
        categories_to_num_points[category] = {}
      categories_to_num_points[category][cluster_id] = num_points

  cluster_ids = []
  for clusters_to_num_points in categories_to_num_points.values():
    cluster_ids.extend(clusters_to_num_points.keys())
  cluster_ids = list(set(cluster_ids))

  # Get some pastel shades for the colors. Note: category index start at 0 
  num_categories = max(categories_to_num_points.keys()) + 1
  colors = plt.cm.BuPu(np.linspace(0, 0.5, num_categories))
  bottom = np.array([0 for _ in range(len(cluster_ids) + 1)])
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
    bars.append(sum(bars))

    # draw the bars for this category
    bar_width = 0.3
    ax.bar(np.array([i for i in range(len(cluster_ids) + 1)]) + 0.3,
           bars,
           bar_width,
           bottom=bottom,
           color=colors[category])
    bottom += np.array(bars)
    cell_text.append([x for x in bars])

  ax.set_title('Number of points per category by cluster ID (Timestep: %s)'
               % timestep)
  ax.set_ylabel('Number of points')

  # Reverse colors and text labels to display the last value at the top.
  colors = colors[::-1]
  cell_text.reverse()

  # Add a table at the bottom of the axes
  rowLabels = ['category %s' % c for c in categories]
  colLabels = ['cluster %s' % c for c in cluster_ids]
  colLabels.append('Tot. pts')
  the_table = plt.table(cellText=cell_text,
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
  plt.savefig(fig_name)
  print('==> saved: %s' % fig_name)
  plt.draw()
