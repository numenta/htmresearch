# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

FONT_SIZE = 12
FIG_SIZE = (12, 10)



def resolve_plot_path(plot_dir, data_id, cell_type, nb_chunks, aggregation):
  plot_name = '%s_%s_chunks=%s_agg=%s.png' % (data_id, cell_type, nb_chunks,
                                      aggregation)
  plot_path = os.path.join(plot_dir, plot_name)
  return plot_path



def make_sup_title(data_id, cell_type, nb_chunks, aggregation):
  return ('Data: %s | Cells: %s | Chunks: %s | Aggregation: %s'
          % (data_id, cell_type.upper(), nb_chunks, aggregation))



def make_subplots(n_rows, n_cols, plot_dir, data_id, cell_type, nb_chunks,
                  agg):
  plot_path = resolve_plot_path(plot_dir, data_id, cell_type, nb_chunks, agg)
  sup_title = make_sup_title(data_id, cell_type, nb_chunks, agg)
  fig, ax = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE)
  fig.suptitle(sup_title, fontsize=FONT_SIZE + 2, fontweight='bold')
  fig.subplots_adjust(hspace=.5, wspace=.5)

  return fig, ax, plot_path



def plot_matrix(embeddings_mat, title, fig, ax):
  heatmap = ax.pcolor(embeddings_mat, cmap=plt.cm.Blues)
  fig.colorbar(heatmap, ax=ax)
  n_sequences = embeddings_mat.shape[0]
  ticks = range(1, n_sequences + 1, n_sequences / 5)
  if n_sequences not in ticks: ticks.append(n_sequences)
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  ax.set_xlabel('Sequence #')
  ax.set_ylabel('Sequence #')
  ax.set_title(title, fontsize=FONT_SIZE)



def plot_projections(embeddings_proj, labels, title, fig, ax):
  # Colors
  unique_labels = list(set(list(labels)))
  nb_colors = len(unique_labels)
  color_names = ['Class %s' % l for l in unique_labels]
  colors = sns.color_palette('colorblind', nb_colors)

  # Plot projections
  ax.set_title(title, fontsize=FONT_SIZE)
  ax.scatter(embeddings_proj[:, 0], embeddings_proj[:, 1],
             c=[colors[unique_labels.index(l)] for l in labels])

  # Add legend
  patches = [mpatches.Patch(color=colors[i], label=color_names[i])
             for i in range(nb_colors)]
  ax.legend(handles=patches, loc='best')



def make_plot_title(plot_name, phase, accuracy):
  return ('%s\n%s accuracy: %s / 100\n'
          % (plot_name.capitalize(), phase.capitalize(), accuracy))
