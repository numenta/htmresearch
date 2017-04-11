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
import datetime
import os
import time

from htmresearch.frameworks.capybara.distance import \
  distance_matrix, sequence_distance, reshaped_sequence_distance
from htmresearch.frameworks.capybara.embedding import \
  convert_to_embeddings, reshape_embeddings
from htmresearch.frameworks.capybara.proj import project_vectors
from htmresearch.frameworks.capybara.sdr import load_sdrs
from htmresearch.frameworks.capybara.supervised.classification import \
  train_and_test
from htmresearch.frameworks.capybara.supervised.plot import \
  plot_matrix, plot_projections, make_plot_title, make_subplots
from htmresearch.frameworks.capybara.util import \
  get_logger, check_shape, indent, hours_minutes_seconds

PHASES = ['train', 'test']
CELL_TYPES = ['sp', 'tm']
SP_OUT_WIDTH = 2048
TM_OUT_WIDTH = 65536

LOGGER = get_logger()



def analyze_sdr_sequences(sdr_sequences_train, sdr_sequences_test, data_id,
                          nb_chunks, n_neighbors, tsne, aggregation, plot_dir,
                          assume_sequence_alignment):
  sdr_widths = {'sp': SP_OUT_WIDTH, 'tm': TM_OUT_WIDTH}
  accuracies = {cell_type: {} for cell_type in CELL_TYPES}
  dist_mats = {cell_type: {} for cell_type in CELL_TYPES}
  embeddings = {cell_type: {} for cell_type in CELL_TYPES}
  X = {cell_type: {} for cell_type in CELL_TYPES}
  y = {}

  # Step 1: convert the SDR sequences to "sequence embeddings" and compute the
  # pair-wise sequence distances.
  for phase, sdr_sequences in zip(PHASES,
                                  [sdr_sequences_train, sdr_sequences_test]):

    # Sort by label to make it easier to visualize embeddings later.
    sorted_sdr_sequences = sdr_sequences.sort_values('label')
    y[phase] = sorted_sdr_sequences.label.values

    # Convert SDRs to embeddings.
    (embeddings['sp'][phase],
     embeddings['tm'][phase]) = convert_to_embeddings(sorted_sdr_sequences,
                                                      aggregation,
                                                      nb_chunks)
    # Make sure the shapes are ok.
    nb_sequences = len(sorted_sdr_sequences)
    for cell_type in CELL_TYPES:
      check_shape(embeddings[cell_type][phase], (nb_sequences, nb_chunks,
                                                 sdr_widths[cell_type]))
      check_shape(embeddings[cell_type][phase], (nb_sequences, nb_chunks,
                                                 sdr_widths[cell_type]))
    check_shape(y[phase], (nb_sequences,))

    # Compute distance matrix.
    distance = lambda a, b: sequence_distance(a, b, assume_sequence_alignment)
    dist_mats['sp'][phase], dist_mats['tm'][phase], _ = distance_matrix(
      embeddings['sp'][phase], embeddings['tm'][phase], distance)

  # Step 2: Flatten the sequence embeddings to be able to classify each
  # sequence with a supervised classifier. The classifier uses the same
  # sequence distance as the distance matrix.
  for cell_type in CELL_TYPES:

    # Flatten embeddings.
    # Note: we have to flatten X because sklearn doesn't allow for X to be > 2D.
    # Here, the initial shape of X (i.e. sequence embeddings) is 3D and
    # therefore has to be flattened to 2D. See the logic of reshape_embeddings()
    # for details on how the embeddings are converted from 2D to 3D.
    nb_sequences = len(embeddings[cell_type]['train'])

    X[cell_type]['train'] = reshape_embeddings(embeddings[cell_type]['train'],
                                               nb_sequences, nb_chunks,
                                               sdr_widths[cell_type])
    X[cell_type]['test'] = reshape_embeddings(embeddings[cell_type]['test'],
                                              nb_sequences, nb_chunks,
                                              sdr_widths[cell_type])

    sequence_embedding_shape = (nb_chunks, sdr_widths[cell_type])
    reshaped_distance = lambda a, b: reshaped_sequence_distance(
      a, b, sequence_embedding_shape, assume_sequence_alignment)

    # Compute train and test accuracies
    (accuracies[cell_type]['train'],
     accuracies[cell_type]['test']) = train_and_test(X[cell_type]['train'],
                                                     y['train'],
                                                     X[cell_type]['test'],
                                                     y['test'],
                                                     reshaped_distance,
                                                     n_neighbors)

    # Step 3: plot the distance matrix and 2D projections for each cell
    # type (SP or TM) and phase (train or test).
    n_plots = 2  # distance matrix + 2d projection
    fig, ax, plot_path = make_subplots(len(PHASES), n_plots, plot_dir, data_id,
                                       cell_type, nb_chunks, aggregation)
    for phase in PHASES:
      phase_idx = PHASES.index(phase)
      title = make_plot_title('Pair-wise distances', phase,
                              accuracies[cell_type][phase])
      plot_matrix(dist_mats[cell_type][phase], title, fig, ax[phase_idx][0])

      if tsne:
        embeddings_proj = project_vectors(X[cell_type][phase],
                                          reshaped_distance)
        # Re-use the distance matrix to compute the 2D projections. It's faster.
        # embeddings_proj = project_matrix(dist_mats[cell_type][phase])

        title = make_plot_title('TSNE 2d projections', phase,
                                accuracies[cell_type][phase])
        plot_projections(embeddings_proj, y[phase], title, fig,
                         ax[phase_idx][1])

    fig.savefig(plot_path)

  return accuracies



def run_analysis(trace_dir, data_ids, chunks, n_neighbors, tsne, aggregations,
                 plot_dir, assume_sequence_alignment):
  if not os.path.exists(plot_dir): os.makedirs(plot_dir)

  tic = time.time()
  LOGGER.info('Analysis tree')
  for data_id in data_ids:
    LOGGER.info(indent(1) + 'load: ' + data_id)
    sdr_sequences = {}
    for phase in PHASES:
      f_path = os.path.join(trace_dir, 'trace_%s_%s' % (data_id, phase.upper()))
      sdr_sequences[phase] = load_sdrs(f_path, SP_OUT_WIDTH, TM_OUT_WIDTH)
      LOGGER.info(indent(2) + 'loaded: ' + f_path)
    LOGGER.info(indent(1) + 'analyze: ' + data_id)
    for aggregation in aggregations:
      LOGGER.info(indent(2) + 'aggregation: ' + aggregation)
      for nb_chunks in chunks:
        LOGGER.info(indent(3) + 'nb_chunks: ' + str(nb_chunks))
        accuracies = analyze_sdr_sequences(
          sdr_sequences['train'].copy(), sdr_sequences['test'].copy(), data_id,
          nb_chunks, n_neighbors, tsne, aggregation, plot_dir,
          assume_sequence_alignment)
        for cell_type, train_test_acc in accuracies.items():
          for phase, acc in train_test_acc.items():
            LOGGER.info(indent(4) + '%s %s accuracy: %s /100'
                        % (cell_type.upper(), phase, acc))

  toc = time.time()
  td = datetime.timedelta(seconds=(toc - tic))
  LOGGER.info('Elapsed time: %dh %02dm %02ds' % hours_minutes_seconds(td))
