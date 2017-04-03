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

import json
import numpy as np
import pandas as pd



def make_embedding(sdrs_chunk, aggregation):
  if aggregation == 'or':
    embedding = sdrs_chunk[0]
    for sdr in sdrs_chunk:
      embedding = np.logical_or(embedding, sdr)
  elif aggregation == 'and':
    embedding = sdrs_chunk[0]
    for sdr in sdrs_chunk:
      embedding = np.logical_and(embedding, sdr)
  elif aggregation == 'mean':
    embedding = np.mean(sdrs_chunk, axis=0)
  else:
    raise ValueError('Invalid aggregation name.')
  return embedding



def make_embeddings(sdrs_sequence, aggregation, nb_chunks):
  """
  Split a sequence of SDRs in chunks and create an embedding for each chunk.
  :param sdrs_sequence: (array of arrays) a sequence of SDRs
  :param aggregation: (str) type of aggregation
  :param nb_chunks: (int) how many chunks in the SDRs sequence
  :return: (array of arrays) embeddings
  """
  chunk_size = len(sdrs_sequence) / nb_chunks
  embeddings = []
  for i in range(nb_chunks):
    chunk = sdrs_sequence[i * chunk_size:(i + 1) * chunk_size]
    embeddings.append(make_embedding(chunk, aggregation))
  embeddings = np.array(embeddings)
  return embeddings



def convert_to_embeddings(sdr_sequences, aggregation, nb_chunks):
  sp_embeddings = []
  for sdr_sequence in sdr_sequences.spActiveColumns.values:
    sp_embeddings.append(make_embeddings(sdr_sequence, aggregation, nb_chunks))
  sp_embeddings = np.array(sp_embeddings)

  tm_embeddings = []
  for sdr_sequence in sdr_sequences.tmPredictedActiveCells.values:
    tm_embeddings.append(make_embeddings(sdr_sequence, aggregation, nb_chunks))
  tm_embeddings = np.array(tm_embeddings)

  return  sp_embeddings, tm_embeddings



def save_embeddings(embeddings, labels, output_file_path):
  assert len(embeddings) == len(labels)

  df = pd.DataFrame(
    data={'embedding': [json.dumps(e.tolist()) for e in embeddings],
          'label': labels})
  print df.head()
  df.to_csv(output_file_path)


def reshape_embeddings(embeddings, nb_sequences, nb_chunks, sdr_width):
  # Initial embeddings shape = (nb_sequences, nb_chunks, sdr_width)
  # Final embeddings shape = (nb_sequences, nb_chunks * sdr_width)
  return embeddings.reshape((nb_sequences,  nb_chunks *sdr_width))
