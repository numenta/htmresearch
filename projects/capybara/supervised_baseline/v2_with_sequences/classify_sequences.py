#!/usr/bin/env python
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
from  htmresearch.frameworks.capybara.supervised.analysis import run_analysis

def main():
  tsne = True
  chunks = [1,2,5]
  n_neighbors = 1
  aggregations = ['mean']
  trace_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, os.pardir, 'htm', 'traces')
  assume_sequence_alignment = True
  if assume_sequence_alignment:
    plot_dir = 'plots_assume_sequences_aligned'
  else:
    plot_dir = 'plots_assume_sequences_not_aligned'

  data_ids = [
    #'debug_body_acc_x',
    'body_acc_x',
    'synthetic_control',
    'Test1'
  ]

  run_analysis(trace_dir, data_ids, chunks, n_neighbors, tsne, aggregations,
               plot_dir, assume_sequence_alignment)



if __name__ == '__main__':
  main()