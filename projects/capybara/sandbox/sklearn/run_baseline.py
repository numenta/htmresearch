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
import numpy as np
from sklearn.neural_network import MLPClassifier

from htmresearch.frameworks.classification.utils.traces import loadTraces
from utils import get_file_name, convert_to_sdrs



def load_sdrs(start_idx, end_idx, exp_name):
  # Params
  input_width = 2048 * 32
  active_cells_weight = 0
  predicted_active_cells_weight = 1
  network_config = 'sp=True_tm=True_tp=False_SDRClassifier'

  # load traces
  file_name = get_file_name(exp_name, network_config)
  traces = loadTraces(file_name)
  num_records = len(traces['sensorValue'])

  # start and end 
  if start_idx < 0:
    start = num_records + start_idx
  else:
    start = start_idx
  if end_idx < 0:
    end = num_records + end_idx
  else:
    end = end_idx

  # input data
  sensor_values = traces['sensorValue'][start:end]
  categories = traces['actualCategory'][start:end]
  active_cells = traces['tmActiveCells'][start:end]
  predicted_active_cells = traces['tmPredictedActiveCells'][start:end]

  # generate sdrs to cluster
  active_cells_sdrs = convert_to_sdrs(active_cells, input_width)
  predicted_active_cells_sdrs = np.array(
    convert_to_sdrs(predicted_active_cells, input_width))
  sdrs = (float(active_cells_weight) * np.array(active_cells_sdrs) +
          float(predicted_active_cells_weight) * predicted_active_cells_sdrs)

  return sdrs, categories



def train_model(X, y):
  clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(5, 2), random_state=1)

  clf.fit(X, y)
  return clf



if __name__ == "__main__":
  exp_name = '1x.40000.body_acc_x'
  start_idx = 600
  end_idx = 800
  sdrs, categories = load_sdrs(start_idx, end_idx, exp_name)

  clf = train_model(sdrs, categories)

  predictions = clf.predict([sdrs[0], sdrs[1]])
  print "Predictions: %s" % predictions
