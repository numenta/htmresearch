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
import inspect, os
import numpy as np
import pickle
import json
import sys
from optparse import OptionParser
from datasets import load_data
from htmresearch.support.lateral_pooler.utils import random_id
from nupic.algorithms.spatial_pooler import SpatialPooler as OldSpatialPooler
from htmresearch.algorithms.lateral_pooler import LateralPooler
from htmresearch.support.lateral_pooler.callbacks import (ModelCheckpoint, ModelOutputEvaluator, 
                                                          Reconstructor, ModelInspector, 
                                                          OutputCollector, Logger)



def dump_dict(path, my_dict):
  filename = "config.json"
  with open(path + filename, 'wb') as file:
      json.dump(my_dict, file, indent=4)


def dump_results(path, results):
  for key in results:
    for i, data in enumerate(results[key]):
      filename = key + "_{}.p".format(i + 1)
      with open(path + filename, 'wb') as file:
        pickle.dump(data, file)


def get_shape(params):
  if "inputDimensions" in params:
    return params["columnDimensions"][0], params["inputDimensions"][0]
  else:
    return params["output_size"], params["input_size"]


def get_permanence_vals(sp):
  m = sp.getNumInputs()
  n = np.prod(sp.getColumnDimensions())
  W = np.zeros((n, m))
  for i in range(sp._numColumns):
    sp.getPermanence(i, W[i, :])
  return W


def parse_argv():
  parser = OptionParser(usage = "python run_experiment.py --sp lateral --data mnist --params mnist -e 2 -b 30 -d 100")
  parser.add_option("--data",   type=str, default='', dest="data_set", help="")
  parser.add_option("-d", "--num_data",  type=int, default=30000, dest="num_data_points", help="")
  parser.add_option("-e",     "--num_epochs", type=int, default=6, dest="num_epochs", help="number of epochs")
  parser.add_option("-b",     "--batch_size", type=int, default=30, dest="batch_size", help="Mini batch size")
  parser.add_option("--sp", type=str, default="ordinary", dest="pooler_type", help="spatial pooler implementations: ordinary, lateral")
  parser.add_option("--params", type=str, dest="sp_params", help="json file with spatial pooler parameters")
  parser.add_option("--name", type=str, default=None, dest="experiment_id", help="")
  parser.add_option("--seed", type=str, default=41, dest="seed", help="random seed for SP and dataset")
  (options, remainder) = parser.parse_args()
  print(options)
  return options, remainder


def main(argv):
  args, _ = parse_argv()
  data_set        = args.data_set
  num_data_points = args.num_data_points
  sp_type         = args.pooler_type
  num_epochs      = args.num_epochs
  batch_size      = args.batch_size
  experiment_id   = args.experiment_id
  seed            = args.seed


  the_scripts_path = os.path.dirname(os.path.realpath(__file__)) # script directory

  sp_params_dict  = json.load(open(the_scripts_path + "/params.json"))
  if args.sp_params is not None:
    sp_params       = sp_params_dict[sp_type][args.sp_params]
  else:
    sp_params       = sp_params_dict[sp_type][data_set]
  sp_params["seed"] = seed

  if experiment_id is None:
    experiment_id = random_id(5)

  path = the_scripts_path + "/../results/{}_pooler_{}_{}/".format(sp_type, data_set,experiment_id)
  os.makedirs(os.path.dirname(path))

  print(
    "Experiment directory:\n\"{}\"\n"
    .format(path))

  X, _, X_test, _ = load_data(data_set, num_inputs = num_data_points) 

  n, m = get_shape(sp_params)
  X    = X[:,:num_data_points]
  d    = X.shape[1]
  

  results = {
    "inputs"      : [],
    "outputs"     : [],
    "feedforward" : []}


  ####################################################
  # 
  #               Old Spatial Pooler
  # 
  ####################################################
  if sp_type == "ordinary":
    
    pooler = OldSpatialPooler(**sp_params)

    print(
      "Training ordinary pooler:\n")

    # "Fit" the model to the training data
    for epoch in range(num_epochs):
      Y = np.zeros((n,d))

      perm = np.random.permutation(d)

      for t in range(d):

        sys.stdout.flush()
        sys.stdout.write(
          "\r{}/{}  {}/{}"
            .format(num_epochs, epoch + 1, d, t + 1))

        x = X[:,perm[t]]
        y = Y[:, t]
        pooler.compute(x, True, y)

      results["inputs"].append(X)
      results["outputs"].append(Y)
      results["feedforward"].append(get_permanence_vals(pooler)) 
      
  
  ####################################################
  # 
  #             New Spatial Pooler with 
  #     learned lateral inhibitory connections
  # 
  #####################################################
  elif sp_type == "lateral":

    pooler = LateralPooler(**sp_params)

    sys.stdout.write(
      "Training dynamic lateral pooler:\n")

    collect_feedforward = ModelInspector(lambda pooler: pooler.feedforward.copy(), on_batch = False )
    # collect_lateral     = ModelInspector(lambda pooler: pooler.inhibitory.copy(),  on_batch = False )
    training_log        = OutputCollector()
    print_training_status = Logger()

    # "Fit" the model to the training datasets
    pooler.fit(X, batch_size=batch_size, num_epochs=num_epochs, initial_epoch=0, callbacks=[collect_feedforward, training_log, print_training_status])

    results["inputs"]      = training_log.get_inputs()
    results["outputs"]     = training_log.get_outputs()
    results["feedforward"] = collect_feedforward.get_results()
    # results["lateral"]     = collect_lateral.get_results() 


  dump_dict(path, sp_params)
  dump_results(path, results)

  print(
    "\nDone.\n")



if __name__ == "__main__":
   main(sys.argv[1:])

