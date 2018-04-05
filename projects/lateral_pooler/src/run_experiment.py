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

from tabulate import tabulate
import itertools

import matplotlib.pyplot as plt
from pprint import PrettyPrinter
pprint = PrettyPrinter(indent=4).pprint

from htmresearch.support.lateral_pooler.datasets import load_data
from htmresearch.support.lateral_pooler.utils import random_id, add_noise
from htmresearch.support.lateral_pooler.metrics import mean_mutual_info_from_model, reconstruction_error
# from htmresearch.frameworks.sp_paper.sp_metrics import reconstructionError

from nupic.algorithms.spatial_pooler import SpatialPooler as SpatialPooler
from sp_wrapper import SpatialPoolerWrapper 

from htmresearch.algorithms.lateral_pooler import LateralPooler


from htmresearch.support.lateral_pooler.callbacks import (ModelCheckpoint, ModelOutputEvaluator, 
                                                          Reconstructor, ModelInspector, 
                                                          OutputCollector, Logger)


def load_pooler(path):
    filename = "{}/pooler.p".format(path)
    with open(filename) as f:
        pooler = pickle.load(f)
    
    return pooler

def dump_pooler(path, t, pooler):
    filename = "{}/pooler/pooler_{:04}.p".format(path, t)
    with open(filename, 'wb') as f:
        pickle.dump(pooler, f)

def dump_entity(path, label, t, X):
    filename = "{}/{}/{}_{:04}.p".format(path, label,label, t)
    with open(filename, 'wb') as f:
        pickle.dump(X, f)

def dump_json(path_to_file, my_dict):
    with open(path_to_file, 'wb') as f:
        json.dump(my_dict, f, indent=4)

def dump_data(path_to_file, data):
    with open(path_to_file, 'wb') as f:
        pickle.dump(data, f)


def dump_results(path, results):
    for key in results:
        os.makedirs(os.path.dirname("{}/{}/".format(path, key)))
        for i, data in enumerate(results[key]):
            filename ='{}/{}/{}_{:04}.p'.format(path, key, key, i + 1)
            with open(filename, 'wb') as file:
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
    parser = OptionParser(usage = "<yourscript> [options]\n\n"\
                                  "Example:\n"\
                                  "python {} --sp lateral --data mnist --params mnist --name example -e 2 -b 1 -d 100"
                                  .format(sys.argv[0]))
    parser.add_option("--data",   type=str, default='', dest="data_set", help="")
    parser.add_option("-d", "--num_data",  type=int, default=30000, dest="num_data_points", help="")
    parser.add_option("-e",     "--num_epochs", type=int, default=6, dest="num_epochs", help="number of epochs")
    parser.add_option("-b",     "--batch_size", type=int, default=1, dest="batch_size", help="Mini batch size")
    parser.add_option("--sp", type=str, default="nupic", dest="pooler_type", help="spatial pooler implementations: nupic, lateral")
    parser.add_option("--params", type=str, dest="sp_params", help="json file with spatial pooler parameters")
    parser.add_option("--name", type=str, default=None, dest="experiment_id", help="")
    parser.add_option("--seed", type=str, default=None, dest="seed", help="random seed for SP and dataset")
    parser.add_option("--cont", type=str, default=None, dest="cont", help="...")
    (options, remainder) = parser.parse_args()
    return options, remainder

####################################################
####################################################
####################################################
### 
###                   Main()
### 
####################################################
####################################################
####################################################

def main(argv):
    args, _ = parse_argv()

    data_set        = args.data_set
    d               = args.num_data_points
    sp_type         = args.pooler_type
    num_epochs      = args.num_epochs
    batch_size      = args.batch_size
    experiment_id   = args.experiment_id
    seed            = args.seed
    cont            = args.cont

    ####################################################
    # 
    #       Create folder for the experiment
    # 
    ####################################################
    if experiment_id is None:
        experiment_id = random_id(5)
    else:
        experiment_id +=  "_" + random_id(5)

    the_scripts_path = os.path.dirname(os.path.realpath(__file__)) # script directory
    relative_path    = "../results/{}_pooler_{}_{}".format(sp_type, data_set, experiment_id)
    path             = the_scripts_path + "/" + relative_path
    
    os.makedirs(os.path.dirname("{}/".format(path)))
    os.makedirs(os.path.dirname("{}/pooler/".format(path)))
    os.makedirs(os.path.dirname("{}/inputs/".format(path)))
    os.makedirs(os.path.dirname("{}/outputs/".format(path)))
    os.makedirs(os.path.dirname("{}/targets/".format(path)))
    os.makedirs(os.path.dirname("{}/test_inputs/".format(path)))
    os.makedirs(os.path.dirname("{}/test_targets/".format(path)))




    print(
        "\nExperiment directory:\n\n\t\"{}\"\n"
        .format(relative_path))


    ####################################################
    # 
    #               Load the sp parameters
    # 
    ####################################################

    sp_params_dict  = json.load(open(the_scripts_path + "/params.json"))

    if args.sp_params is not None:
        sp_params = sp_params_dict["nupic"][args.sp_params]
    else:
        sp_params = sp_params_dict["nupic"][data_set]

    if seed is not None:
        sp_params["seed"] = seed

    pprint(sp_params)

    ####################################################
    # 
    #                   Load the SP
    # 
    ####################################################
    if sp_type == "nupic":
        if cont == None:
            pooler = SpatialPoolerWrapper(**sp_params)
        else:
            pooler = load_pooler(cont)

    elif sp_type == "lateral":
        if cont == None:
            pooler = LateralPooler(**sp_params)
        else:
            pooler = load_pooler(cont)
    else:
        raise "I don't know an SP of that type:{}".format(sp_type)

    print(
        "\nUsing {} pooler.\n\n"\
        "\tdesired sparsity: {}\n"\
        "\tdesired code weight: {}\n"
        .format(sp_type, pooler.sparsity, pooler.code_weight))

    ####################################################
    # 
    #                Load the data
    # 
    ####################################################

    X, T, X_test, T_test = load_data(data_set) 
    X      = X[:,:d]

    dump_entity(path, "inputs", 0, X)
    dump_entity(path, "targets", 0, T)
    dump_entity(path, "test_inputs", 0, X_test)
    dump_entity(path, "test_targets", 0, T_test)


    ####################################################
    # 
    #                   Training
    # 
    ####################################################

    if batch_size != 1:
        raise Exception(
                "Let's stick with online learning for now..."\
                "set the batch size -b to 1.")

    checkpoints = set(range(0, d, 100))

    results = {
        "outputs_test": [],
        "inputs_test": [],
        "feedforward" : [],
        "avg_activity_units" : [],
        "avg_activity_pairs" : []}

    metrics = {
        "code_weight" : [],
        "rec_error"   : [],
        "mutual_info" : []}

    config = {
        "num_checkpoints" : len(checkpoints),
        "path": relative_path,
        "sp_type": sp_type,
        "data_set": data_set,
        "sparsity": pooler.sparsity,
        "code_weight": pooler.code_weight
    }



    print(
        "Training (online, i.e. batch size = 1)...\n"
        .format(sp_type))

    # 
    # "Fit" the model to the training data
    # 
    n, m   = pooler.shape
    for epoch in range(num_epochs):

        print(
            "\te:{}/{}"
            .format(epoch + 1, num_epochs,))

        Targets = np.zeros((T.shape[0],d))
        Y    = np.zeros((n,d))
        perm = np.random.permutation(d)

        check_count = 0
        for t in range(d):

            if t%500 == 0:
                print(  
                    "\t\tb:{}/{}"
                    .format(t, d))

            x = X[:,perm[t]]
            y = Y[:, t]
            Targets[:, t] = T[:,perm[t]]
            pooler.compute(x, True, y)

            if t in checkpoints:
                check_count += 1
                # Y_test = pooler.encode(X_test)

                # results["avg_activity_units"].append(pooler._activeDutyCycles.copy()) 
                # results["avg_activity_pairs"].append(pooler.avgActivityPairs.copy()) 
                
                # results["inputs_test"].append(X_test)
                # dump_entity(path, "pooler", check_count, pooler)
                # dump_entity(path, "inputs", check_count, X)
                # dump_entity(path, "outputs", check_count, Y)
                # dump_entity(path, "targets", check_count, Targets)
                
                # results["outputs_test"].append(Y_test)
                # results["feedforward"].append(pooler.feedforward.copy()) 

                # metrics["code_weight"].append(np.mean(np.sum(Y_test, axis=0)))
                # metrics["mutual_info"].append(mean_mutual_info_from_model(pooler))
                # metrics["rec_error"].append(reconstruction_error(pooler, X_test))






    print("")
    print(tabulate(metrics, headers="keys"))

    print(
        "\nSaving results to file...")

    # dump_json(path + "/metrics.json", metrics)
    dump_json(path + "/config.json", config)
    # dump_results(path, results)
    dump_pooler(path, check_count, pooler)

    print(
        "Done.")


if __name__ == "__main__":
    main(sys.argv[1:])

