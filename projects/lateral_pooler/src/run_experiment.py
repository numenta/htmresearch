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
from tabulate import tabulate

from pprint import PrettyPrinter
pprint = PrettyPrinter(indent=4).pprint

from htmresearch.support.lateral_pooler.utils import random_id
from htmresearch.support.lateral_pooler.metrics import mean_mutual_info_from_data, mean_mutual_info_from_model
from htmresearch.frameworks.sp_paper.sp_metrics import reconstructionError

from nupic.algorithms.spatial_pooler import SpatialPooler as SpatialPooler
from sp_wrapper import SpatialPoolerWrapper 
from htmresearch.algorithms.lateral_pooler import LateralPooler
from htmresearch.algorithms.lateral_pooler_wrapper import LateralPoolerWrapper as LateralPoolerWrapper

from htmresearch.support.lateral_pooler.callbacks import (ModelCheckpoint, ModelOutputEvaluator, 
                                                          Reconstructor, ModelInspector, 
                                                          OutputCollector, Logger)



def dump_json(path_to_file, my_dict):
    with open(path_to_file, 'wb') as f:
        json.dump(my_dict, f, indent=4)


def dump_results(path, results):
    for key in results:
        for i, data in enumerate(results[key]):
            filename ="{}/{}_{}.p".format(paht, key, i + 1)
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
    parser = OptionParser(usage = "python run_experiment.py --sp lateral --data mnist --params mnist -e 2 -b 1 -d 100")
    parser.add_option("--data",   type=str, default='', dest="data_set", help="")
    parser.add_option("-d", "--num_data",  type=int, default=30000, dest="num_data_points", help="")
    parser.add_option("-e",     "--num_epochs", type=int, default=6, dest="num_epochs", help="number of epochs")
    parser.add_option("-b",     "--batch_size", type=int, default=1, dest="batch_size", help="Mini batch size")
    parser.add_option("--sp", type=str, default="ordinary", dest="pooler_type", help="spatial pooler implementations: ordinary, lateral")
    parser.add_option("--params", type=str, dest="sp_params", help="json file with spatial pooler parameters")
    parser.add_option("--name", type=str, default=None, dest="experiment_id", help="")
    parser.add_option("--seed", type=str, default=41, dest="seed", help="random seed for SP and dataset")
    (options, remainder) = parser.parse_args()
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
    path = the_scripts_path + "/../results/{}_pooler_{}_{}".format(sp_type, data_set,experiment_id)
    os.makedirs(os.path.dirname(path+"/"))

    print(
        "\nExperiment directory:\n\n\t\"{}\"\n"
        .format(path))


    ####################################################
    # 
    #               Load the sp parameters
    # 
    ####################################################

    sp_params_dict  = json.load(open(the_scripts_path + "/params.json"))

    if args.sp_params is not None:
        # sp_params       = sp_params_dict[sp_type][args.sp_params]
        sp_params         = sp_params_dict["ordinary"][args.sp_params]
    else:
        # sp_params       = sp_params_dict[sp_type][data_set]
        sp_params         = sp_params_dict["ordinary"][data_set]

    pprint(sp_params)
    ####################################################
    # 
    #                   Load the SP
    # 
    ####################################################
    if sp_type == "ordinary":
        pooler = SpatialPooler(**sp_params)

    elif sp_type == "lateral":
        pooler = LateralPoolerWrapper(**sp_params)

    print(
        "Using {} pooler.\n\n"\
        "\tdesired sparsity: {}\n"\
        "\tdesired code weight: {}\n"
        .format(sp_type, pooler._localAreaDensity, pooler.code_weight))
    ####################################################
    # 
    #                Load the data
    # 
    ####################################################

    X, _, X_test, _ = load_data(data_set, num_inputs = num_data_points) 

    n, m = pooler._numColumns, X.shape[0]
    X    = X[:,:num_data_points]
    d    = X.shape[1]

    ####################################################
    # 
    #                   Training
    # 
    ####################################################

    if batch_size != 1:
        raise Exception(
                "Let's stick with online learning for now..."\
                "set the batch size -b to 1.")


    results = {
        "inputs"      : [],
        "outputs"     : [],
        "feedforward" : [],
        "avg_activity_units" : []}

    metrics = {
        "code_weight" : [],
        "rec_error"   : [],
        "mutual_info" : []}


    checkpoints = set(range(0, num_epochs, 10))
    checkpoints.add(num_epochs-1)

    print(
        "Training (online, i.e. batch size = 1)...\n"
        .format(sp_type))

    # "Fit" the model to the training data
    for epoch in range(num_epochs):

        print(
            "\te:{}/{}"
            .format(num_epochs, epoch + 1))

        Y    = np.zeros((n,d))
        perm = np.random.permutation(d)

        for t in range(d):

            x = X[:,perm[t]]
            y = Y[:, t]
            pooler.compute(x, True, y)


        if t in checkpoints:
            results["inputs"].append(X)
            results["outputs"].append(Y)
            # results["avg_activity_units"].append(pooler.avg_activity_units) 
            # results["avg_activity_pairs"].append(pooler.avg_activity_pairs) 

        # results["feedforward"].append(get_permanence_vals(pooler)) 

        # metrics["mutual_info"].append(mean_mutual_info_from_data(Y))

        # metrics["mutual_info"].append(mean_mutual_info_from_model(pooler))
        metrics["code_weight"].append(np.mean(np.sum(Y, axis=0)))
        # metrics["rec_error"].append(reconstructionError(pooler, X.T, Y.T, threshold=0.))


    # ####################################################
    # # 
    # #             New Spatial Pooler with 
    # #     learned lateral inhibitory connections
    # # 
    # #####################################################
    # elif sp_type == "lateral":

    #   pooler = LateralPooler(**sp_params)

    #   sys.stdout.write(
    #     "Training dynamic lateral pooler:\n")

    #   collect_feedforward = ModelInspector(lambda pooler: pooler.feedforward.copy(), on_batch = False )
    #   # collect_lateral     = ModelInspector(lambda pooler: pooler.inhibitory.copy(),  on_batch = False )
    #   training_log        = OutputCollector()
    #   print_training_status = Logger()

    #   # "Fit" the model to the training datasets
    #   pooler.fit(X, batch_size=batch_size, num_epochs=num_epochs, initial_epoch=0, callbacks=[collect_feedforward, training_log, print_training_status])

    #   results["inputs"]      = training_log.get_inputs()
    #   results["outputs"]     = training_log.get_outputs()
    #   results["feedforward"] = collect_feedforward.get_results()
    #   # results["lateral"]     = collect_lateral.get_results() 

    print("")
    print(tabulate(metrics, headers="keys"))

    print(
        "\nSaving results to file...")

    dump_json(path + "/metrics", metrics)
    dump_results(path, results)

    print(
        "Done.")





if __name__ == "__main__":
   main(sys.argv[1:])

