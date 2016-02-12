#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

"""
Script to run temporal memory on NLP documents
"""

import argparse
from textwrap import TextWrapper
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import nupic
from nupic.data.file_record_stream import FileRecordStream
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
	createModel, getNetworkConfig)
from htmresearch.support.csv_helper import readDataAndReshuffle

plt.ion()
wrapper = TextWrapper(width=100)



def getTMRegion(network):
	tmRegion = None
	for region in network.regions.values():
		regionInstance = region
		if type(regionInstance.getSelf()) is nupic.regions.TPRegion.TPRegion:
			tmRegion = regionInstance.getSelf()
	return tmRegion



def instantiateModel(args):
	"""
	Return an instance of the model we will use.
	"""
	# Some values of K we know work well for this problem for specific model types
	kValues = {"keywords": 21, "docfp": 3}

	# Create model after setting specific arguments required for this experiment
	args.networkConfig = getNetworkConfig(args.networkConfigPath)
	args.k = kValues.get(args.modelName, 1)
	args.numLabels = 2
	model = createModel(**vars(args))

	return model



def trainModel(args, model, trainingData, labelRefs):
	"""
	Train the given model on trainingData. Return the trained model instance.
	"""

	tmRegion = getTMRegion(model.network)

	print
	print "=======================Training model on sample text================"
	for recordNum, doc in enumerate(trainingData):
		document = doc[0]
		labels = doc[1]
		docId = doc[2]
		if args.verbosity > 0:
			print
			print "Document=", wrapper.fill(document)
			print "label=", labelRefs[labels[0]], "id=", docId
		model.trainDocument(document, labels, docId)

		numActiveCols = tmRegion._tfdr.mmGetTraceActiveColumns().makeCountsTrace().data
		numPredictedActiveCells = \
			tmRegion._tfdr.mmGetTracePredictedActiveCells().makeCountsTrace().data

		if args.verbosity > 0:
			print "Word # %s, Avg Active Cols # %s, Avg predicted-active cell # %s " % (
				len(numActiveCols),
				np.mean(np.array(numActiveCols)),
				np.mean(np.array(numPredictedActiveCells))
			)
		tmRegion._tfdr.mmClearHistory()

	return model



def runExperiment(args):
	"""
	Create model according to args, train on training data, save model,
	restore model, test on test data.
	"""

	args.numLabels = 2
	(trainingData, labelRefs, documentCategoryMap,
	 documentTextMap) = readDataAndReshuffle(args)

	# Create model
	model = instantiateModel(args)

	model = trainModel(args, model, trainingData, labelRefs)

	# TODO: Visualize prediction quality



if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument("-c", "--networkConfigPath",
	                    default="data/network_configs/tm_knn_4k_retina.json",
	                    help="Path to JSON specifying the network params.",
	                    type=str)
	parser.add_argument("-m", "--modelName",
	                    default="htm",
	                    type=str,
	                    help="Name of model class. Options: [keywords,htm,docfp]")
	parser.add_argument("--retinaScaling",
	                    default=1.0,
	                    type=float,
	                    help="Factor by which to scale the Cortical.io retina.")
	parser.add_argument("--retina",
	                    default="en_associative_64_univ",
	                    type=str,
	                    help="Name of Cortical.io retina.")
	parser.add_argument("--apiKey",
	                    default=None,
	                    type=str,
	                    help="Key for Cortical.io API. If not specified will "
	                         "use the environment variable CORTICAL_API_KEY.")
	parser.add_argument("--modelDir",
	                    default="MODELNAME.checkpoint",
	                    help="Model will be saved in this directory.")
	parser.add_argument("-v", "--verbosity",
	                    default=1,
	                    type=int,
	                    help="verbosity 0 will print out experiment steps, "
	                         "verbosity 1 will include results, and verbosity > "
	                         "1 will print out preprocessed tokens and kNN "
	                         "inference metrics.")
	args = parser.parse_args()

	# By default set checkpoint directory name based on model name
	if args.modelDir == "MODELNAME.checkpoint":
		args.modelDir = args.modelName + ".checkpoint"
		print "Save dir: ", args.modelDir

	runExperiment(args)
