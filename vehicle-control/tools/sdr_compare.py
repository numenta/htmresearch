#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""This script contains functions for collecting depth data
and comparing SDRs encoded from the data. It is used to aid in
assessing the quality of the depth encoder.

Data is collected directly from unity by pressing "c" during game
play while this script is running with collectDepthData().
It may be necessary to press, hold, and then release the key until
"Collecting data." is printed. This will save the depth data as a numpy
array in a file titled with the datetime. A screenshot will also be saved
with the datetime in vehicle-control/simulations/Screenshots/. Compare
the datetime of both files to ensure that screenshots and depth data were
collected.

To pairwise compare all the n encodings of saved depth data, run
compareDepthData(). This will generate a nxn plot where the bottom left
triangle depicts the SDRs overlapped and the top right triangle displays
the percent overlap (relative to the density).
"""


import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from unity_client.fetcher import Fetcher
from sensorimotor.encoders.one_d_depth import OneDDepthEncoder

positions = [i*10 for i in range(18, 54)]
nPerPosition = 28

_SHAPE = (36, nPerPosition)
_OUTPUT_DIR = "depths/"
_ENCODER_PARAMS = {
		"positions": positions,
		"radius": 5,
		"wrapAround": True,
		"nPerPosition": nPerPosition,
		"wPerPosition": 3,
		"minVal": 0,
		"maxVal": 1
}

def compareTwoSDRS(firstSDR, secondSDR, ax=None, shape=_SHAPE):

	overlap = np.dot(firstSDR, secondSDR)
	density = max(sum(firstSDR), sum(secondSDR))

	similarityScore = (1.0*overlap)/density

	if ax:

		firstEncoding = np.array(firstSDR).reshape(shape).transpose()
		secondEncoding = np.array(secondSDR).reshape(shape).transpose()

		ax.imshow(firstEncoding,
		                cmap=cm.Blues,
		                interpolation="nearest",
		                aspect='auto',
		                vmin=0,
		                vmax=1,
		                alpha=0.5)

		ax.imshow(secondEncoding,
						cmap=cm.Reds,
						interpolation="nearest",
		                aspect='auto',
		                vmin=0,
		                vmax=1,
		                alpha=0.5)

	return similarityScore

def comparePlotSDRs(SDRList, shape=_SHAPE):

	n = len(SDRList)

	f, axs = plt.subplots(n, n, sharex='col', sharey='row')

	for i in range(n):
		for j in range(i+1):

			firstSDR = SDRList[i]
			secondSDR = SDRList[j]
			score = compareTwoSDRS(firstSDR, secondSDR, axs[i][j], shape)

			axs[i][j].xaxis.set_ticklabels([])
			axs[j][i].xaxis.set_ticklabels([])
			axs[i][j].yaxis.set_ticklabels([])
			axs[j][i].yaxis.set_ticklabels([])

			title = "{0:.2f}".format(score)
			axs[j][i].text(.5, 0.5, title, horizontalalignment='center',
        				   transform=axs[j][i].transAxes)

	plt.show()

def collectDepthData(outputDir=_OUTPUT_DIR):

	fetcher = Fetcher()

	holdingDownKey = False

	while True:

		outputData = fetcher.sync()

		if outputData is None or "collectKeyPressed" not in outputData:
		  holdingDownKey = False
		  continue

		if outputData["collectKeyPressed"] == 0:
			holdingDownKey = False
			continue

		if holdingDownKey:
			print "Stop holding down the key!"
			continue

		print "Collecting data."

		holdingDownKey = True

		sensor = outputData["ForwardsSweepSensor"]
		depthData = np.array(sensor)

		dt = time.strftime("%y%m%d%H%M%S")
		depthID = 'depth_'+dt+'.txt'

		depthOutputPath = os.path.join(os.path.dirname(__file__)+outputDir,
									   depthID)

		np.savetxt(depthOutputPath, depthData)

		print "Wrote out to ", depthOutputPath

def prepareDepthData(inputDir, encoder_params):
	depthDir = os.path.dirname(__file__)+inputDir
	SDRs = []

	encoder = OneDDepthEncoder(**encoder_params)

	for f in os.listdir(depthDir):
		if f.endswith(".txt"):
			print "Loading ", f
			dataPath = os.path.join(depthDir, f)
			depthData = np.loadtxt(dataPath)
			SDR = encoder.encode(depthData)
			SDRs.append(SDR)

	return SDRs


def compareDepthData(inputDir=_OUTPUT_DIR, encoder_params=_ENCODER_PARAMS):

	SDRs =prepareDepthData(inputDir, encoder_params)
	comparePlotSDRs(SDRs)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--record",
                    	default=False,
                    	action='store_true',
                    	help="Start recording depth data.")

	parser.add_argument("--compare",
                    	default=False,
                    	action='store_true',
                   		help="Compare depth data.")

	args = parser.parse_args()

	if args.record and args.compare:
		raise "Cannot record and compare simultaneously."

	if not (args.record or args.compare):
		args.record = True

	if args.record:
		collectDepthData()

	if args.compare:
		compareDepthData()