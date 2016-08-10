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

import unittest
import numpy as np

import scipy.sparse as sparse

from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
	TemporalMemoryMonitorMixin
	)
from htmresearch.algorithms.column_pooler import ColumnPooler, realDType


class MonitoredColumnPooler(TemporalMemoryMonitorMixin, ColumnPooler): 
	pass


class ExtensiveColumnPoolerTest(unittest.Testcase):
	"""Algorithmic tests for the ColumnPooler region."""

	inputWidth = 2048
	inputActive = 40
	outputWidth = 2048
	outputActive = 40
	seed = 42


	def setUp(self):
		self.pooler = None
		self.patternMachine = PatternMachine(
			self.inputWidth, 
			self.outputActive, 
			self.seed
			)
		np.seed(self.seed)


	def init(self, overrides=None):
		params = self._computeParams(overrides)
		self.pooler = MonitoredColumnPooler(params)


	def getDefaultPoolerParams(self):
	  return {
	  	"inputWidth": self.inputWidth,
	  	"numActivecolumnsPerInhArea": self.outputActive,
	  	"synPermProximalInc": 0.1,
	  	"synPermProximalDec": 0.001,
	    "columnDimensions": (self.outputWidth,),
	    "cellsPerColumn": 32,
	    "initialPermanence": 0.5,
	    "connectedPermanence": 0.6,
	    "minThreshold": 25,
	    "maxNewSynapseCount": 30,
	    "permanenceIncrement": 0.1,
	    "permanenceDecrement": 0.02,
	    "predictedSegmentDecrement": 0.08,
	    "activationThreshold": 25,
	    "seed": self.seed,
	    "learnOnOneCell": False,
	  }


	def learnObject(self,
									feedforwardPatterns,
									lateralPatterns=None, 
									numRepetitions=1, 
									randomOrder=True):
		self.pooler.mmClearHistory()
		self.pooler.reset()

		# set-up
		indices = range(len(feedforwardPatterns))
		if lateralPatterns is None:
			lateralPatterns = [None] * len(feedforwardPatterns)

		for _ in xrange(numRepetitions):
			if randomOrder:
				np.random.shuffle(indices)

			for idx in indices:
				self.pooler.compute(feedforwardInput=feedforwardPatterns[idx], 
														activeExternalCells=lateralPatterns[idx],
														learn=True)


	def infer(self, feedforwardPattern, lateralPattern, printMetrics=False):



	def _computeParams(self, overrides):
		params = self.getDefaultPoolerParams()
		params.update(overrides)
		return params


if __name__ == "__main__":
	unittest.main()
