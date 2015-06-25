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
import numpy as np



class _InputPairCreator(object):
	def getNewPair(self):
		"""Returns a tuple of inputs."""
		raise "Unimplemented."

class InputPairCreatorFromList(_InputPairCreator):
	def __init__(self, sampleList):
		self.sampleList = sampleList

	def getNewPair(self):
		first = np.random.choice(self.sampleList)
		second = np.random.choice(self.sampleList)

		return (first, second)

class InputPairCreatorFromGenerator(_InputPairCreator):
	def __init__(self, sampleGenerator):
		self.sampleGenerator = sampleGenerator

	def getNewPair(self):
		first = next(self.sampleGenerator)
		second = next(self.sampleGenerator)

		return (first, second)


def SDRSimilarity(first_sdr, second_sdr):
	"""Percent similarity between the two SDRs."""
	overlap = np.sum(np.logical_and(first_sdr, second_sdr))
	density = max(np.sum(first_sdr), np.sum(second_sdr))
	return (1.0*overlap)/density

def singleSampleDiff(encoder, first_input, second_input, similarity_function,
					 verbosity):
	"""The difference between two inputs before and after encoding."""
	before_similarity = similarity_function(first_input, second_input)
	first_encoded = encoder.encode(first_input)
	second_encoded = encoder.encode(second_input)
	after_similarity = SDRSimilarity(first_encoded, second_encoded)

	if verbosity:
		print "Input similarity: ", before_similarity
		print "SDR similarity: ", after_similarity

	return abs(after_similarity-before_similarity)


def encoderError(encoder, similarity_function, input_pairs_source, trials=1000,
				 verbosity=0):
	"""Estimate the 'error' of an encoder.

	Error captures the amount that encoding disrupts pairwise relationships
	between elements in the input space.

	@param encoder An Encoder object that has an encode method for converting
				    inputs into SDRs.

	@param similarity_function A function that takes two inputs and returns a
					scalar between 0 and 1 capturing semantic similarity.

	@param input_pairs_source A _InputPairCreator object.

	@param trials The number of pairs compared to estimate encoder error.

	@param verbosity How much chatter during running.
	"""

	total_error = 0.0

	next_input = input_pairs_source.getNewPair()
	counter = 0

	while next_input and counter < trials:

		first_input, second_input = next_input

		if verbosity:
			print "Samples: ", next_input

		error = singleSampleDiff(encoder, first_input, second_input,
								 similarity_function, verbosity)

		if verbosity:
			print "Error: ", error
			print

		total_error += error
		next_input = input_pairs_source.getNewPair()
		counter += 1

	print "Ran "+str(counter)+" samples."

	avg_error = total_error/counter

	return avg_error


# TEMPORARY - FOR TESTING DURING DEVELOPMENT

from nupic.encoders.scalar import ScalarEncoder

# Demo with ScalarEncoder
print "Testing ScalarEncoder Quality"
maxval = 100.0
minval = -100.0
Nsamples = 1000

encoder = ScalarEncoder(name="scalar", n=14, w=3, minval=minval, maxval=maxval,
                            periodic=True, forced=True)

# Anything more than 1/4 outside the range is completely not similar.
similarity_function = lambda x,y : max(0.0, 1.0-abs(x-y)/(.25*(maxval-minval)))

sample_space = np.random.uniform(minval, maxval, (Nsamples,))
input_pairs_source = InputPairCreatorFromList(sample_space)

err = encoderError(encoder, similarity_function, input_pairs_source)
print "Average error: ",
print err
