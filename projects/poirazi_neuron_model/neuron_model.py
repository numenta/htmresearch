import random
import numpy
import copy
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from nupic.bindings.math import *

numpy.set_printoptions(threshold=numpy.inf)

def sigmoid_derivative(x, center = 5, scale = 2):
	"""
	The derivative of a sigmoid f(x) = 1/(1 + exp(2 * (5 - x)))), which can be
	used in Poirazi & Mel style training.  This sigmoid is effectively a 
	differentiable threshold function with a threshold of center.  The 
	steepness is given by scale.
	"""
	numerator = scale*numpy.exp(scale*(x + center))
	denominator = (numpy.exp(scale*x) + numpy.exp(scale*center)) ** 2
	return numerator/denominator

def power_nonlinearity(power):
	def l(activations):
		original_activations = copy.deepcopy(activations)
		for i in range(power - 1):
			activations.elementNZMultiply(original_activations)
	return l

def threshold_nonlinearity(threshold):
	def l(activations):
		activations.threshold(threshold)
	return l


class Matrix_Neuron(object):
	def __init__(self,
				 size = 10000,
				 num_dendrites = 1000,
				 dendrite_length = 10,
				 dim = 400,
				 nonlinearity = threshold_nonlinearity(6),
				 nonlinearity_derivative = sigmoid_derivative,
				 num_replacement = 25,
				 initial_permanence = 0.5,
				 permanence_threshold = 0.15,
				 perm_dec = 0.1,
				 perm_inc = 0.05):
		self.size = size
		self.num_dendrites = num_dendrites
		self.dendrite_length = dendrite_length
		self.dim = dim
		self.nonlinearity = nonlinearity
		self.nonlinearity_derivative = nonlinearity_derivative
		self.num_replacement = num_replacement
		self.initial_permanence = initial_permanence
		self.permanence_threshold = permanence_threshold
		self.perm_dec = perm_dec
		self.perm_inc = perm_inc
		self.initialize_dendrites()
		self.initialize_permanences()

	def initialize_dendrites(self):
		"""
		Initialize all the dendrites of the neuron to a set of random connections
		"""
		# Wipe any preexisting connections by creating a new connection matrix
		self.dendrites = SM32()
		self.dendrites.reshape(self.dim, self.num_dendrites)

		for row in range(self.num_dendrites):
			synapses = numpy.random.choice(self.dim, self.dendrite_length, replace = False)
			for synapse in synapses:
				self.dendrites[synapse, row] = 1


	def initialize_permanences(self):
		self.permanences = copy.deepcopy(self.dendrites)
		self.permanences = self.dendrites*0.3

	def calculate_activation(self, datapoint):
		"""
		Only for a single datapoint
		"""

		activations = datapoint * self.dendrites
		self.nonlinearity(activations)
		return activations.sum()

	def choose_substitute_pool(self):
		candidates = numpy.random.choice(self.dim, self.num_replacement)
		return candidates

	def calculate_on_entire_dataset(self, data):
		activations = data * self.dendrites
		self.nonlinearity(activations)
		return activations.rowSums()


	def HTM_style_initialize_on_data(self, data, labels):
		"""
		Uses a style of initialization inspired by the temporal memory.  When a new positive example is found,
		a dendrite is chosen and a number of synapses are created to the example.

		This works intelligently with an amount of data larger than the number of available dendrites.
		In this case, data is clustered, and then similar datapoints are allotted to shared dendrites,
		with as many overlapping bits as possible chosen.  In practice, it is still better to simply
		allocate enough dendrites to have one per datapoint, but this method at least allows initialization
		to work on larger amounts of data.
		"""
		current_dendrite = 0
		self.dendrites = SM32()
		self.dendrites.reshape(self.dim, self.num_dendrites)

		# We want to avoid training on any negative examples
		data = copy.copy(data)
		data.deleteRows([i for i, v in enumerate(labels) if v != 1])

		if data.nRows() > self.num_dendrites:
			print "Neuron using clustering to initialize dendrites"
			data = data.toDense()
			model = AgglomerativeClustering(n_clusters = self.num_dendrites, affinity = "manhattan", linkage = "average")
			clusters = model.fit_predict(data)
			multisets = [[Counter(), []] for i in range(self.num_dendrites)]
			sparse_data = [[i for i, d in enumerate(datapoint) if d == 1] for datapoint in data]

			for datapoint, cluster in zip(sparse_data, clusters):
				multisets[cluster][0] = multisets[cluster][0] + Counter(datapoint)
				multisets[cluster][1].append(set(datapoint))

			for i, multiset in enumerate(multisets):
				shared_elements = set(map(lambda x: x[0], filter(lambda x: x[1] > 1, multiset[0].most_common(self.dendrite_length))))
				dendrite_connections = shared_elements
				while len(shared_elements) < self.dendrite_length:
					most_distant_point = multiset[1][numpy.argmin([len(dendrite_connections.intersection(point)) for point in multiset[1]])]
					new_connection = random.sample(most_distant_point - dendrite_connections, 1)[0]
					dendrite_connections.add(new_connection)

				for synapse in dendrite_connections:
					self.dendrites[synapse, current_dendrite] = 1.
				current_dendrite += 1

		else:
			for i in range(data.nRows()):
				ones = data.rowNonZeros(i)[0]
				dendrite_connections = numpy.random.choice(ones, size = self.dendrite_length, replace = False)
				for synapse in dendrite_connections:
					self.dendrites[synapse, current_dendrite] = 1.

				current_dendrite += 1

		self.initialize_permanences()

	def HTM_style_train_on_data(self, data, labels):
		for i in range(data.nRows()):
			self.HTM_style_train_on_datapoint(data.getSlice(i, i+1, 0, data.nCols()), labels[i])

	def HTM_style_train_on_datapoint(self, datapoint, label):
		"""
		Run a version of permanence-based training on a datapoint.  Due to the fixed dendrite count and dendrite length,
		we are forced to more efficiently use each synapse, deleting synapses and resetting them if they are not found useful.
		"""
		activations = datapoint * self.dendrites
		self.nonlinearity(activations)

		#activations will quite likely still be sparse if using a threshold nonlinearity, so want to keep it sparse
		activation = numpy.sign(activations.sum())


		if label >= 1 and activation >= 1:
			strongest_branch = activations.rowMax(0)[0]
			datapoint.transpose()
			inc_vector = self.dendrites.getSlice(0, self.dim, strongest_branch, strongest_branch + 1) * self.perm_inc
			inc_vector.elementNZMultiply(datapoint)
			dec_vector = self.dendrites.getSlice(0, self.dim, strongest_branch, strongest_branch + 1) * self.perm_dec
			dec_vector.elementNZMultiply(1 - datapoint)



			self.permanences.setSlice(0, strongest_branch, self.permanences.getSlice(0, self.dim, strongest_branch, strongest_branch + 1) + inc_vector - dec_vector)

			positions, scores = self.permanences.colNonZeros(strongest_branch)[0], self.permanences.colNonZeros(strongest_branch)[1]

			for position, score in zip(positions, scores):
				if score < self.permanence_threshold:
					self.dendrites[position, strongest_branch] = 0
					self.permanences[position, strongest_branch] = 0
					new_connection = random.sample(set(datapoint.colNonZeros(0)[0]) - set(self.dendrites.colNonZeros(strongest_branch)[0]), 1)[0]
					self.dendrites[new_connection, strongest_branch] = 1.
					self.permanences[new_connection, strongest_branch] = self.initial_permanence


		elif label < 1 and activation >= 1:
			# Need to weaken some connections
			strongest_branch = activations.rowMax(0)[0]

			dec_vector = self.dendrites.getSlice(0, self.dim, strongest_branch, strongest_branch + 1) * self.perm_dec
			datapoint.transpose()
			dec_vector.elementNZMultiply(datapoint)
			self.permanences.setSlice(0, strongest_branch, self.permanences.getSlice(0, self.dim, strongest_branch, strongest_branch + 1) - dec_vector)


		elif label >= 1 and activation < 1:
			# Need to create some new connections
			weakest_branch = numpy.argmin(self.permanences.colSums())
			if numpy.mean(self.permanences.getCol(weakest_branch)) < self.permanence_threshold:
				self.permanences.setColToZero(weakest_branch)
				self.dendrites.setColToZero(weakest_branch)

				ones = datapoint.colNonZeros(0)[0]
				dendrite_connections = numpy.random.choice(ones, size = self.dendrite_length, replace = False)
				for synapse in dendrite_connections:
					self.dendrites[synapse, weakest_branch] = 1.
					self.permanences[synapse, weakest_branch] = self.initial_permanence

		#print self.permanences.min()



class Neuron(object):
	def __init__(self,
				 size = 10000,
				 num_dendrites = 1000,
				 dendrite_length = 10,
				 dim = 400,
				 nonlinearity = lambda x : 1 if x > 8 else 0,
				 nonlinearity_derivative = sigmoid_derivative,
				 num_replacement = 25):
		self.size = size
		self.num_dendrites = num_dendrites
		self.dendrite_length = dendrite_length
		self.dim = dim
		self.nonlinearity = nonlinearity
		self.nonlinearity_derivative = nonlinearity_derivative
		self.num_replacement = num_replacement
		self.permanences = [0.3 for i in range(self.size)]
		self.perm_dec = 0.1
		self.perm_inc = 0.05

		self.initialize_dendrites()

	def initialize_dendrites(self):
		self.dendrites = [[random.randint(0, self.dim - 1) for x in range(self.dendrite_length)] for i in range(self.num_dendrites)]

	def calculate_activation(self, datapoint):
		"""
		Only for a single datapoint
		"""
		activations = [sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)]) for i in range(self.num_dendrites)]
		return sum(map(self.nonlinearity, activations))

	def choose_substitute_pool(self):
		candidates = numpy.random.choice(self.dim, self.num_replacement)
		return candidates

	def calculate_on_entire_dataset(self, data):
		activations = []
		for datapoint in data:
			activations.append(self.calculate_activation(datapoint))

		return activations


	def HTM_style_initialize_on_data(self, data, labels):
		"""
		Uses a style of initialization inspired by the temporal memory.  When a new positive example is found,
		a dendrite is chosen and a number of synapses are created to the example.

		TODO: Make this work intelligently with more than one example per dendrite.
		Plan: when a new pattern is found and all dendrites are storing at least one pattern,
		find the learned pattern with the most overlap with the new one, and store all
		and set that dendrite to store all overlapping bits, plus a selection from each pattern
		if there is insufficient overlap.

		To deal with multiple patterns, cluster the positive examples into num_dendrites
		clusters, and use a dendrite for each.
		"""
		current_dendrite = 0

		filtered_data = [d for l, d in zip(labels, data) if l == 1]
		if len(filtered_data) > self.num_dendrites:
			print "Neuron using clustering to initialize dendrites"
			model = AgglomerativeClustering(n_clusters = self.num_dendrites, affinity = "manhattan", linkage = "average")
			clusters = model.fit_predict(filtered_data)
			multisets = [[Counter(), []] for i in range(self.num_dendrites)]
			sparse_data = [[i for i, d in enumerate(datapoint) if d == 1] for datapoint in filtered_data]

			for datapoint, cluster in zip(sparse_data, clusters):
				multisets[cluster][0] = multisets[cluster][0] + Counter(datapoint)
				multisets[cluster][1].append(set(datapoint))

			for i, multiset in enumerate(multisets):
				shared_elements = set(filter(lambda x: x[1] > 1, multiset[0].most_common(self.dendrite_length)))
				dendrite_connections = shared_elements
				while len(shared_elements) < self.dendrite_length:
					most_distant_point = multiset[1][numpy.argmin([len(dendrite_connections.intersection(point)) for point in multiset[1]])]
					dendrite_connections.add(numpy.random.choice(dendrite_connections - most_distant_point))
				self.dendrites[i] = list(dendrite_connections)

		else:
			for datapoint, label in zip(data, labels):
				if label == 1:
					ones = [i for i, d in enumerate(datapoint) if d == 1]
					for i, neuron in enumerate(self.dendrites[current_dendrite]):
						self.dendrites[current_dendrite] = numpy.random.choice(ones, size = self.dendrite_length, replace = False)
					current_dendrite += 1



	def HTM_style_train_on_datapoint(self, datapoint, label):
		"""
		Run a version of permanence-based training on a datapoint.  Due to the fixed dendrite count and dendrite length,
		we are forced to more efficiently use each synapse, deleting synapses and resetting them if they are not found useful.
		"""
		activations = [sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)]) for i in range(self.num_dendrites)]
		activation = sum(map(self.nonlinearity, activations))
		if activation == label:
			strongest_branch = numpy.argmax(activations)
			for i, neuron in enumerate(self.dendrites[strongest_branch]):
				if datapoint[neuron] == 1:
					self.permanences[strongest_branch*self.dendrite_length + i] = min(self.permanences[strongest_branch*self.dendrite_length + i] + self.perm_inc, 1.0)

		elif activation > label:
			# Need to weaken some connections
			for b in range(self.num_dendrites):
				if self.nonlinearity(activations[b]) < 1:
					continue
				for i, neuron in enumerate(self.dendrites[b]):
					if datapoint[neuron] == 1:
						self.permanences[b*self.dendrite_length + i] = max(self.permanences[b*self.dendrite_length + i] - self.perm_dec, 0.0)
						if self.permanences[b*self.dendrite_length + i] == 0:
							self.dendrites[b][i] = random.randint(0, self.dim - 1)
							self.permanences[b*self.dendrite_length + i] = 0.35
		else:
			# Need to create some new connections
			strongest_branch = numpy.argmax(activations)
			if numpy.mean(self.permanences[strongest_branch*self.dendrite_length:(strongest_branch+1)*self.dendrite_length]) > 0.6:
				strongest_branch = random.randint(0, self.num_dendrites - 1)
			for i, neuron in enumerate(self.dendrites[strongest_branch]):
				if datapoint[neuron] == 0:
					self.permanences[strongest_branch*self.dendrite_length + i] = max(self.permanences[strongest_branch*self.dendrite_length + i] - self.perm_dec, 0.0)
					if self.permanences[strongest_branch*self.dendrite_length + i] == 0:
						finger = random.randint(0, self.dim - 1)
						while datapoint[finger] == 0:
							finger = (finger + 1) % self.dim
						self.dendrites[strongest_branch][i] = finger
						self.permanences[strongest_branch*self.dendrite_length + i] = 0.35



	def train_on_entire_dataset(self, data, labels, temperature):
		"""
		Run training based on the training rule in (Poirazi & Mel, 2001), equation 6.
		Due to some lack of clarity about their formulation, this rule has been substantially
		simplified, with the terms g'(y) and g(y) removed.  The final rule is thus
		Thetaij = <Xij*b'(x)*t>, with the lowest-theta synapse replaced by the highest-
		theta replacement candidate if it is an improvement, subject to temperature.
		Note that this method of training is exceptionally slow relative to using the
		HTM-style learning rule, requiring up to 96000 passes per neuron, according
		to Poirazi & Mel.
		"""
		replacement_candidates = numpy.random.choice(self.size, self.num_replacement, replace = False)
		substitutes = self.choose_substitute_pool()
		substitute_scores = [0 for x in range(self.num_replacement)]
		candidate_scores = [0 for x in range(self.num_replacement)]


		# Find the worst synapse in the replacement set.
		for datapoint, label in zip(data, labels):
			for i, candidate in enumerate(replacement_candidates):

				# Rule: Thetaij = <Xij*b'(x)*t>
				candidate_branch = candidate / self.dendrite_length
				candidate_id = candidate % self.dendrite_length
				current_score = sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)])
				candidate_scores[i] += datapoint[self.dendrites[candidate_branch][candidate_id]]*self.nonlinearity_derivative(sum([datapoint[self.dendrites[candidate_branch][x]] for x in range(self.dendrite_length)]))*label

		#Store the location and value of the worst synapse for the next step.
		weakest_synapse = replacement_candidates[numpy.argmin(candidate_scores)]
		weakest_synapse_branch = weakest_synapse / self.dendrite_length
		weakest_synapse_id = weakest_synapse % self.dendrite_length
		weakest_synapse_value = self.dendrites[weakest_synapse_branch][weakest_synapse_id]


		# Calculate the best candidate, now that we know which dendrite to
		# look at.
		for datapoint, label in zip(data, labels):
					for i, substitute in enumerate(substitutes):

						# Rule: Thetaij = <Xij*b'(x)*t>
						self.dendrites[weakest_synapse_branch][weakest_synapse_id] = substitute
						current_score = sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)])
						substitute_scores[i] += datapoint[self.dendrites[weakest_synapse_branch][weakest_synapse_id]]*self.nonlinearity_derivative(sum([datapoint[self.dendrites[weakest_synapse_branch][x]] for x in range(self.dendrite_length)]))*label
						self.dendrites[weakest_synapse_branch][weakest_synapse_id] = weakest_synapse_value

		strongest_substitute = substitutes[numpy.argmax(substitute_scores)]

		if max(substitute_scores) > min(candidate_scores) or random.random() < 1/(1 + numpy.exp((numpy.sqrt(min(candidate_scores)) - numpy.sqrt(max(substitute_scores)))/temperature)):
			self.dendrites[weakest_synapse_branch][weakest_synapse_id] = strongest_substitute
