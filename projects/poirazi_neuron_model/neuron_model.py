import random
import numpy

class Neuron(object):
	def __init__(self,
				 size = 10000,
				 num_dendrites = 1000,
				 dendrite_length = 10,
				 dim = 400,
				 nonlinearity = lambda x : x**2,
				 nonlinearity_derivative = lambda x: 2*x,
				 num_replacement = 25):
		self.size = size
		self.num_dendrites = num_dendrites
		self.dendrite_length = dendrite_length
		self.dim = dim
		self.nonlinearity = nonlinearity
		self.nonlinearity_derivative = nonlinearity_derivative
		self.num_replacement = num_replacement
		self.permanences = [0.3 for i in range(self.size)]
		self.perm_dec = 0.02
		self.perm_inc = 0.1

		assert dim <= size
		assert size == num_dendrites*dendrite_length

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


	def train_on_entire_dataset(self, data, labels, temperature):
		replacement_candidates = numpy.random.choice(self.size, self.num_replacement, replace = False)
		substitutes = self.choose_substitute_pool()
		substitute_scores = [0 for x in range(self.num_replacement)]
		candidate_scores = [0 for x in range(self.num_replacement)]

		for datapoint, label in zip(data, labels):
			for i, candidate in enumerate(replacement_candidates):

				# Rule: Thetaij = <Xij*b'(x)*t>
				candidate_branch = candidate / self.dendrite_length
				candidate_id = candidate % self.dendrite_length
				current_score = sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)])
				candidate_scores[i] += datapoint[self.dendrites[candidate_branch][candidate_id]]*self.nonlinearity_derivative(sum([datapoint[self.dendrites[candidate_branch][x]] for x in range(self.dendrite_length)]))*label

		weakest_synapse = replacement_candidates[numpy.argmin(candidate_scores)]
		weakest_synapse_branch = weakest_synapse / self.dendrite_length
		weakest_synapse_id = weakest_synapse % self.dendrite_length
		weakest_synapse_value = self.dendrites[weakest_synapse_branch][weakest_synapse_id]

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


	def HTM_style_train_on_datapoint(self, datapoint, label, decrement_rate = 0.001):
		activations = [sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)]) for i in range(self.num_dendrites)]
		activation = sum(map(self.nonlinearity, activations))
		if activation == label:
			strongest_branch = numpy.argmax(activations)
			for i, neuron in enumerate(self.dendrites[strongest_branch]):
				if datapoint[neuron] == 1:
					self.permanences[strongest_branch*self.dendrite_length + i] = min(self.permanences[strongest_branch*self.dendrite_length + i] + self.perm_inc, 1.0)

		#	for i in range(self.num_dendrites):
				if i == strongest_branch:
					continue
				for x, neuron in enumerate(self.dendrites[i]):
					if datapoint[neuron] == 1:
						self.permanences[i*self.dendrite_length + x] = max(self.permanences[i*self.dendrite_length + x] - self.perm_dec, 0.0)
						if self.permanences[i*self.dendrite_length + x] < 0.2:
							self.dendrites[i][x] = random.randint(0, self.dim - 1)
							self.permanences[i*self.dendrite_length + x] = 0.35	

		elif activation > label:
			# Need to weaken some connections
			strongest_branch = numpy.argmax(activations)
			for i, neuron in enumerate(self.dendrites[strongest_branch]):
				if datapoint[neuron] == 1:
					self.permanences[strongest_branch*self.dendrite_length + i] = max(self.permanences[strongest_branch*self.dendrite_length + i] - self.perm_dec, 0.0)
					if self.permanences[strongest_branch*self.dendrite_length + i] < 0.2:
						self.dendrites[strongest_branch][i] = random.randint(0, self.dim - 1)
						self.permanences[strongest_branch*self.dendrite_length + i] = 0.35

		else:
			# Need to create some new connections
			chosen_branch = numpy.argmax(activations)
			if numpy.mean(self.permanences[chosen_branch*self.dendrite_length:(chosen_branch+1)*self.dendrite_length]) > 0.8:
				branch_permanences = [sum(self.permanences[branch*self.dendrite_length:(branch+1)*self.dendrite_length]) for branch in range(self.num_dendrites)]
				chosen_branch = np.argmin(branch_permanences)
			for i, neuron in enumerate(self.dendrites[chosen_branch]):
				if datapoint[neuron] == 0:
					self.permanences[chosen_branch*self.dendrite_length + i] = max(self.permanences[chosen_branch*self.dendrite_length + i] - self.perm_dec, 0.0)
					if self.permanences[chosen_branch*self.dendrite_length + i] < 0.2:
						finger = random.randint(0, self.dim - 1)
						while datapoint[finger] == 0:
							finger = (finger + 1) % self.dim
						self.dendrites[chosen_branch][i] = finger
						self.permanences[chosen_branch*self.dendrite_length + i] = 0.35



def sigmoid_derivative(x, center = 5, scale = 2):
	numerator = scale*numpy.exp(scale*(x + center))
	denominator = (numpy.exp(scale*x) + numpy.exp(scale*center)) ** 2
	return numerator/denominator



class HTM_Style_Neuron(object):
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
		self.data_dictionary = {}

		assert size == num_dendrites*dendrite_length

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
		current_dendrite = 0
		for datapoint, label in zip(data, labels):
			if label == 1:
				ones = [i for i, d in enumerate(datapoint) if d == 1]
				for i, neuron in enumerate(self.dendrites[current_dendrite]):
					self.dendrites[current_dendrite] = numpy.random.choice(ones, size = self.dendrite_length, replace = False)
					#self.dendrites[current_dendrite][i] = finger
					#self.permanences[current_dendrite*self.dendrite_length + i] = 0.6
				current_dendrite += 1

	def HTM_style_train_on_datapoint(self, datapoint, label):
		activations = [sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)]) for i in range(self.num_dendrites)]
		activation = sum(map(self.nonlinearity, activations))
		if activation == label:
			strongest_branch = numpy.argmax(activations)
			for i, neuron in enumerate(self.dendrites[strongest_branch]):
				if datapoint[neuron] == 1:
					self.permanences[strongest_branch*self.dendrite_length + i] = min(self.permanences[strongest_branch*self.dendrite_length + i] + self.perm_inc, 1.0)

		#	for i in range(self.num_dendrites):
		#		if i == strongest_branch or self.nonlinearity(activations[i]) == 0:
		#			continue
		#		for x, neuron in enumerate(self.dendrites[i]):
		#			if datapoint[neuron] == 1:
		#				self.permanences[i*self.dendrite_length + x] = max(self.permanences[i*self.dendrite_length + x] - self.perm_dec, 0.0)
		#				if self.permanences[i*self.dendrite_length + x] == 0:
		#					self.dendrites[i][x] = random.randint(0, self.dim - 1)
		#					self.permanences[i*self.dendrite_length + x] = 0.35	


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



	def train_on_entire_dataset(self, data, labels):
		replacement_candidates = numpy.random.choice(self.size, self.num_replacement, replace = False)
		substitutes = self.choose_substitute_pool()
		substitute_scores = [0 for x in range(self.num_replacement)]
		candidate_scores = [0 for x in range(self.num_replacement)]

		for datapoint, label in zip(data, labels):
			for i, candidate in enumerate(replacement_candidates):

				# Rule: Thetaij = <Xij*b'(x)*t>
				candidate_branch = candidate / self.dendrite_length
				candidate_id = candidate % self.dendrite_length
				current_score = sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)])
				candidate_scores[i] += datapoint[self.dendrites[candidate_branch][candidate_id]]*self.nonlinearity_derivative(sum([datapoint[self.dendrites[candidate_branch][x]] for x in range(self.dendrite_length)]))*label

		weakest_synapse = replacement_candidates[numpy.argmin(candidate_scores)]
		weakest_synapse_branch = weakest_synapse / self.dendrite_length
		weakest_synapse_id = weakest_synapse % self.dendrite_length
		weakest_synapse_value = self.dendrites[weakest_synapse_branch][weakest_synapse_id]

		for datapoint, label in zip(data, labels):
					for i, substitute in enumerate(substitutes):

						# Rule: Thetaij = <Xij*b'(x)*t>
						self.dendrites[weakest_synapse_branch][weakest_synapse_id] = substitute
						current_score = sum([datapoint[self.dendrites[i][x]] for x in range(self.dendrite_length)])
						substitute_scores[i] += datapoint[self.dendrites[weakest_synapse_branch][weakest_synapse_id]]*self.nonlinearity_derivative(sum([datapoint[self.dendrites[weakest_synapse_branch][x]] for x in range(self.dendrite_length)]))*label
						self.dendrites[weakest_synapse_branch][weakest_synapse_id] = weakest_synapse_value

		strongest_substitute = substitutes[numpy.argmax(substitute_scores)]

		if max(substitute_scores) > min(candidate_scores):
			self.dendrites[weakest_synapse_branch][weakest_synapse_id] = strongest_substitute
