import numpy

def generate_data(dim = 40, num_samples = 30000, num_bins = 10):
	positives, negatives = [], []
	positive, negative = generate_matrices(dim)

	for i in range(num_samples):
		phase_1 = generate_phase_1(dim)
		phase_2 = generate_phase_2(phase_1, dim)


		if i < num_samples/2:
			positives.append(numpy.dot(positive, phase_2))
		else:
			negatives.append(numpy.dot(negative, phase_2))

	binned_data = bin_data(positives + negatives, dim, num_bins)
	positives = binned_data[:len(binned_data)/2]
	negatives = binned_data[len(binned_data)/2:]

	return positives, negatives

def generate_evenly_distributed_data(dim = 2000, num_active = 40, num_samples = 1000, num_negatives = 500):
	sparse_data = [numpy.random.choice(dim, size = num_active, replace = False) for i in range(num_samples)]
	data = [[0 for i in range(dim)] for i in range(num_samples)]
	for datapoint, sparse_datapoint in zip(data, sparse_data):
		for i in sparse_datapoint:
			datapoint[i] = 1

	negatives = data[:num_negatives]
	positives = data[num_negatives:]
	return positives, negatives

def generate_phase_1(dim = 40):
	phase_1 = numpy.random.normal(0, 1, dim)
	for i in range(36, dim):
		phase_1[i] = 1.0
	return phase_1

def generate_phase_2(phase_1, dim = 40):
	phase_2 = []
	for i in range(dim):
		indices = [numpy.random.randint(0, 40) for i in range(4)]
		phase_2.append(numpy.prod([phase_1[i] for i in indices]))
	return phase_2

def generate_matrices(dim = 40):
	positive = numpy.random.uniform(-1, 1, (dim, dim))
	negative = positive + numpy.random.normal(0, 1, (dim, dim))
	return positive, negative

def generate_RF_bins(data, dim = 40, num_bins = 10):
	intervals = []
	for i in range(dim):
		current_dim_data = [data[x][i] for x in range(len(data))]
		current_dim_data = numpy.sort(current_dim_data)
		intervals.append([current_dim_data[int(len(current_dim_data)*x/num_bins)] for x in range(1, num_bins)])
	return intervals

def bin_datapoint(datapoint, intervals):
	index = numpy.searchsorted(intervals, datapoint)
	return [0 if index != i else 1 for i in range(len(intervals) + 1)]

def bin_data(data, dim = 40, num_bins = 10):
	intervals = generate_RF_bins(data, dim, num_bins)
	binned_data = [numpy.concatenate([bin_datapoint(data[x][i], intervals[i]) for i in range(len(data[x]))]) for x in range(len(data))]
	return binned_data

if __name__ == "__main__":
	pos, neg = generate_data(num_samples = 30000)
	posdata = [(i, 1) for i in pos]
	negdata = [(i, -1) for i in neg]
	data = posdata + negdata
	#print pos, neg
	#binned_data = bin_data(pos + neg, 40, 10)
	with open("data.txt", "wb") as f:
		for line in data:
			f.write(str(line) + "\n")
