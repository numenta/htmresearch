import numpy
from nupic.bindings.math import *


def apply_noise(data, noise):
    """
    Applies noise to a sparse matrix.  Noise should be an integer between 0 and
    100, indicating the percentage of ones in the original input to move.
    The input matrix is modified in-place, and nothing is returned.
    This operation does not affect the sparsity of the matrix, or of any
    individual datapoint.
    """

    for i in range(data.nRows()):
        ones = data.rowNonZeros(i)[0]
        replace_indices = numpy.random.choice(ones, size=int(1.0*len(ones)*noise/100), replace = False)
        for index in replace_indices:
            data[i, index] = 0

        new_indices = numpy.random.choice(data.nCols(), size = int(1.0*len(ones)*noise/100.), replace = False)

        for index in new_indices:
            while data[i, index] == 1:
                index = numpy.random.randint(0, data.nCols())
            data[i, index] = 1


def shuffle_sparse_matrix_and_labels(matrix, labels):
    """
    Shuffles a sparse matrix and set of labels together.
    Resorts to densifying and then re-sparsifying the matrix, for convenience.
    """
    print "Shuffling data"
    new_matrix = matrix.toDense()
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(new_matrix)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(labels)

    print "Data shuffled"
    return SM32(new_matrix), numpy.asarray(labels)

def split_sparse_matrix(matrix, num_categories):
    """
    An analog of numpy.split for our sparse matrix.  If the number of 
    categories does not divide the number of rows in the matrix, all overflow
    is placed in the final bin.

    In the event that there are more categories than rows, all later categories
    are considered to be an empty sparse matrix.
    """
    if matrix.nRows() < num_categories:
        return [matrix.getSlice(i, i+1, 0, matrix.nCols()) for i in range(matrix.nRows())] + [SM32() for i in range(num_categories - matrix.nRows())]
    else:
        inc = matrix.nRows()/num_categories
        divisions = [matrix.getSlice(i*inc, (i+1)*inc, 0, matrix.nCols()) for i in range(num_categories - 1)]

        # Handle the last bin separately.  All overflow goes into it.
        divisions.append(matrix.getSlice((num_categories - 1)*inc, matrix.nRows(), 0, matrix.nCols()))

        return divisions    

def generate_evenly_distributed_data_sparse(dim = 2000, num_active = 40, num_samples = 1000):
	"""
	Generates a set of data drawn from a uniform distribution.  The binning structure from Poirazi & Mel is
	ignored, and all (dim choose num_active) arrangements are possible.  num_negatives samples are put into
	a separate negatives category for output compatibility with generate_data, but are otherwise identical.

	"""
	indices = [numpy.random.choice(dim, size = num_active, replace = False) for i in range(num_samples)]
	data = SM32()
	data.reshape(num_samples, dim)
	for sample, datapoint in enumerate(indices):
		for i in datapoint:
			data[sample, i] = 1.

	return data


def generate_evenly_distributed_data(dim = 2000, num_active = 40, num_samples = 1000, num_negatives = 500):
	"""
	Generates a set of data drawn from a uniform distribution.  The binning structure from Poirazi & Mel is
	ignored, and all (dim choose num_active) arrangements are possible.  num_negatives samples are put into
	a separate negatives category for output compatibility with generate_data, but are otherwise identical.

	"""

	sparse_data = [numpy.random.choice(dim, size = num_active, replace = False) for i in range(num_samples)]
	data = [[0 for i in range(dim)] for i in range(num_samples)]
	for datapoint, sparse_datapoint in zip(data, sparse_data):
		for i in sparse_datapoint:
			datapoint[i] = 1

	negatives = data[:num_negatives]
	positives = data[num_negatives:]
	return positives, negatives

def generate_data(dim = 40, num_samples = 30000, num_bins = 10, sparse = False):
	"""
	Generates data following appendix V of (Poirazi & Mel, 2001).
	Positive and negative examples are drawn from the same distribution, but
	are multiplied by different square matrices, one of them uniform on [-1, 1] and one
	the sum of a uniform matrix and a normal one.  It is assumed that half
	the samples are negative in this case.

	Initially samples of dimension dim are produced, with values in each
	dimension being floats, but they are binned into discrete categories, with
	num_bins bins per dimension.  This binning produces an SDR.  

	"""

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

	if sparse:
		positives = SM32(positives)
		negatives = SM32(negatives)

	return positives, negatives

def generate_phase_1(dim = 40):
	"""
	The first step in creating datapoints in the Poirazi & Mel model.
	This returns a vector of dimension dim, with the last four values set to
	1 and the rest drawn from a normal distribution.

	"""
	phase_1 = numpy.random.normal(0, 1, dim)
	for i in range(dim - 4, dim):
		phase_1[i] = 1.0
	return phase_1

def generate_phase_2(phase_1, dim = 40):
	"""
	The second step in creating datapoints in the Poirazi & Mel model.
	This takes a phase 1 vector, and creates a phase 2 vector where each point
	is the product of four elements of the phase 1 vector, randomly drawn with
	replacement.
	"""
	phase_2 = []
	for i in range(dim):
		indices = [numpy.random.randint(0, dim) for i in range(4)]
		phase_2.append(numpy.prod([phase_1[i] for i in indices]))
	return phase_2

def generate_matrices(dim = 40):
	"""
	Generates the matrices that positive and negative samples are multiplied
	with.  The matrix for positive samples is randomly drawn from a uniform
	distribution, with elements in [-1, 1].  The matrix for negative examples
	is the sum of the positive matrix with a matrix drawn from a normal
	distribution with mean 0 variance 1.

	"""
	positive = numpy.random.uniform(-1, 1, (dim, dim))
	negative = positive + numpy.random.normal(0, 1, (dim, dim))
	return positive, negative

def generate_RF_bins(data, dim = 40, num_bins = 10):
	"""
	Generates bins for the encoder.  Bins are designed to have equal frequency,
	per Poirazi & Mel (2001), which requires reading the data once.
	Bins are represented as the intervals dividing them.
	"""
	intervals = []
	for i in range(dim):
		current_dim_data = [data[x][i] for x in range(len(data))]
		current_dim_data = numpy.sort(current_dim_data)
		intervals.append([current_dim_data[int(len(current_dim_data)*x/num_bins)] for x in range(1, num_bins)])
	return intervals

def bin_number(datapoint, intervals):
	"""
	Given a datapoint and intervals representing bins, returns the number
	represented in binned form, where the bin including the value is
	set to 1 and all others are 0.

	"""
	index = numpy.searchsorted(intervals, datapoint)
	return [0 if index != i else 1 for i in range(len(intervals) + 1)]

def bin_data(data, dim = 40, num_bins = 10):
	"""
	Fully bins the data generated by generate_data, using generate_RF_bins and
	bin_number.

	"""
	intervals = generate_RF_bins(data, dim, num_bins)
	binned_data = [numpy.concatenate([bin_number(data[x][i], intervals[i]) for i in range(len(data[x]))]) for x in range(len(data))]
	return binned_data

if __name__ == "__main__":
	"""
	Generates a set of test data and prints it to data.txt.
	This is only for inspection; normally, data is freshly generated for each
	experiment using the functions in this file.

	"""
	pos, neg = generate_data(num_samples = 30000)
	posdata = [(i, 1) for i in pos]
	negdata = [(i, -1) for i in neg]
	data = posdata + negdata
	#print pos, neg
	#binned_data = bin_data(pos + neg, 40, 10)
	with open("data.txt", "wb") as f:
		for line in data:
			f.write(str(line) + "\n")
