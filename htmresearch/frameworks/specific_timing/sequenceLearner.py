import numpy as np
import string
from nupic.encoders import ScalarEncoder
from itertools import izip as zip, count
from htmresearch.algorithms.apical_dependent_temporal_memory import ApicalDependentSequenceMemory as TM


class DoubleADTM(object):

    """
    Use the apical dependent sequence memory for learning sequences in time
    """

    def __init__(self, num_columns, num_active_cells, num_time_columns, num_active_time_cells, num_time_steps):

        self.adtm = TM(columnCount=num_columns,
                       apicalInputSize=num_time_columns,
                       cellsPerColumn=32,
                       activationThreshold=13,
                       reducedBasalThreshold=10,
                       initialPermanence=0.50,
                       connectedPermanence=0.50,
                       minThreshold=10,
                       sampleSize=20,
                       permanenceIncrement=0.1,
                       permanenceDecrement=0.1,
                       basalPredictedSegmentDecrement=0.01,
                       apicalPredictedSegmentDecrement=0.0,
                       maxSynapsesPerSegment=-1,
                       seed=42)

        self.numColumns = num_columns
        self.numActiveCells = num_active_cells
        self.numTimeColumns = num_time_columns
        self.numActiveTimeCells = num_active_time_cells
        self.numTimeSteps = num_time_steps

        self.letters = list(string.ascii_uppercase)
        self.letterIndices = self._encode_letters()
        self.letterIndexArray = np.array(self.letterIndices)
        self.timeIndices = self._encode_time()

        self.results = dict()
        self.results['active_cells'] = []
        self.results['predicted_cells'] = []
        self.results['basal_predicted_cells'] = []
        self.results['apical_predicted_cells'] = []

    def reset_results(self):

        self.results = dict()
        self.results['active_cells'] = []
        self.results['predicted_cells'] = []
        self.results['basal_predicted_cells'] = []
        self.results['apical_predicted_cells'] = []

    def learn(self, train_seq, num_iter):

        for _ in range(num_iter):

            prev_apical_input = self.timeIndices[0]
            for jv in range(len(train_seq)):
                active_columns = self.letterIndices[self.letters.index(train_seq[jv][0])]
                apical_input = self.timeIndices[train_seq[jv][1]]

                self.adtm.compute(active_columns,
                                  apicalInput=prev_apical_input,
                                  apicalGrowthCandidates=None,
                                  learn=True)
                self.adtm.compute(active_columns,
                                  apicalInput=apical_input,
                                  apicalGrowthCandidates=None,
                                  learn=True)

                prev_apical_input = apical_input

            self.adtm.reset()
        print('{:<30s}{:<10s}'.format('Train Sequence:', train_seq))

    def infer(self, test_seq):

        self.reset_results()
        prev_apical_input = self.timeIndices[0]
        for jv in range(len(test_seq)):
            active_columns = self.letterIndices[self.letters.index(test_seq[jv][0])]
            apical_input = self.timeIndices[test_seq[jv][1]]

            self.adtm.compute(active_columns,
                              apicalInput=prev_apical_input,
                              apicalGrowthCandidates=None,
                              learn=False)
            self.results['active_cells'].append(self.adtm.getActiveCells())

            self.adtm.compute(active_columns,
                              apicalInput=apical_input,
                              apicalGrowthCandidates=None,
                              learn=False)
            self.results['basal_predicted_cells'].append(self.adtm.getNextBasalPredictedCells())
            self.results['apical_predicted_cells'].append(self.adtm.getNextApicalPredictedCells())
            self.results['predicted_cells'].append(self.adtm.getNextPredictedCells())

            prev_apical_input = apical_input

        print('{:<30s}{:<10s}'.format('Test Sequence:', test_seq))
        print('{:<30s}{:<10s}'.format('--------------', '--------------'))
        self.display_results()

        print('{:<30s}{:<10s}'.format('~~~~~~~~~~~~~~', '~~~~~~~~~~~~~~'))

        self.adtm.reset()

    def display_results(self):

        result_lengths = {k: [len(i) for i in self.results[k]] for k in self.results}
        result_letters = {k: self.letter_converter(self.results[k]) for k in self.results}

        sort_order = ['active_cells', 'basal_predicted_cells', 'apical_predicted_cells', 'predicted_cells']

        for k in sort_order:
            print('{:<30s}{:<10s}'.format(k, map(lambda x, y: (x, y), result_lengths[k], result_letters[k])))

    def letter_converter(self, results_key):

        converted_letters = []

        for c in range(len(results_key)):
            column_idx = [int(i / self.adtm.cellsPerColumn) for i in results_key[c]]

            if not column_idx:
                converted_letters.append(['-'])
            else:
                converted_letters.append(
                    [self.letters[i] for i in np.unique(np.where(self.letterIndexArray == np.unique(column_idx))[0])])

        return converted_letters

    def _encode_time(self):

        time_encoder = ScalarEncoder(n=self.numTimeColumns, w=self.numActiveTimeCells, minval=0,
                                     maxval=self.numTimeSteps, forced=True)

        time_array = np.zeros((self.numTimeSteps, self.numTimeColumns))
        time_indices = []
        for k in range(self.numTimeSteps):
            time_array[k, :] = time_encoder.encode(k)
            idx_times = [i for i, j in zip(count(), time_array[k]) if j == 1]
            time_indices.append(idx_times)

        return time_indices

    def _encode_letters(self):

        letter_encoder = ScalarEncoder(n=self.numColumns, w=self.numActiveCells, minval=0, maxval=25)

        num_letters = np.shape(self.letters)[0]
        letter_array = np.zeros((num_letters, self.numColumns))
        letter_indices = []
        for k in range(num_letters):
            letter_array[k, :] = letter_encoder.encode(k)
            idx_letters = [i for i, j in zip(count(), letter_array[k]) if j == 1]
            letter_indices.append(idx_letters)

        return letter_indices









