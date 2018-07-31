# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from htmresearch.frameworks.specific_timing.timing_adtm import TimingADTM

Sequence1 = (('A', 5), ('B', 8), ('C', 12), ('D', 16))
Sequence2 = (('X', 5), ('B', 8), ('C', 12), ('Y', 16))
Sequence3 = (('A', 5), ('X', 8), ('C', 12), ('D', 16))
Sequence4 = (('A', 4), ('B', 8), ('C', 12), ('D', 14))
Sequence5 = (('A', 5), ('B', 9), ('C', 12), ('D', 16))

Sequence6 = (('A', 1), ('B', 2), ('C', 4), ('D', 5), ('E', 3), ('F', 7), ('G', 6), ('H', 9))  # long 1x
Sequence7 = (('A', 2), ('B', 4), ('C', 8), ('D', 10), ('E', 6), ('F', 14), ('G', 12), ('H', 18))  # long 0.5x
Sequence8 = (('A', 0.5), ('B', 1), ('C', 2), ('D', 2.5), ('E', 1.5), ('F', 3.5), ('G', 3), ('H', 4.5))  # long 2x
Sequence9 = (('A', 0.25), ('B', 0.5), ('C', 1), ('D', 1.25), ('E', 0.75), ('F', 1.75), ('G', 1.5), ('H', 2.25))  # 4x
Sequence10 = (('A', 4), ('B', 8), ('C', 16), ('D', 20), ('E', 12), ('F', 28), ('G', 24), ('H', 36))  # long 0.25x

# Sequence11 = (('I', 1), ('J', 2), ('K', 4), ('L', 5), ('M', 3), ('N', 7), ('O', 6), ('P', 9))
# Sequence12 = (('I', 2), ('J', 4), ('K', 8), ('L', 10), ('M', 6), ('N', 14), ('O', 12), ('P', 18))
# Sequence13 = (('A', 2), ('B', 3), ('C', 6), ('D', 8), ('E', 4), ('F', 9), ('G', 5), ('H', 1))


def testAny(trainSeqs, testSeqs, numIter):
  """

  :param trainSeqs: list of lists of(feature, timestamp) tuples i.e. [(('A', 5), ('B', 8)), (('A', 1), ('B', 3))]
  :param testSeqs: list of lists of(feature, timestamp) tuples i.e. [(('A', 5), ('B', 8)), (('A', 1), ('B', 3))]
  :param numIter: Number of times to train set of sequences in trainSeqs
  :return: active, basally predicted, apically predicted and predicted cells for
           each timestamp of each sequence in testSeqs
  """
  tempoADTM = TimingADTM(numColumns=2048,
                         numActiveCells=39,
                         numTimeColumns=1024,
                         numActiveTimeCells=19,
                         numTimeSteps=20)

  for _ in range(numIter):

    for item in enumerate(trainSeqs):
      tempoADTM.learn(trainSeq=item[1], numIter=1)

  res = []
  for item in enumerate(testSeqs):
    tempoADTM.infer(testSeq=item[1])
    res.append(tempoADTM.debugResults())

  return res


'Goal 0, test 0.0'
# results = testAny(trainSeqs=[Sequence1, Sequence2], testSeqs=[Sequence1, Sequence2], numIter=4)

'Goal 0, test 0.1'
# results = testAny(trainSeqs=[Sequence1], testSeqs=[Sequence1, Sequence3], numIter=1)

'Goal 1, test 1.0'
# results = testAny(trainSeqs=[Sequence1, Sequence4], testSeqs=[Sequence1, Sequence4], numIter=3)

'Goal 1, test 1.1 (FAIL)'
# results = testAny(trainSeqs=[Sequence1], testSeqs=[Sequence1, Sequence5], numIter=1)

'Goal 2, test 2.1'
results = testAny(trainSeqs=[Sequence6], testSeqs=[Sequence6, Sequence7, Sequence8, Sequence9, Sequence10], numIter=1)

