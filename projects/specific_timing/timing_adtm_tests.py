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

Sequence11 = (('I', 1), ('J', 2), ('K', 4), ('L', 5), ('M', 3), ('N', 7), ('O', 6), ('P', 9))
Sequence12 = (('I', 2), ('J', 4), ('K', 8), ('L', 10), ('M', 6), ('N', 14), ('O', 12), ('P', 18))
Sequence13 = (('A', 2), ('B', 3), ('C', 6), ('D', 8), ('E', 4), ('F', 9), ('G', 5), ('H', 1))

Sequence21 = (('A', 6), ('B', 8), ('C', 16), ('D', 18), ('E', 14), ('F', 18), ('G', 10))
Sequence22 = (('A', 2), ('B', 1), ('C', 6), ('D', 3), ('E', 5), ('F', 1), ('G', 4))
Sequence23 = (('A', 3), ('B', 4), ('C', 8), ('D', 9), ('E', 7), ('F', 9), ('G', 5))

Sequence24 = (('A', 12), ('B', 10), ('C', 16), ('D', 18), ('E', 14), ('F', 18), ('G', 10))
Sequence25 = (('A', 5), ('B', 0), ('C', 17), ('D', 3), ('E', 5), ('F', 1), ('G', 4))
Sequence26 = (('A', 4), ('B', 0), ('C', 3), ('D', 2), ('E', 4), ('F', 1), ('G', 3))
Sequence27 = (('A', 6), ('B', 5), ('C', 8), ('D', 9), ('E', 7), ('F', 9), ('G', 5))

Sequence28 = (('A', 12), ('B', 16), ('C', 4), ('D', 18), ('E', 14), ('F', 18), ('G', 10))
Sequence29 = (('A', 0), ('B', 0), ('C', 17), ('D', 3), ('E', 5), ('F', 1), ('G', 7))
Sequence30 = (('A', 4), ('B', 3), ('C', 19), ('D', 2), ('E', 5), ('F', 1), ('G', 3))
Sequence31 = (('A', 6), ('B', 8), ('C', 2), ('D', 9), ('E', 7), ('F', 9), ('G', 5))

Sequence32 = (('A', 12), ('B', 16), ('C', 8), ('D', 18), ('E', 14), ('F', 18), ('G', 10))
Sequence33 = (('A', 0), ('B', 11), ('C', 2), ('D', 3), ('E', 5), ('F', 1), ('G', 7))
Sequence34 = (('A', 4), ('B', 13), ('C', 3), ('D', 2), ('E', 5), ('F', 1), ('G', 3))
Sequence35 = (('A', 6), ('B', 8), ('C', 4), ('D', 9), ('E', 7), ('F', 9), ('G', 5))



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

    for sequence in trainSeqs:
      tempoADTM.learn(trainSeq=sequence, numIter=1)

  res = []
  for sequence in testSeqs:
    tempoADTM.infer(testSeq=sequence)
    res.append(tempoADTM.debugResults())

  return res

test_results = {}

print 'Goal 0, test 0.0'
test_results['test 0.0'] = testAny(trainSeqs=[Sequence1, Sequence2],
                                   testSeqs=[Sequence1, Sequence2],
                                   numIter=4)
print '----------------------------------------------------------------------------------------'

print 'Goal 0, test 0.1'
test_results['test 0.1'] = testAny(trainSeqs=[Sequence1],
                                   testSeqs=[Sequence1, Sequence3],
                                   numIter=1)
print '----------------------------------------------------------------------------------------'

print 'Goal 1, test 1.0'
test_results['test 1.0'] = testAny(trainSeqs=[Sequence1, Sequence4],
                                   testSeqs=[Sequence1, Sequence4],
                                   numIter=3)
print '----------------------------------------------------------------------------------------'

print 'Goal 1, test 1.1 (FAIL)'
test_results['test 1.1'] = testAny(trainSeqs=[Sequence1],
                                   testSeqs=[Sequence1, Sequence5],
                                   numIter=1)
print '----------------------------------------------------------------------------------------'

print 'Goal 2, test 2.1'
test_results['test 2.1'] = testAny(trainSeqs=[Sequence6],
                                   testSeqs=[Sequence6, Sequence7, Sequence8, Sequence9, Sequence10],
                                   numIter=1)
print '----------------------------------------------------------------------------------------'

'Higher order tempo tests'

print 'Goal 2, test 2.2'
test_results['test 2.2'] = testAny(trainSeqs=[Sequence6, Sequence11],
                                   testSeqs=[Sequence6, Sequence7, Sequence8, Sequence9, Sequence10, Sequence11, Sequence12],
                                   numIter=3)
print '----------------------------------------------------------------------------------------'

print 'Goal 2, test 2.3'
test_results['test 2.3'] = testAny(trainSeqs=[Sequence6, Sequence13],
                                   testSeqs=[Sequence6, Sequence7, Sequence10, Sequence13],
                                   numIter=3)
print '----------------------------------------------------------------------------------------'

print 'constructed tricky case 1: for each time-step, equal votes to slow down vs. speed up tempo'
test_results['tricky 1'] = testAny(trainSeqs=[Sequence21, Sequence22],
                                   testSeqs=[Sequence21, Sequence22, Sequence23],
                                   numIter=3)
print '----------------------------------------------------------------------------------------'

print 'constructed tricky case 2: for each time-step, more votes to move in wrong direction'
test_results['tricky 2'] = testAny(trainSeqs=[Sequence24, Sequence25, Sequence26],
                                   testSeqs=[Sequence24, Sequence25, Sequence26, Sequence27],
                                   numIter=10)

print '----------------------------------------------------------------------------------------'

print 'constructed tricky case 3: for each time-step, more votes to move in wrong direction - including post hard-rest'
test_results['tricky 3'] = testAny(trainSeqs=[Sequence28, Sequence29, Sequence30],
                                   testSeqs=[Sequence28, Sequence29, Sequence30, Sequence31],
                                   numIter=10)

print '----------------------------------------------------------------------------------------'

print 'constructed tricky case 4: at second time-step, more votes to move in wrong direction -- makes inertia useless '
test_results['tricky 4'] = testAny(trainSeqs=[Sequence32, Sequence33, Sequence34],
                                   testSeqs=[Sequence32, Sequence33, Sequence34, Sequence35],
                                   numIter=10)

