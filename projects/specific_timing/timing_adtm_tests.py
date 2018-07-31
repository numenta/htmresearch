import numpy as np
from projects.specific_timing.timing_adtm import TimingADTM



Sequence1 = (('A', 5), ('B', 8), ('C', 12), ('D', 16))
Sequence2 = (('X', 5), ('B', 8), ('C', 12), ('Y', 16))
Sequence3 = (('A', 5), ('X', 8), ('C', 12), ('D', 16))
Sequence4 = (('A', 4), ('B', 8), ('C', 12), ('D', 14))
Sequence5 = (('A', 5), ('B', 9), ('C', 12), ('D', 16))

Sequence6 = (('A', 3), ('B', 4), ('C', 2), ('D', 1))  # 1x
Sequence7 = (('A', 6), ('B', 8), ('C', 4), ('D', 2))  # 0.5x
Sequence8 = (('A', 1.5), ('B', 2), ('C', 1), ('D', 0.5))  # 2x
Sequence9 = (('A', 12), ('B', 16), ('C', 8), ('D', 4))  # 0.25x
Sequence10 = (('A', 0.75), ('B', 1), ('C', 0.5), ('D', 0.25))  # 4x

Sequence11 = (('A', 4), ('B', 6), ('C', 3), ('D', 2))  # random other seq
Sequence12 = (('A', 6), ('B', 4), ('C', 3), ('D', 5))  # tempo distractor
Sequence13 = (('E', 4), ('F', 6), ('G', 3), ('H', 2))  #

Sequence14 = (('A', 9), ('B', 12), ('C', 6), ('D', 3))


def testAny(trainSeqs, testSeqs, numIter):
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
results = testAny(trainSeqs=[Sequence1, Sequence2], testSeqs=[Sequence1, Sequence2], numIter=4)


