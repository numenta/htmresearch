import numpy
import matplotlib.pyplot as plt


trainingPasses = 1
outputDir = 'results/'

def SDRsimilarity(SDR1, SDR2):
  return float(len(SDR1 & SDR2)) / float(max(len(SDR1), len(SDR2) ))


def getUnionSDRSimilarityCurve(activeCellsTrace, trainingPasses, sequenceLength, maxSeparation, skipBeginningElements=0):
  
  print activeCellsTrace, len(activeCellsTrace), trainingPasses, sequenceLength, maxSeparation
  print "HERE"
  
  similarityVsSeparation = numpy.zeros((trainingPasses, maxSeparation))
  for rpts in xrange(trainingPasses):
    for sep in xrange(maxSeparation):
      similarity = []
      for i in xrange(rpts*sequenceLength+skipBeginningElements, rpts*sequenceLength+sequenceLength-sep):
        similarity.append(SDRsimilarity(activeCellsTrace[i], activeCellsTrace[i+sep]))

      similarityVsSeparation[rpts, sep] = numpy.mean(similarity)

  return similarityVsSeparation


def plotSDRsimilarityVsTemporalSeparation(similarityVsSeparationBefore, similarityVsSeparationAfter):
  # plot SDR similarity as a function of temporal separation
  f, (axs) = plt.subplots(nrows=2 ,ncols=2)
  # f, (axs) = plt.subplots(nrows=1 ,ncols=1)
  rpt = 0
  ax1 = axs[0,0]
  ax1.plot(similarityVsSeparationBefore[rpt,:],label='Before')
  ax1.plot(similarityVsSeparationAfter[rpt,:],label='After')
  ax1.set_xlabel('Separation in time between SDRs')
  ax1.set_ylabel('SDRs overlap')
  # ax1.set_title('Initial Cycle')
  ax1.set_ylim([0,1])
  ax1.legend(loc='upper right')
  # rpt=4
  # ax2.plot(similarityVsSeparationBefore[rpt,:],label='Before')
  # ax2.plot(similarityVsSeparationAfter[rpt,:],label='After')
  # ax2.set_xlabel('Separation in time between SDRs')
  # ax2.set_ylabel('SDRs overlap')
  # ax2.set_title('Last Cycle')
  # ax2.set_ylim([0,1])
  # ax2.legend(loc='upper right')
  f.savefig('results/UnionSDRoverlapVsTemporalSeparation.eps',format='eps')


def plotSimilarityMatrix(similarityMatrixBefore, similarityMatrixAfter, sequenceLength):
  f, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
  im = ax1.imshow(similarityMatrixBefore[0:sequenceLength, 0:sequenceLength],interpolation="nearest")
  ax1.set_xlabel('Time (steps)')
  ax1.set_ylabel('Time (steps)')
  ax1.set_title('Overlap - Before Learning')

  im = ax2.imshow(similarityMatrixAfter[0:sequenceLength, 0:sequenceLength],interpolation="nearest")
  ax2.set_xlabel('Time (steps)')
  ax2.set_ylabel('Time (steps)')
  ax2.set_title('Overlap - After Learning')
  # cax,kw = mpl.colorbar.make_axes([ax1, ax2])
  # plt.colorbar(im, cax=cax, **kw)
  # plt.tight_layout()
  f.savefig(outputDir+'/UnionSDRoverlapBeforeVsAfterLearning.eps',format='eps')


def calculateSimilarityMatrix(activeCellsTraceBefore, activeCellsTraceAfter, sequenceLength):
  nSteps = sequenceLength # len(activeCellsTraceBefore)
  similarityMatrixBeforeAfter = numpy.zeros((nSteps, nSteps))
  similarityMatrixBefore = numpy.zeros((nSteps, nSteps))
  similarityMatrixAfter = numpy.zeros((nSteps, nSteps))
  
  print len(activeCellsTraceBefore[3] & activeCellsTraceBefore[4])
  
  for i in xrange(nSteps):
    for j in xrange(nSteps):
      similarityMatrixBefore[i,j] = SDRsimilarity(activeCellsTraceBefore[i], activeCellsTraceBefore[j])
      similarityMatrixAfter[i,j] = SDRsimilarity(activeCellsTraceAfter[i], activeCellsTraceAfter[j])
      similarityMatrixBeforeAfter[i,j] = SDRsimilarity(activeCellsTraceBefore[i], activeCellsTraceAfter[j])

  return (similarityMatrixBefore, similarityMatrixAfter, similarityMatrixBeforeAfter)


def plotTPRvsUPROverlap(similarityMatrix, sequenceLength):
  f = plt.figure()
  plt.subplot(2,2,1)
  im = plt.imshow(similarityMatrix[0:sequenceLength, 0:sequenceLength],
                  interpolation="nearest",aspect='auto', vmin=0, vmax=0.3)
  plt.colorbar(im)
  plt.xlabel('UPR over time')
  plt.ylabel('TPR over time')
  plt.title(' Overlap between UPR & TPR')
  f.savefig('results/OverlapTPRvsUPR.eps',format='eps')


def bitLifeVsLearningCycles(activeCellsTrace, numColumns,learningPasses, sequenceLength):
  bitLifeVsLearning = numpy.zeros(learningPasses)
  for i in xrange(learningPasses):
    bitLifeList = []
    bitLifeCounter = numpy.zeros(numColumns)

    for t in xrange(sequenceLength):
      preActiveCells = set(numpy.where(bitLifeCounter>0)[0])
      currentActiveCells = activeCellsTrace[i*sequenceLength+t]
      newActiveCells = list(currentActiveCells - preActiveCells)
      stopActiveCells = list(preActiveCells - currentActiveCells)
      if t == sequenceLength-1:
        stopActiveCells = list(currentActiveCells)
      continuousActiveCells = list(preActiveCells & currentActiveCells)
      bitLifeList += list(bitLifeCounter[stopActiveCells])

      bitLifeCounter[stopActiveCells] = 0
      bitLifeCounter[newActiveCells] = 1
      bitLifeCounter[continuousActiveCells] += 1

    bitLifeVsLearning[i] = numpy.mean(bitLifeList)

  return bitLifeVsLearning


def plotSummaryResults(upBeforeLearning, upDuringLearning, upAfterLearning, learningPasses, sequenceLength, numColumns):
  maxSeparation = 30
  skipBeginningElements = 10
  activeCellsTraceBefore = upBeforeLearning._mmTraces['activeCells'].data
  similarityVsSeparationBefore = getUnionSDRSimilarityCurve(activeCellsTraceBefore,  trainingPasses, sequenceLength,
                                                            maxSeparation, skipBeginningElements)

  activeCellsTraceAfter = upAfterLearning._mmTraces['activeCells'].data
  similarityVsSeparationAfter = getUnionSDRSimilarityCurve(activeCellsTraceAfter,  trainingPasses, sequenceLength,
                                                           maxSeparation, skipBeginningElements)

  plotSDRsimilarityVsTemporalSeparation(similarityVsSeparationBefore, similarityVsSeparationAfter)

  (similarityMatrixBefore, similarityMatrixAfter, similarityMatrixBeforeAfter) = \
    calculateSimilarityMatrix(activeCellsTraceBefore, activeCellsTraceAfter, sequenceLength)

  plotTPRvsUPROverlap(similarityMatrixBeforeAfter, sequenceLength)

  plotSimilarityMatrix(similarityMatrixBefore, similarityMatrixAfter, sequenceLength)


  activeCellsTrace = upDuringLearning._mmTraces["activeCells"].data
  meanBitLifeVsLearning = bitLifeVsLearningCycles(activeCellsTrace, numColumns, learningPasses, sequenceLength)

  numBitsUsed = []
  avgBitLatency = []
  for rpt in xrange(learningPasses):
    allActiveBits = set()
    for i in xrange(sequenceLength):
      allActiveBits |= (set(activeCellsTrace[rpt*sequenceLength+i]))

    bitActiveTime = numpy.ones(numColumns) * -1
    for i in xrange(sequenceLength):
      curActiveCells = list(activeCellsTrace[rpt*sequenceLength+i])
      for j in xrange(len(curActiveCells)):
        if bitActiveTime[curActiveCells[j]] < 0:
          bitActiveTime[curActiveCells[j]] = i

    bitActiveTimeSummary = bitActiveTime[bitActiveTime>0]
    print 'pass ', rpt, ' num bits: ', len(allActiveBits), ' latency : ',numpy.mean(bitActiveTimeSummary)

    numBitsUsed.append(len(allActiveBits))
    avgBitLatency.append(numpy.mean(bitActiveTimeSummary))

  # print "numBitsUsed", numBitsUsed
  # print "avgBitLatency", avgBitLatency
  # print "meanBitLifeVsLearning", meanBitLifeVsLearning
  
  f = plt.figure()
  plt.subplot(2,2,1)
  plt.plot(numBitsUsed)
  plt.xlabel(' learning pass #')
  plt.ylabel(' number of cells in UPR')
  plt.ylim([100,600])
  plt.subplot(2,2,2)
  plt.plot(avgBitLatency)
  plt.xlabel(' learning pass #')
  plt.ylabel(' average latency ')
  plt.ylim([20,25])
  plt.subplot(2,2,3)
  plt.plot(meanBitLifeVsLearning)
  plt.xlabel(' learning pass #')
  plt.ylabel(' average lifespan ')
  plt.ylim([0,15])
  f.savefig('results/SDRpropertyOverLearning.pdf')

