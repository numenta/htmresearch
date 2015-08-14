# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

"""This file implements the SequenceClassifier."""

import array
import itertools

import numpy


# This determines how large one of the duty cycles must get before each of the
# duty cycles are updated to the current iteration.
# This must be less than float32 size since storage is float32 size
DUTY_CYCLE_UPDATE_INTERVAL = numpy.finfo(numpy.float32).max / ( 2**20 )

g_debugPrefix = "SequenceClassifier"



def _pFormatArray(array_, fmt="%.2f"):
  """Return a string with pretty-print of a numpy array using the given format
  for each element"""
  return "[ " + " ".join(fmt % x for x in array_) + " ]"



class BitHistory(object):
  """Class to store an activationPattern  bit history."""

  __slots__ = ("_classifier", "_id", "_stats", "_lastUpdate",
               "_learnIteration", "_version")

  __VERSION__ = 1


  def __init__(self, classifier, bitNum):
    """Constructor for bit history.

    Parameters:
    ---------------------------------------------------------------------
    classifier:    instance of the SequenceClassifier that owns us
    bitNum:        activation pattern bit number this history is for,
                        used only for debug messages
    """
    # Store reference to the classifier
    self._classifier = classifier

    # Form our "id"
    self._id = bitNum

    # Dictionary of bucket entries. The key is the bucket index, the
    # value is the dutyCycle, which is the rolling average of the duty cycle
    self._stats = array.array("f")

    # lastUpdate is the iteration number of the last time it was updated.
    self._lastUpdate = None

    # The bit's learning iteration. This is updated each time store() gets
    # called on this bit.
    self._learnIteration = 0

    # Set the version to the latest version.
    # This is used for serialization/deserialization
    self._version = BitHistory.__VERSION__


  def store(self, iteration, bucketIdx):
    """Store a new item in our history.

    This gets called for a bit whenever it is active and learning is enabled

    Parameters:
    --------------------------------------------------------------------
    iteration:  the learning iteration number, which is only incremented
                  when learning is enabled
    bucketIdx:  the bucket index to store

    Save duty cycle by normalizing it to the same iteration as
    the rest of the duty cycles which is lastUpdate.

    This is done to speed up computation in inference since all of the duty
    cycles can now be scaled by a single number.

    The duty cycle is brought up to the current iteration only at inference and
    only when one of the duty cycles gets too large (to avoid overflow to
    larger data type) since the ratios between the duty cycles are what is
    important. As long as all of the duty cycles are at the same iteration
    their ratio is the same as it would be for any other iteration, because the
    update is simply a multiplication by a scalar.
    """

    # If lastTotalUpdate has not been set, set it to the current iteration.
    if self._lastUpdate is None:
      self._lastUpdate = iteration
    # Get the duty cycle stored for this bucket.
    statsLen = len(self._stats) - 1
    if bucketIdx > statsLen:
      self._stats.extend(itertools.repeat(0.0, bucketIdx - statsLen))

    # Update it now.
    # duty cycle n steps ago is dc{-n}
    # duty cycle for current iteration is (1-alpha)*dc{-n}*(1-alpha)**(n)+alpha
    dc = self._stats[bucketIdx]

    # To get the duty cycle from n iterations ago that when updated to the
    # current iteration would equal the dc of the current iteration we simply
    # divide the duty cycle by (1-alpha)**(n). This results in the formula
    # dc'{-n} = dc{-n} + alpha/(1-alpha)**n where the apostrophe symbol is used
    # to denote that this is the new duty cycle at that iteration. This is
    # equivalent to the duty cycle dc{-n}
    denom = ((1.0 - self._classifier.alpha) **
                                  (iteration - self._lastUpdate))
    if denom > 0:
      dcNew = dc + (self._classifier.alpha / denom)

    # This is to prevent errors associated with inf rescale if too large
    if denom == 0 or dcNew > DUTY_CYCLE_UPDATE_INTERVAL:
      exp =  (1.0 - self._classifier.alpha) ** (iteration-self._lastUpdate)
      for (bucketIdxT, dcT) in enumerate(self._stats):
        dcT *= exp
        self._stats[bucketIdxT] = dcT

      # Reset time since last update
      self._lastUpdate = iteration

      # Add alpha since now exponent is 0
      dc = self._stats[bucketIdx] + self._classifier.alpha
    else:
      dc = dcNew

    self._stats[bucketIdx] = dc
    if self._classifier.verbosity >= 2:
      print "updated DC for %s, bucket %d to %f" % (self._id, bucketIdx, dc)


  def infer(self, iteration, votes):
    """Look up and return the votes for each bucketIdx for this bit.

    Parameters:
    --------------------------------------------------------------------
    iteration:  the learning iteration number, which is only incremented
                  when learning is enabled
    votes:      a numpy array, initialized to all 0's, that should be filled
                  in with the votes for each bucket. The vote for bucket index N
                  should go into votes[N].
    """
    # Place the duty cycle into the votes and update the running total for
    # normalization
    total = 0
    for (bucketIdx, dc) in enumerate(self._stats):
    # Not updating to current iteration since we are normalizing anyway
      if dc > 0.0:
        votes[bucketIdx] = dc
        total += dc

    # Experiment... try normalizing the votes from each bit
    if total > 0:
      votes /= total
    if self._classifier.verbosity >= 2:
      print "bucket votes for %s:" % (self._id), _pFormatArray(votes)



class SequenceClassifier(object):
  """
  A Sequence classifier accepts a binary input from the level below (the
  "activationPattern") and information from the sensor and encoders (the
  "classification") describing the input to the system at that time step.

  When learning, for every bit in activation pattern, it records a history of the
  classification each time that bit was active. The history is weighted so that
  more recent activity has a bigger impact than older activity. The alpha
  parameter controls this weighting.

  For inference, it takes an ensemble approach. For every active bit in the
  activationPattern, it looks up the most likely classification(s) from the
  history stored for that bit and then votes across these to get the resulting
  classification(s).
  """

  __VERSION__ = 1


  def __init__(self, alpha=0.001, actValueAlpha=0.3, verbosity=0):
    """Constructor for the Sequence classifier.

    Parameters:
    ---------------------------------------------------------------------
    alpha:    The alpha used to compute running averages of the bucket duty
              cycles for each activation pattern bit. A lower alpha results
              in longer term memory.
    actValueAlpha:  
              The alpha used to compute running averages of each 
              bucketIdx actual values.
    verbosity: verbosity level, can be 0, 1, or 2
    """
    # Save constructor args
    self.alpha = alpha
    self.actValueAlpha = actValueAlpha
    self.verbosity = verbosity

    # Init learn iteration index
    self._learnIteration = 0

    # This contains the offset between the recordNum (provided by caller) and
    #  learnIteration (internal only, always starts at 0).
    self._recordNumMinusLearnIteration = None

    # These are the bit histories. Each one is a BitHistory instance, stored in
    # this dict, where the key is 'bit'. The 'bit' is the index of the
    # bit in the activation pattern.
    self._activeBitHistory = dict()

    # This contains the value of the highest bucket index we've ever seen
    # It is used to pre-allocate fixed size arrays that hold the weights of
    # each bucket index during inference
    self._maxBucketIdx = 0

    # This keeps track of the actual value to use for each bucket index. We
    # start with 1 bucket, no actual value so that the first infer has something
    # to return
    self._actualValues = [None]

    # Set the version to the latest version.
    # This is used for serialization/deserialization
    self._version = SequenceClassifier.__VERSION__


  def compute(self, recordNum, patternNZ, classification, learn, infer):
    """
    Process one input sample.

    Parameters:
    --------------------------------------------------------------------
    recordNum:  Record number of this input pattern. Record numbers should
                normally increase sequentially by 1 each time unless there
                are missing records in the dataset. Knowing this information
                insures that we don't get confused by missing records.
    patternNZ:  list of the active indices from the output below
    classification: dict of the classification information:
                      bucketIdx: index of the encoder bucket
                      actValue:  actual value going into the encoder
    learn:      if true, learn this sample
    infer:      if true, perform inference

    retval:     dict containing inference results. The entry 'probabilities'
                is an array containing the relative likelihood for
                each bucketIdx starting from bucketIdx 0.

                There is also an entry containing the average actual value to
                use for each bucket. The key is 'actualValues'.

                for example:
                  {'probabilities': [0.1, 0.3, 0.2, 0.7],
                   'actualValues': [1.5, 3,5, 5,5, 7.6],
                  }
    """

    # Save the offset between recordNum and learnIteration if this is the first
    #  compute
    if self._recordNumMinusLearnIteration is None:
      self._recordNumMinusLearnIteration = recordNum - self._learnIteration

    # Update the learn iteration
    self._learnIteration = recordNum - self._recordNumMinusLearnIteration


    if self.verbosity >= 1:
      print "\n%s: compute" % g_debugPrefix
      print "  recordNum:", recordNum
      print "  learnIteration:", self._learnIteration
      print "  patternNZ (%d):" % len(patternNZ), patternNZ
      print "  classificationIn:", classification

    # To allow multi-class classification, we need to be able to run leaning
    # without inference being on. So initialize retval outside 
    # of the inference block.
    retval = None
    
    # ------------------------------------------------------------------------
    # Inference:
    # For each active bit in the activationPattern, get the classification
    # votes
    if infer:
      # Return value dict. For buckets which we don't have an actual value
      # for yet, just plug in any valid actual value. It doesn't matter what
      # we use because that bucket won't have non-zero likelihood anyways.

      # NOTE: we shouldn't use any knowledge of the classification input 
      # during inference. So put the default value to 0.
      defaultValue = 0
      actValues = [x if x is not None else defaultValue
                   for x in self._actualValues]
      retval = {"actualValues": actValues}

      # Accumulate bucket index votes and actValues into these arrays
      sumVotes = numpy.zeros(self._maxBucketIdx+1)
      bitVotes = numpy.zeros(self._maxBucketIdx+1)

      # For each active bit, get the votes
      for bit in patternNZ:
        history = self._activeBitHistory.get(bit, None)
        if history is None:
          continue

        bitVotes.fill(0)
        history.infer(iteration=self._learnIteration, votes=bitVotes)

        sumVotes += bitVotes

      # Return the votes for each bucket, normalized
      total = sumVotes.sum()
      if total > 0:
        sumVotes /= total
      else:
        # If all buckets have zero probability then simply make all of the
        # buckets equally likely. There is no actual prediction for this
        # timestep so any of the possible predictions are just as good.
        if sumVotes.size > 0:
          sumVotes = numpy.ones(sumVotes.shape)
          sumVotes /= sumVotes.size

      retval["probabilities"] = sumVotes

    # ------------------------------------------------------------------------
    # Learning:
    # For each active bit in the activationPattern, store the classification
    # info. If the bucketIdx is None, we can't learn. This can happen when the
    # field is missing in a specific record.
    if learn and classification["bucketIdx"] is not None:

      # Get classification info
      bucketIdx = classification["bucketIdx"]
      actValue = classification["actValue"]

      # Update maxBucketIndex
      self._maxBucketIdx = max(self._maxBucketIdx, bucketIdx)

      # Update rolling average of actual values if it's a scalar. If it's
      # not, it must be a category, in which case each bucket only ever
      # sees one category so we don't need a running average.
      while self._maxBucketIdx > len(self._actualValues)-1:
        self._actualValues.append(None)
      if self._actualValues[bucketIdx] is None:
        self._actualValues[bucketIdx] = actValue
      else:
        if isinstance(actValue, int) or isinstance(actValue, float):
          self._actualValues[bucketIdx] = \
                  (1.0 - self.actValueAlpha) * self._actualValues[bucketIdx] \
                   + self.actValueAlpha * actValue
        else:
          self._actualValues[bucketIdx] = actValue

      # Train pattern.
      # Store classification info for each active bit.
      for bit in patternNZ:

        # Get the history structure for this bit 
        history = self._activeBitHistory.get(bit, None)
        if history is None:
          history = self._activeBitHistory[bit] = BitHistory(self, bitNum=bit)

        # Store new sample
        history.store(iteration=self._learnIteration,
                      bucketIdx=bucketIdx)

    # ------------------------------------------------------------------------
    # Verbose print
    if infer and self.verbosity >= 1:
      print "  inference: combined bucket likelihoods:"
      print "    actual bucket values:", retval["actualValues"]
      votes = retval["probabilities"]
      print "    probabilities: %s" %_pFormatArray(votes)
      bestBucketIdx = votes.argmax()
      print "      most likely bucket idx: %d, value: %s" % (bestBucketIdx,
                          retval["actualValues"][bestBucketIdx])
      print

    return retval


