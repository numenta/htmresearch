# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
from lateral_pooler import LateralPooler
import numpy as np

class LateralPoolerWrapper(LateralPooler):
  """
  This is a wrapper for the lateral pooler, in order to provide an interface 
  similar to the original spatial pooler.
  """
  def __init__(self,
          inputDimensions            = (32**2, 1),
          columnDimensions           = (64**2, 1),
          potentialRadius            = 16,
          potentialPct               = 0.5,
          globalInhibition           = False,
          localAreaDensity           = -1.0,
          numActiveColumnsPerInhArea = 10.0,
          stimulusThreshold          = 0,
          synPermInactiveDec         = 0.008,
          synPermActiveInc           = 0.05,
          synPermConnected           = 0.10,
          minPctOverlapDutyCycle     = 0.001,
          dutyCyclePeriod            = 1000,
          boostStrength              = 100.0,
          seed                       = -1,
          spVerbosity                = 0,
          wrapAround                 = True):



    if numActiveColumnsPerInhArea < 0.:
      numActiveColumnsPerInhArea = localAreaDensity*columnDimensions[0]

    super_args = {
        "input_size"            : inputDimensions[0], 
        "output_size"           : columnDimensions[0], 
        "code_weight"           : numActiveColumnsPerInhArea, 
        "seed"                  : seed,
        "learning_rate"         : synPermActiveInc,
        "inc_dec_ratio"         : synPermActiveInc/synPermInactiveDec,
        "smoothing_period"      : dutyCyclePeriod, 
        "boost_strength"        : boostStrength,
        "boost_strength_hidden" : boostStrength,
        "permanence_threshold"  : synPermConnected
    }
    super(LateralPoolerWrapper, self).__init__(**super_args)
              


  def compute(self, inputVector, learn, activeArray):
    """
    This method resembles the primary public method of the SpatialPooler class. 
    It takes a input vector and outputs the indices of the active columns. If 'learn' 
    is set to True, this method also performs weight updates and updates to the activity 
    statistics according to the respective methods implemented below.
    """
    m = self.input_size
    X = inputVector.reshape((m,1))
    Y = self.encode(X)

    if learn:
      self.update_connections_online(X, Y)

    active_units = np.where(Y[:,0]==1.)[0]
    
    activeArray[active_units] = 1.

    return active_units


  def getPermanence(self, columnIndex, permanence):
    """
    Returns the permanence values for a given column. ``permanence`` size
    must match the number of inputs.
    
    :param columnIndex: (int) column index to get permanence for.
    :param permanence: (list) will be overwritten with permanences. 
    """
    assert(columnIndex < self.output_size)
    permanence[:] = self.feedforward[columnIndex]


  def getColumnDimensions(self):
    """
    :returns: (iter) the dimensions of the columns in the region
    """
    return (self.output_size, 1)

  def getInputDimensions(self):
    """
    :returns: (iter) the dimensions of the input vector
    """
    return (self.input_size, 1)


  def getNumColumns(self):
    """
    :returns: (int) the total number of columns
    """
    return self.output_size

  def getNumInputs(self):
    """
    :returns: (int) the total number of inputs.
    """
    return self.input_size


  def getConnectedSynapses(self, columnIndex, connectedSynapses):
    """
    :param connectedSynapses: (list) will be overwritten
    :returns: (iter) the connected synapses for a given column.
              ``connectedSynapses`` size must match the number of inputs"""
    assert(columnIndex < self.output_size)
    connectedSynapses[:] = np.greater(self.feedforward[columnIndex], self.permanence_threshold, dtype=float)


  def getConnectedCounts(self, connectedCounts):
    """
    :param connectedCounts: (list) will be overwritten
    :returns: (int) the number of connected synapses for all columns.
              ``connectedCounts`` size must match the number of columns.
    """
    connectedCounts[:] = np.sum( (self.feedforward < self.permanence_threshold).astype(int), axis=1)


  def getActiveDutyCycles(self, activeDutyCycles):
    """
    Gets the activity duty cycles for all columns. Input list will be 
    overwritten.
    
    :param activeDutyCycles: (list) size must match number of columns. 
    """
    activeDutyCycles[:] = self.avg_activity_units[:]


  def getBoostFactors(self, boostFactors):
    """
    Gets the boost factors for all columns. Input list will be overwritten.

    :param boostFactors: (list) size must match number of columns. 
    """
    boostFactors[:] = self.boostfactor[:,0]


  def getOverlaps(self):
    """
    :returns: (iter) the overlap score for each column.
    """
    return self._scores[:,0]

  @property
  def _activeDutyCycles(self):
    return self.avg_activity_units

  @property
  def _numColumns(self):
    return self.output_size

  @property
  def _localAreaDensity(self):
    """
    :returns: (float) the local area density. Returns a value less than 0 if 
              parameter is unused.
    """
    return self.sparsity

  @property
  def _dutyCyclePeriod(self):
    return self.smoothing_period

