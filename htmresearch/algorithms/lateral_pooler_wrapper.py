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


class SpatialPooler(LateralPooler):
    """
    This is a wrapper for the lateral pooler, in order to provide an interface 
    similar to the original spatial pooler.
    """
    def __init__(self,
            inputDimensions  =(32**2, 1),
            columnDimensions =(64**2, 1),
            potentialRadius  =16,
            potentialPct     =0.5,
            globalInhibition=False,
            localAreaDensity=-1.0,
            numActiveColumnsPerInhArea=10.0,
            stimulusThreshold=0,
            synPermInactiveDec=0.008,
            synPermActiveInc=0.05,
            synPermConnected=0.10,
            minPctOverlapDutyCycle=0.001,
            dutyCyclePeriod=1000,
            boostStrength=0.0,
            seed=-1,
            spVerbosity=0,
            wrapAround=True):

        assert(inputDimensions[1] == 1 and columnDimensions[1] == 1)

        super_args = {
            "inputSize"           : inputDimensions[0], 
            "outputSize"          : columnDimensions[0], 
            "codeWeight"          : numActiveColumnsPerInhArea, 
            "seed"                : seed,
            "learningRate"        : synPermActiveInc,
            "incDecRatio"         : synPermActiveInc/synPermInactiveDec,
            "smoothingPeriod"     : dutyCyclePeriod, 
            "boostStrength"       : boostStrength,
            "boostStrengthHidden" : boostStrength,
            "permanenceThreshold" : synPermConnected
        }
        super(SpatialPooler, self).__init__(**super_args)
                


    def compute(self, input_vector, learn, active_array):
    """
    This method resembles the primary public method of the SpatialPooler class. 
    It takes a input vector and outputs the indices of the active columns. If 'learn' 
    is set to True, this method also performs weight updates and updates to the activity 
    statistics according to the respective methods implemented below.
    """
        m = self.input_size
        X = input_vector.reshape((m,1))
        Y = self.encode(X)

        if learn:
        self.update_connections(X, Y)

        active_units = np.where(Y[:,0]==1.)[0]
        active_array[active_units] = 1.

        return active_units

    def getPermanence(self, columnIndex, permanence):
        assert(columnIndex < self.output_size)
        permanence[:] = self.feedforward[columnIndex]




