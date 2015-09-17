# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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


from nupic.frameworks.opf.clamodel import CLAModel
from nupic.frameworks.opf.opfutils import (InferenceType)

class CLAModel_custom(CLAModel):

  def __init__(self,
               **kwargs):

    super(CLAModel_custom, self).__init__(**kwargs)

    self._spLearningEnabled = True
    self._tpLearningEnabled = True

  # override _spCompute
  def _spCompute(self):
    sp = self._getSPRegion()
    if sp is None:
      return

    sp.setParameter('topDownMode', False)
    sp.setParameter('inferenceMode', self.isInferenceEnabled())
    sp.setParameter('learningMode', self._spLearningEnabled)
    sp.prepareInputs()
    sp.compute()

  # overide _tpCompute
  def _tpCompute(self):
    tp = self._getTPRegion()
    if tp is None:
      return

    if (self.getInferenceType() == InferenceType.TemporalAnomaly or
        self._isReconstructionModel()):
      topDownCompute = True
    else:
      topDownCompute = False

    tp = self._getTPRegion()
    tp.setParameter('topDownMode', topDownCompute)
    tp.setParameter('inferenceMode', self.isInferenceEnabled())
    tp.setParameter('learningMode', self._tpLearningEnabled)
    tp.prepareInputs()
    tp.compute()
