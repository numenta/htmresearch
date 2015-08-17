# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.ion()
rcParams.update({'figure.autolayout': True})


def plotResult():
  resultTM = np.load('result/reberSequenceTM.npz')
  resultLSTM = np.load('result/reberSequenceLSTM.npz')

  plt.figure()
  plt.hold(True)
  plt.subplot(2,2,1)
  plt.semilogx(resultTM['trainSeqN'], 100*np.mean(resultTM['correctRateAll'],1),'-*',label='TM')
  plt.semilogx(resultLSTM['trainSeqN'], 100*np.mean(resultLSTM['correctRateAll'],1),'-s',label='LSTM')
  plt.legend()
  plt.xlabel(' Training Sequence Number')
  plt.ylabel(' Hit Rate (Best Match) (%)')

  plt.subplot(2,2,4)
  plt.semilogx(resultTM['trainSeqN'], 100*np.mean(resultTM['missRateAll'],1),'-*',label='TM')
  plt.semilogx(resultLSTM['trainSeqN'], 100*np.mean(resultLSTM['missRateAll'],1),'-*',label='LSTM')
  plt.legend()
  plt.xlabel(' Training Sequence Number')
  plt.ylabel(' Miss Rate (%)')

  plt.subplot(2,2,3)
  plt.semilogx(resultTM['trainSeqN'], 100*np.mean(resultTM['fpRateAll'],1),'-*',label='TM')
  plt.semilogx(resultLSTM['trainSeqN'], 100*np.mean(resultLSTM['fpRateAll'],1),'-*',label='LSTM')
  plt.legend()
  plt.xlabel(' Training Sequence Number')
  plt.ylabel(' False Positive Rate (%)')
  plt.savefig('result/ReberSequence_CompareTM&LSTMperformance.pdf')

if __name__ == "__main__":
  plotResult()
