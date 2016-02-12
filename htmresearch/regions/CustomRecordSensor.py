#!/usr/bin/env python

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-16, Numenta, Inc.  Unless you have an agreement
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

from nupic.regions.RecordSensor import RecordSensor



class CustomRecordSensor(RecordSensor):
  """
  A slightly modified version of the nupic.regions.RecordSensor.

  This region offers the following additional functionalities:
  1. Ability to stop using the RecordStream as the source of input data.
  2. Ability to set the next record timestamp, value, and category manually.
  """


  @classmethod
  def getSpec(cls):
    """
    This extends the specs of the base class RecordSensor with a couple of 
    params for increased flexibility.
    """
    ns = super(CustomRecordSensor, cls).getSpec()
    # Extend RecordSensor params
    return dict(ns,
                parameters=dict(ns["parameters"],
                                useDataSource=dict(
                                  description='1 if the RecordStream should '
                                              'be used as the input source '
                                              '(default is 0).',
                                  accessMode='ReadWrite',
                                  dataType='UInt32',
                                  count=1,
                                  constraints='bool'),
                                nextTimestamp=dict(
                                  description="Next timestamp to be processed "
                                              "by the network.",
                                  dataType="Int32",
                                  accessMode="ReadWrite",
                                  count=1,
                                  constraints=""),
                                nextValue=dict(
                                  description="Next value to be processed by "
                                              "the network.",
                                  dataType="Real32",
                                  accessMode="ReadWrite",
                                  count=1,
                                  constraints=""),
                                nextCategory=dict(
                                  description="Next category to be processed "
                                              "by the network.",
                                  dataType="Int32",
                                  accessMode="ReadWrite",
                                  count=1,
                                  constraints="")
                                )
                )


  def __init__(self, verbosity=0, numCategories=1):

    super(CustomRecordSensor, self).__init__(verbosity, numCategories)
    self.useDataSource = True
    self.nextTimestamp = None
    self.nextValue = None
    self.nextCategory = None


  def getNextRecord(self):

    if self.useDataSource:
      super(CustomRecordSensor, self).getNextRecord()
    else:
      data = {
        '_timestamp': None, '_category': [self.nextCategory],
        'label': [self.nextCategory], '_sequenceId': 0, 'y': self.nextValue,
        'x': self.nextTimestamp, '_timestampRecordIdx': None, '_reset': 0
      }

      data, filterHasEnoughData = super(CustomRecordSensor, self).applyFilters(
        data)

      if not filterHasEnoughData:
        raise ValueError("One of the filters need more data but data is being "
                         "fed manually with the CustomRecordSensor.py region. "
                         "Consider using only filters that don't need "
                         "additional data (i.e. avoid delta filter) or use "
                         "the regular RecordSensor.py region.")

      self.lastRecord = data
      return data


  def setParameter(self, parameterName, index, parameterValue):
    """
      Set the value of a Spec parameter. Most parameters are handled
      automatically by PyRegion's parameter set mechanism. The ones that need
      special treatment are explicitly handled here.
    """
    if parameterName == 'topDownMode':
      self.topDownMode = parameterValue
    elif parameterName == 'useDataSource':
      self.useDataSource = int(parameterValue)
    elif parameterName == 'nextValue':
      self.nextValue = parameterValue
    elif parameterName == 'nextTimestamp':
      self.nextTimestamp = parameterValue
    elif parameterName == 'nextCategory':
      self.nextCategory = parameterValue
    else:
      raise Exception('Unknown parameter: ' + parameterName)
