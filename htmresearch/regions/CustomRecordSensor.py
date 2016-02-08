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
  1. Getting input data from the InputStream can be turned off.
  2. Set the next record manually
  
  """


  @classmethod
  def getSpec(cls):
    ns = super(CustomRecordSensor, cls).getSpec()

    return dict(ns,
                parameters=dict(ns["parameters"],
                                useDataSource=dict(
                                    description='1 if the DataSource should '
                                                'be used as the data '
                                                'source (default 0).',
                                    accessMode='ReadWrite',
                                    dataType='UInt32',
                                    count=1,
                                    constraints='bool'),
                                nextTimestamp=dict(
                                    description="Next timestamp (int) to be "
                                                "processed by the network.",
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
                                    description="Next category (int) to be "
                                                "processed by the network.",
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
    """Get the next record to encode. Includes getting a record
    from the datasource and applying filters. If the filters
    request more data from the datasource continue to get data
    from the datasource until all filters are satisfied.
    This method is separate from compute() so that we can use
    a standalone RecordSensor to get filtered data"""

    foundData = False
    while not foundData:

      # Get the data from the dataSource
      if self.useDataSource:
        data = self.dataSource.getNextRecordDict()
      else:
        data = {
          '_timestamp': None, '_category': [self.nextCategory],
          'label': [self.nextCategory], '_sequenceId': 0, 'y': self.nextValue,
          'x': self.nextTimestamp, '_timestampRecordIdx': None, '_reset': 0
        }

      if not data:
        raise StopIteration("Datasource has no more data")

      # temporary check
      if "_reset" not in data:
        data["_reset"] = 0
      if "_sequenceId" not in data:
        data["_sequenceId"] = 0
      if "_category" not in data:
        data["_category"] = [None]

      if self.verbosity > 0:
        print "RecordSensor got data: %s" % data

      # Apply pre-encoding filters.
      # These filters may modify or add data
      # If a filter needs another record (e.g. a delta filter)
      # it will request another record by returning False and the current record
      # will be skipped (but will still be given to all filters)
      #
      # We have to be very careful about resets. A filter may add a reset,
      # but other filters should not see the added reset, each filter sees
      # the original reset value, and we keep track of whether any filter
      # adds a reset.
      foundData = True
      if len(self.preEncodingFilters) > 0:
        originalReset = data['_reset']
        actualReset = originalReset
        for f in self.preEncodingFilters:
          # if filter needs more data, it returns False
          result = f.process(data)
          foundData = foundData and result
          actualReset = actualReset or data['_reset']
          data['_reset'] = originalReset
        data['_reset'] = actualReset

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
