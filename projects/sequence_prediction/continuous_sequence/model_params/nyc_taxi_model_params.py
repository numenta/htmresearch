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

MODEL_PARAMS = {'aggregationInfo': {'days': 0,
                     'fields': [],
                     'hours': 0,
                     'microseconds': 0,
                     'milliseconds': 0,
                     'minutes': 0,
                     'months': 0,
                     'seconds': 0,
                     'weeks': 0,
                     'years': 0},
 'model': 'CLA',
 'modelParams': {
   'anomalyParams': {u'anomalyCacheRecords': None,
                     u'autoDetectThreshold': None,
                     u'autoDetectWaitRecords': None},
   'clParams': {'alpha': 0.005,
                'regionName': 'SDRClassifierRegion',
                'implementation': 'py',
                'steps': '5'},
   'inferenceType': 'TemporalMultiStep',
   'sensorParams': {'encoders': {
        '_classifierInput': {'classifierOnly': True,
                            'clipInput': True,
                            'fieldname': 'passenger_count',
                            'maxval': 40000,
                            'minval': 0,
                            'n': 50 ,
                            'name': '_classifierInput',
                            'type': 'ScalarEncoder',
                            'w': 29},
        u'timeofday':       {'clipInput': True,
                            'fieldname': 'timeofday',
                            'maxval': 1411,
                            'minval': 0,
                            'n': 600,
                            'name': 'timeofday',
                            'type': 'ScalarEncoder',
                            'periodic': True,
                            'verbosity': 0,
                            'w': 29},
        u'dayofweek':       {'clipInput': True,
                            'fieldname': 'dayofweek',
                            'maxval': 7,
                            'minval': 0,
                            'n': 100,
                            'name': 'dayofweek',
                            'type': 'ScalarEncoder',
                            'periodic': True,
                            'verbosity': 0,
                            'w': 29},
        u'passenger_count': {'clipInput': True,
                            'fieldname': 'passenger_count',
                            'maxval': 40000,
                            'minval': 0,
                            'n': 109,
                            'name': 'passenger_count',
                            'type': 'ScalarEncoder',
                            'verbosity': 0,
                            'w': 29}},
        'sensorAutoReset': None,
        'verbosity': 0},
   'spEnable': True,
   'spParams': {'columnCount': 2048,
                'globalInhibition': 1,
                'inputWidth': 0,
                'maxBoost': 1.0,
                'numActiveColumnsPerInhArea': 40,
                'potentialPct': 0.8,
                'seed': 1956,
                'spVerbosity': 1,
                'spatialImp': 'cpp',
                'synPermActiveInc': 0.0001,
                'synPermConnected': 0.5,
                'synPermInactiveDec': 0.0005},
   'tpEnable': True,
   'tpParams': {'activationThreshold': 15,
                'cellsPerColumn': 32,
                'columnCount': 2048,
                'globalDecay': 0.0,
                'initialPerm': 0.21,
                'inputWidth': 2048,
                'maxAge': 0,
                'maxSegmentsPerCell': 128,
                'maxSynapsesPerSegment': 128,
                'minThreshold': 15,
                'newSynapseCount': 32,
                'outputType': 'normal',
                'pamLength': 1,
                'permanenceDec': 0.1,
                'permanenceInc': 0.1,
                'predictedSegmentDecrement': 0.01,
                'seed': 1960,
                'temporalImp': 'tm_py',
                'verbosity': 0},
   'trainSPNetOnlyIfRequested': False},
 'predictAheadTime': None,
 'version': 1}