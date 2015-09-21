MODEL_PARAMS = \
  {
    'aggregationInfo': {
      'days': 0,
      'fields': [],
      'hours': 0,
      'microseconds': 0,
      'milliseconds': 0,
      'minutes': 0,
      'months': 0,
      'seconds': 0,
      'weeks': 0,
      'years': 0
      },
    'model': 'CLA',
    'modelParams': {
      'anomalyParams': {
        u'anomalyCacheRecords': None,
        u'autoDetectThreshold': None,
        u'autoDetectWaitRecords': None
        },
      'clParams': {
        'alpha': 0.01962508905154251,
        'clVerbosity': 0,
        'regionName': 'CLAClassifierRegion',
        'steps': '1'
        },
      'inferenceType': 'TemporalAnomaly',
      'sensorParams': {
        'encoders': {
          '_classifierInput': {
            'classifierOnly': True,
            'clipInput': True,
            'fieldname': 'metric_value',
            'maxval': None,
            'minval': None,
            'n': 134,
            'name': '_classifierInput',
            'type': 'ScalarEncoder',
            'w': 21
            },
          u'metric_value': {
            'clipInput': True,
            'fieldname': 'metric_value',
            'maxval': None,
            'minval': None,
            'n': 134,
            'name': 'metric_value',
            'type': 'ScalarEncoder',
            'w': 21
            }
          },
        'sensorAutoReset': None,
        'verbosity': 0
        },
      'spEnable': True,
      "spParams": {
        "spVerbosity": 0,
        "spatialImp": "cpp",
        "globalInhibition": 1,
        "columnCount": 2048,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "potentialPct": 0.8,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.0001,
        "synPermInactiveDec": 0.0005,
        "maxBoost": 1.0
      },
      'tpEnable': True,
      "tpParams": {
        "columnCount": 2048,
        "activationThreshold": 13,
        "pamLength": 3,
        "cellsPerColumn": 32,
        "permanenceInc": 0.10000000000000001,
        "minThreshold": 10,
        "verbosity": 0,
        "maxSynapsesPerSegment": 32,
        "outputType": "normal",
        "globalDecay": 0.0,
        "initialPerm": 0.20999999999999999,
        "permanenceDec": 0.10000000000000001,
        "seed": 1960,
        "maxAge": 0,
        "newSynapseCount": 20,
        "maxSegmentsPerCell": 128,
        "temporalImp": "cpp",
        "inputWidth": 2048
      },
      'trainSPNetOnlyIfRequested': False
      },
    'predictAheadTime': None,
    'version': 1
    }
