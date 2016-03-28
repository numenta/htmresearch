// Copyright © 2016, Numenta, Inc.  Unless you have purchased from
// Numenta, Inc. a separate commercial license for this software code, the
// following terms and conditions apply:
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero Public License version 3 as published by the Free
// Software Foundation.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
//
// You should have received a copy of the GNU Affero Public License along with
// this program.  If not, see http://www.gnu.org/licenses.
//
// http://numenta.org/licenses/

export default {
  CioWordFingerprint: {
    label: 'CioWordFingerprint',
    description: 'the Cortical.io API gives us one fingerprint per word, from which we create one (sparsified) union SDR to represent each document'
  },
  CioDocumentFingerprint: {
    label: 'CioDocumentFingerprint',
    description: 'the Cortical.io API gives us one fingerprint per document.'
  },
  Keywords: {
    label: 'Keywords',
    description: 'encoding is a random SDR for each word, and the model looks for exact-matching SDRs.'
  },
  HTM_sensor_knn: {
    label: 'Sensor-kNN',
    description: 'encoding is a fingerprint for each word, and the model looks in the kNN for overlapping SDRs.'
  },
  HTM_sensor_tm_knn: {
    label: 'Sensor-TM-kNN',
    description: 'encoding is a fingerprint for each word, which is then run through temporal memory and into the kNN, where the model looks for overlapping SDRs.'
  },
  HTM_sensor_simple_tp_knn: {
    label: 'Sensor-simpleUP-kNN',
    description: 'each window of 10 words is represented by a union SDR of the 10 individual fingerprints, and the model looks in the kNN for overlapping SDRs.'
  }
};
