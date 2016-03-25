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

import request from 'superagent';

const DATASET_REQUEST_URL = '/fluent/datasets';

export default function ListDatasetAction(context) {
  return new Promise((resolve, reject) => {
    request
      .get(DATASET_REQUEST_URL)
      .set('Accept', 'application/json')
      .set('Access-Control-Allow-Origin', '*')
      .end((error, results) => {
        if (error) {
          reject(error);
        } else {
          context.dispatch('LOAD_DATASET_LIST', results.body);
          resolve(results.body);
        }
      });
  });
}
