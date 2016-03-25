/* -----------------------------------------------------------------------------
 * Copyright © 2015, Numenta, Inc. Unless you have purchased from
 * Numenta, Inc. a separate commercial license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero Public License version 3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for
 * more details.
 *
 * You should have received a copy of the GNU Affero Public License along with
 * this program. If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * -------------------------------------------------------------------------- */

import request from 'superagent';

const API_HOST = '/fluent';

export default (context, payload) => {
  let {dataset, model, query} = payload;
  return new Promise((resolve, reject) => {
    if (model) {
      // Request results for given model
      let url = `${API_HOST}/${model}`
      if (dataset) {
        url += `/${dataset}`;
      }
      request
        .post(url)
        .send(query)
        .set('Accept', 'application/json')
        .set('Access-Control-Allow-Origin', '*')
        .end((error, results) => {
          if (error) {
            context.dispatch('SEARCH_RECEIVED_DATA', {
              query, dataset, model
            });
            console.error(error);
          } else {
            context.dispatch('SEARCH_RECEIVED_DATA', {
              query, dataset, model, results: results.body
            });
            resolve(results.body);
          }
        }
      );
    } else {
      // No model given, just update the query
      context.dispatch('SEARCH_RECEIVED_DATA', {query});
      resolve();
    }
  });
};
