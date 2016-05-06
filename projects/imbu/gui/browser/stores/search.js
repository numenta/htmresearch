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

import BaseStore from 'fluxible/addons/BaseStore';

/**
  Imbu search store. Used to query and access Imbu search results.
*/
export default class SearchStore extends BaseStore {

  static storeName = 'SearchStore';

  static handlers = {
    SEARCH_QUERY_MODEL: '_handleQueryModel',
    SEARCH_RECEIVED_DATA: '_handleReceivedData',
    SEARCH_RECEIVED_DATA_ERROR: '_handleReceivedData',
    SEARCH_CLEAR_DATA: '_handleClearData'
  };

  constructor(dispatcher) {
    super(dispatcher);
    // Text used to query
    this.query = null;
    // Last query results
    this.results = new Map();
    // Past queries
    this.history = new Set();
  }

  /**
   * @return {string} current query
   */
  getQuery() {
    return this.query;
  }

  /**
   * @return {iterator} past queries history
   */
  getHistory() {
    return this.history.values();
  }

  /**
   * Get results for the given model
   * @param  {[type]} model [description]
   * @returns {array} current query results
   */
  getResults(model) {
    if (this.results.has(model)) {
      return this.results.get(model).data || [];
    }
    return [];
  }

  /**
   * Get status for the given model
   * @param  {[type]} model [description]
   * @returns {string} current status ("error", "ready", "pending") or null
   */
  getStatus(model) {
    if (this.results.has(model)) {
      return this.results.get(model).status;
    }
    return null;
  }

  /**
   * Get error for the given model
   * @param  {[type]} model [description]
   * @returns {string} last error or null
   */
  getError(model) {
    if (this.results.has(model)) {
      return this.results.get(model).error;
    }
    return null;
  }

  /**
   * Handle new query event
   * @param  {object} payload New data
   */
  _handleQueryModel(payload) {
    let {query, model} = payload;
    this.query = query;
    if (this.query) {
      // Remove whitespaces
      this.query = query.trim();
      // Do not add empty queries to history
      if (this.query.length > 0) {
        this.history.add(this.query);
      }
    }
    if (model) {
      this.results.set(model, {status:'pending'});
    } else {
      // No model
      this.results.clear();
    }
    this.emitChange();
  }

   /**
    * Handle new data event
    * @param  {object} payload New data
    */
  _handleReceivedData(payload) {
    let {query, model, results, error} = payload;

    if (query !== this.query) {
      this.results.set(model, {
        status:'error',
        error: `Unexpected query results: Query = ${query}.` +
               `Expecting Query = ${this.query}`
      });
      this.emitChange();
      return;
    }

    if (error) {
      this.results.set(model, {status:'error', error});
    } else if (results) {
      // Process results data
      let data = []
      Object.keys(results)
        .map((id) => {
          let record = results[id];
          let text = record.text;
          let scores = record.scores;
          let windowSize = record.windowSize;
          let fragmentIndices = record.indices;

          if (fragmentIndices) {
            // Data is query results, so break results into their fragments
            for (let i=0; i < fragmentIndices.length; i++) {
              let startIndex = fragmentIndices[i][0]
              let endIndex = fragmentIndices[i][1]

              let fragScores = scores.slice(startIndex, endIndex)
              // Consistent with ImbuModels, we tokenize simply on spaces
              let words = text.split(' ')
              let fragWords = words.slice(startIndex, endIndex);
              let nullScore = 0
              if (startIndex > 0) {
                fragWords.unshift('...')
                fragScores.unshift(nullScore)
              }
              if (endIndex < words.length) {
                fragWords.push('...')
                fragScores.push(nullScore)
              }
              let fragText = fragWords.join(' ')

              // Find this result's max score and sum of scores
              let maxScore = fragScores.reduce((prev, current) => {
                return prev > current ? prev : current;
              });
              let sumScore = fragScores.reduce((prev, current) => {
                return prev + current ;
              });
              // Add this fragment
              let fragment = {
                fragText, maxScore, sumScore, fragScores, windowSize
              };
              data.push(fragment)
            }
          } else {
            // Data is not query results, just the dataset
            let fragText = text
            let fragScores = scores
            let maxScore = 0
            let sumScore = 0
            let fragment = {
              fragText, maxScore, sumScore, fragScores, windowSize
            };
            data.push(fragment)
          }
        })
      // Sort the data by max scores, breaking ties with score sums
      data.sort((a, b) => {
        let res = b.maxScore - a.maxScore;
        if (res === 0) {
          res = b.sumScore - a.sumScore;
        }
        return res;
      });
      this.results.set(model, {status:'ready', data});
    } else {
      // No data
      this.results.delete(model);
      this.results.set(model, {status:'ready'});
    }
    this.emitChange();
  }

  /**
   * Handle clear requests
   */
  _handleClearData() {
    this.query = null;
    this.results.clear();
    this.history.clear();
    this.emitChange();
  }
}
