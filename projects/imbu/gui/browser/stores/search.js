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
    SEARCH_RECEIVED_DATA: '_handleReceivedData',
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
    return this.results.get(model) || [];
  }

   /**
    * Handle new data event
    * @param  {object} payload New data
    */
  _handleReceivedData(payload) {
    if (payload.query) {
      // Remove whitespaces
      this.query = payload.query.trim();
    } else {
      this.query = '';
    }

    // Do not add empty queries to history
    if (this.query) {
      this.history.add(this.query);
    }

    if (payload.model) {
      let model = payload.model;
      if (payload.results) {
        // Find and sort results by max score
        let records = Object.keys(payload.results)
          .map((id) => {
            let record = payload.results[id];
            let text = record.text;
            let scores = record.scores;
            let windowSize = record.windowSize;

            // Find max
            let maxScore = record.scores.reduce((prev, current) => {
              return prev > current ? prev : current;
            });
            let sumScore = record.scores.reduce((prev, current) => {
              return prev + current ;
            });
            return {
              text, maxScore, sumScore, scores, windowSize
            };
          })
          .sort((a, b) => {
            let res = b.maxScore - a.maxScore;
            if (res === 0) {
              res = b.sumScore - a.sumScore;
            }
            return res;
          });
        this.results.set(model, records);
      } else {
        // No data
        this.results.delete(model);
      }
    } else {
      // No model
      this.results.clear();
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
