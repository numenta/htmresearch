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

import BaseStore from 'fluxible/addons/BaseStore';

export default class DatasetStore extends BaseStore {
  static storeName = 'DatasetStore';
  static handlers = {
    LOAD_DATASET_LIST: '_handleLoadDatasetList',
    SELECT_DATASET: '_handleSelectDataset'
  };

  constructor(dispatcher) {
    super(dispatcher);
    this._datasets = [];
    this._current = null;
  }

  _handleLoadDatasetList(list) {
    this._datasets.push(...list);
    if (this._current === null && this._datasets.length > 0) {
      // Select first dataset by default
      this._current = this._datasets[0];
    }
    this.emitChange();
  }

  _handleSelectDataset(selection) {
    if (this._current !== selection) {
      if (this._datasets.includes(selection)) {
        this._current = selection;
      } else {
        this._current = null;
      }
      this.emitChange();
    }
  }

  getDatasets() {
    return this._datasets;
  }

  getCurrent() {
    return this._current;
  }
}
