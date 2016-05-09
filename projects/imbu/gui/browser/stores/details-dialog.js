/* -----------------------------------------------------------------------------
 * Copyright © 2016, Numenta, Inc. Unless you have purchased from
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
 * IMBU Document Details Dialog store
 */
export default class DetailsDialogStore extends BaseStore {

  static storeName = 'DetailsDialogStore';

  static handlers = {
    DETAILS_DIALOG_CLOSE: '_close',
    DETAILS_DIALOG_OPEN: '_open'
  };

  constructor(dispatcher) {
    super(dispatcher);
    this._clear();
  }

  _clear() {
    this._open = false;
    this._title = null;
    this._body = null;
  }

  _close() {
    this._clear();
    this.emitChange();
  }

  _open(payload) {
    this._open = true;
    this._title = payload.title;
    this._body = payload.body;
    this.emitChange();
  }

}
