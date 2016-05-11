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
 * IMBU Dialog store (for Document/Row details, etc)
 */
export default class DialogStore extends BaseStore {

  static storeName = 'DialogStore';

  static handlers = {
    DIALOG_CLOSE: '_closeDialog',
    DIALOG_OPEN: '_openDialog'
  };

  constructor(dispatcher) {
    super(dispatcher);
    this._clear();
  }

  _clear() {
    this._open = false;
    this._title = null;
    this._body = null;
    this._actions = [];
  }

  _closeDialog() {
    this._clear();
    this.emitChange();
  }

  _openDialog(payload) {
    this._open = true;
    this._title = payload.title;
    this._body = payload.body;
    this._actions = payload.actions || [];
    this.emitChange();
  }

  getCurrent() {
    return {
      open: this._open,
      title: this._title,
      body: this._body,
      actions: this._actions
    };
  }

}
