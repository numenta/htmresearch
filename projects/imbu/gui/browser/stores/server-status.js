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
  Imbu server status store
*/
export default class ServerStatusStore extends BaseStore {

  static storeName = 'ServerStatusStore';

  static handlers = {
    SERVER_STATUS: '_handleReceivedData'
  };

  constructor(dispatcher) {
    super(dispatcher);
    this._ready = false;
  }

  /**
   * Return true if the server is up and ready to received requests
   */
  isReady() {
    return this._ready;
  }

  _handleReceivedData(payload) {
    if (this._ready !== payload.ready) {
      this._ready = payload.ready;
      this.emitChange();
    }
  }
}
