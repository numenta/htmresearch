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

import connectToStores from 'fluxible-addons-react/connectToStores';
import Material from 'material-ui';
import React from 'react';

import DialogCloseAction from '../actions/dialog-close';
import DialogStore from '../stores/dialog';

const {Dialog, FlatButton} = Material;


/**
 * GUI Dialog (Document/Row Details, etc.) via Material-UI
 * @see http://www.material-ui.com/#/components/dialog
 */
@connectToStores([DialogStore], (context, props) => ({ // eslint-disable-line
  dialog: context.getStore(DialogStore).getCurrent()
}))
export default class extends React.Component {

  static contextTypes = {
    executeAction: React.PropTypes.func,
    getStore: React.PropTypes.func
  };

  constructor(props, context) {
    super(props);
  }

  _handleClose() {
    this.context.executeAction(DialogCloseAction);
  }

  render() {
    let {open, title, body, actions} = this.props.dialog;

    if (! actions.length) {
      actions = (
        <FlatButton label="Close" onTouchTap={this._handleClose.bind(this)} />
      );
    }

    return (
      <Dialog
        actions={actions}
        modal={false}
        open={open}
        onRequestClose={this._handleClose.bind(this)}
        title={title}
        titleStyle={{fontSize:14}}
      >
        {body}
      </Dialog>
    );
  }

}
