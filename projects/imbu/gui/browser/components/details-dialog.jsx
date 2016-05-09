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

import React from 'react';
import Material from 'material-ui';
import connectToStores from 'fluxible-addons-react/connectToStores';
import DatasetStore from '../stores/dataset';

const {Dialog, FlatButton} = Material;


/**
 * Display Document Details GUI Dialog
 */
@connectToStores([DatasetStore], (context, props) => ({ // eslint-disable-line
  dataset: context.getStore(DatasetStore).getCurrent()
}))
export default class extends React.Component {

  static contextTypes = {
    getStore: React.PropTypes.func
  };

  constructor(props, context) {
    super(props);
    console.log(this.props.dataset);
    this.state = {open: false};
  }

  _handleClose() {
    this.setState({open: false});
  }

  render() {
    const actions = (
      <FlatButton label="Close" onTouchTap={this._handleClose.bind(this)} />
    );

    return (
      <Dialog
        actions={actions}
        modal={false}
        open={this.state.open}
        onRequestClose={this._handleClose.bind(this)}
      >
        Dialog Content
      </Dialog>
    );
  }

}
