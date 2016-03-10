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

import connectToStores from 'fluxible-addons-react/connectToStores';
import ReactDOM from 'react-dom';
import React from 'react';
import Material from 'material-ui';

import CheckServerStatusAction from '../actions/server-status';
import SearchQueryAction from '../actions/search-query';
import SearchStore from '../stores/search';
import ServerStatusStore from '../stores/server-status';

const {
  RaisedButton, TextField, Styles, ClearFix, LinearProgress
} = Material;

const {
  Spacing, Colors
} = Styles;

@connectToStores([SearchStore, ServerStatusStore], (context) => ({
  ready: context.getStore(ServerStatusStore).isReady(),
  query: context.getStore(SearchStore).getQuery()
}))
export default class SearchComponent extends React.Component {

  static contextTypes = {
    executeAction: React.PropTypes.func,
    getStore: React.PropTypes.func
  };

  constructor() {
    super();
  }

  componentDidMount() {
    // Check server status
    this._checkServerStatus();
  }

  componentDidUpdate() {
    const el = ReactDOM.findDOMNode(this.refs.query);
    this.refs.query.setValue(this.props.query);
    el.focus();
  }

  /**
   * Pool server until all models are ready
   */
  _checkServerStatus() {
    if (!this.props.ready) {
      this.context.executeAction(CheckServerStatusAction);
      // Wait 5 seconds before next poll
      setTimeout(() =>  this._checkServerStatus(), 5000);
    }
  }

  _search() {
    let query = this.refs.query.getValue() || '';
    this.context.executeAction(SearchQueryAction, {query});
  }

  _getStyles() {
    return {
      content: {
        padding: `${Spacing.desktopGutterMini}px`,
        margin: '0 auto',
        display: 'table',
        boxSizing: 'border-box'
      },
      searchField: {
        display: 'table-cell',
        width: '100%'
      },
      searchButton: {
        display: 'table-cell',
        width: '1px',
        float: 'right'
      },
      progress: {
        textAlign: 'center',
        padding: 5,
        color: Colors.red500
      }
    };
  }

  render() {
    let styles = this._getStyles();
    let progress;
    let ready = this.props.ready;
    if (!ready) {
      progress = (
        <ClearFix>
          <h3 height={styles.progress.height} style={styles.progress}>
            Please wait while models are being built
          </h3>
          <LinearProgress mode="indeterminate"/>
        </ClearFix>);
    }
    return (
      <div>
        {progress}
        <ClearFix style={styles.content}>
          <TextField floatingLabelText="Enter query:"
                     fullWidth={true}
                     id="query" name="query"
                     disabled={!ready}
                     onEnterKeyDown={this._search.bind(this)}
                     style={styles.searchField}
                     ref="query"/>
          <RaisedButton label="Search" onTouchTap={this._search.bind(this)}
                        disabled={!ready}
                        style={styles.searchButton}
                        role="search" secondary={true}/>
      </ClearFix>
    </div>);
  }
}
