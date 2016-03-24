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

import React from 'react';
import ReactDOM from 'react-dom';
import Material from 'material-ui';
import connectToStores from 'fluxible-addons-react/connectToStores';
import SearchQueryAction from '../actions/search-query';
import SearchStore from '../stores/search';
import ServerStatusStore from '../stores/server-status';

const {
  Styles, Paper,
  Table, TableHeader, TableRow, TableHeaderColumn, TableBody, TableRowColumn
} = Material;

const {
  Spacing, Colors
} = Styles;

/**
 * Display Search Results on a Material UI Table
 */
@connectToStores([SearchStore, ServerStatusStore], (context, props) => ({
  ready: context.getStore(ServerStatusStore).isReady(),
  query: context.getStore(SearchStore).getQuery()
}))
export default class SearchResultsComponent extends React.Component {

  static contextTypes = {
    getStore: React.PropTypes.func,
    executeAction: React.PropTypes.func
  };

  static propTypes = {
    model: React.PropTypes.string.isRequired
  };

  constructor(props, context) {
    super(props);
    let model = props.model;
    let results = context.getStore(SearchStore).getResults(model);
    this.state = {model, results};
  }

  _getStyles() {
    return {
      header: {
        score: {
          width: '50px'
        }
      },
      column: {
        summary: {
          whiteSpace: 'normal',
          overflow: 'auto'
        },
        score: {
          width: '120px',
          textAlign: 'right'
        }
      },
      content: {
        paddingLeft: `${Spacing.desktopGutterMini}px`,
        maxWidth: '1200px',
        margin: '1 auto'
      },
      modelsMenu: {
        height: '36px',
        fontSize: '12pt',
        border: '1px solid lightgray'
      },
      table: {
        height: '500px'
      }
    };
  }

  _modelChanged(event) {
    let model = event.target.value;
    this.setState({model});
    this._search(this.props.query, model);
  }

  _search(query, model) {
    this.context.executeAction(SearchQueryAction, {query, model});
  }

  componentDidMount() {
    this._search(this.props.query, this.state.model);
  }

  componentDidUpdate() {
    let table = ReactDOM.findDOMNode(this.refs.resultBody)
    if (table.firstChild) {
      table.firstChild.scrollIntoViewIfNeeded()
    }
  }

  componentWillReceiveProps(nextProps) {
    let model = this.state.model;
    let results = this.context.getStore(SearchStore).getResults(model);
    if (results.length === 0) {
      this._search(nextProps.query, this.state.model);
    } else {
      this.setState({model, results});
    }
  }

  formatResults(data) {
    let {text, scores, maxScore, windowSize} = data;

    if (scores.length > 1) {
      let words = text.split(' ');
      let elements = [];
      let highlightStyle = {
        backgroundColor: Colors.purple200
      };

      for (let i=0; i < words.length; i++) {
        let score = scores[i];
        let currentElement = {
          score,
          text: words[i],
          style: {}
        };
        elements.push(currentElement);

        if (score > 0 && score === maxScore) {
          // Highlight word or window with maxScore
          elements.slice(-windowSize).forEach((obj) => {
            obj.style = highlightStyle;
          });
        }
      }

      return elements.map((obj) => {
        return (<span title={obj.score} style={obj.style}>{obj.text} </span>);
      });
    }
    return text;
  }

  render() {
    let styles = this._getStyles();
    let ready = this.props.ready;

    // Convert SearchStore results to Table rows
    let rows = this.state.results.map((result, idx) => {
      return (
        <TableRow key={idx}>
          <TableRowColumn key={0} style={styles.column.summary}>
            {this.formatResults(result)}
          </TableRowColumn>
          <TableRowColumn key={1} style={styles.column.score}>
            {result.maxScore.toFixed(4)}
          </TableRowColumn>
        </TableRow>);
    });

    return (
      <Paper style={styles.content} depth={1}>

        <select height={styles.modelsMenu.height}
                disabled={!ready}
                onChange={this._modelChanged.bind(this)}
                value={this.state.model}
                style={styles.modelsMenu}>
          <option value="CioDocumentFingerprint">
            Cortical.io document-level fingerprints
          </option>
          <option value="CioWordFingerprint">
            Cortical.io word-level fingerprints (unioned)
          </option>
          <option value="Keywords">
            Keywords (random encodings)
          </option>
          <option value="HTM_sensor_knn">
            Cortical.io word-level fingerprints
          </option>
          <option value="HTM_sensor_tm_knn">
            Sensor-TM-kNN Network
          </option>
          <option value="HTM_sensor_simple_tp_knn">
            Sensor-simple UP-kNN Network
          </option>
        </select>

        <Table selectable={false} fixedHeader={true}
          height={styles.table.height} ref="results">
          <TableHeader  adjustForCheckbox={false} displaySelectAll={false}>
            <TableRow>
              <TableHeaderColumn key={0} style={styles.column.summary}>
                Match
              </TableHeaderColumn>
              <TableHeaderColumn key={1} style={styles.column.score}>
                Percent Overlap of Query
              </TableHeaderColumn>
            </TableRow>
          </TableHeader>
          <TableBody displayRowCheckbox={false} ref="resultBody">
            {rows}
          </TableBody>
        </Table>
      </Paper>
    );
  }
}
