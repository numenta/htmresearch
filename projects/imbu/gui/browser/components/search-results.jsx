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
import Material from 'material-ui';
import React from 'react';
import ReactDOM from 'react-dom';

import DatasetStore from '../stores/dataset';
import DialogOpenAction from '../actions/dialog-open';
import MODELS from '../constants/models';
import SearchQueryAction from '../actions/search-query';
import SearchStore from '../stores/search';
import ServerStatusStore from '../stores/server-status';

const {
  Styles, Paper, DropDownMenu, MenuItem, RefreshIndicator,
  Table, TableHeader, TableRow, TableHeaderColumn, TableBody, TableRowColumn
} = Material;

const {
  Spacing, Colors
} = Styles;


/**
 * Display Search Results on a Material UI Table
 */
@connectToStores([SearchStore, ServerStatusStore, DatasetStore], (context, props) => ({ // eslint-disable-line
  ready: context.getStore(ServerStatusStore).isReady(),
  dataset: context.getStore(DatasetStore).getCurrent(),
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
    let error = context.getStore(SearchStore).getError(model);
    let status = context.getStore(SearchStore).getStatus(model);
    this.state = {model, results, error, status};
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
          cursor: 'pointer',
          overflow: 'auto',
          whiteSpace: 'normal'
        },
        score: {
          cursor: 'pointer',
          textAlign: 'right',
          verticalAlign: 'top',
          width: '120px'
        }
      },
      content: {
        paddingLeft: `${Spacing.desktopGutterMini}px`,
        maxWidth: '1200px',
        margin: '1 auto'
      },
      modelsMenu: {
        border: '1px solid lightgray',
        borderRadius: '5px',
        backgroundColor: Colors.grey50
      },
      modelsMenuUnderline: {
        display: 'none'
      },
      modelDescription: {
        fontSize: 'smaller',
        whiteSpace: 'normal',
        fontStyle: 'italic',
        position: 'relative',
        color: Colors.black,
        left: '1rem'
      },
      modelItem: {
        lineHeight: '1rem',
        paddingBottom: '1rem',
        width: '40rem'
      },
      table: {
        height: '500px'
      },
      refresh: {
        margin: 5,
        display: 'inline-block',
        position: 'relative',
        verticalAlign: 'bottom'
      },
      error: {
        color: Colors.red500
      }
    };
  }

  _modelChanged(event, index, value) {
    let model = value;
    this.setState({model});
    this._search(this.props.query, this.props.dataset, model);
  }

  /**
   * Handle click on search results MaterialUI Table cell. Trigger a material-ui
   *  Modal Dialog popup layer with Document/Row details.
   * @param {Number} rowNumber - Clicked Table cell row index
   * @param {Number} columnId - Clicked Table cell column index
   * @see http://www.material-ui.com/#/components/table
   */
  _onTableCellClick(rowNumber, columnId) {
    let result = this.state.results[rowNumber];
    let payload = {
      title: `Document Row#${rowNumber} Details`,
      body: result.text
    };
    this.context.executeAction(DialogOpenAction, payload);
  }

  _search(query, dataset, model) {
    this.context.executeAction(SearchQueryAction, {query, dataset, model});
  }

  componentDidMount() {
    this._search(this.props.query, this.props.dataset, this.state.model);
  }

  componentDidUpdate() {
    let table = ReactDOM.findDOMNode(this.refs.resultBody)
    if (table.firstChild) {
      table.firstChild.scrollIntoViewIfNeeded()
    }
  }

  componentWillReceiveProps(nextProps) {
    let model = this.state.model;
    if (this.props.dataset !== nextProps.dataset ||
        this.props.query !== nextProps.query) {
      this._search(nextProps.query, nextProps.dataset, this.state.model);
    }
    let results = this.context.getStore(SearchStore).getResults(model);
    let error = this.context.getStore(SearchStore).getError(model);
    let status = this.context.getStore(SearchStore).getStatus(model);
    this.setState({model, results, error, status});
  }

  formatResults(data) {
    let {text, startIndex, endIndex, scores, maxScore, windowSize} = data;

    if (scores.length > 1) {
      // Consistent with ImbuModels methods, we tokenize simply on spaces
      let words = text.split(' ');
      let fragWords = words.slice(startIndex, endIndex);
      let fragScores = scores.slice(startIndex, endIndex)
      let nullScore = 0
      if (startIndex > 0) {
        // Add ellipsis to show the document continues before fragment
        fragWords.unshift('...')
        fragScores.unshift(nullScore)
      }
      if (endIndex < words.length) {
        // Add ellipsis to show the document continues after fragment
        fragWords.push('...')
        fragScores.push(nullScore)
      }

      let elements = [];
      let highlightStyle = {
        backgroundColor: Colors.purple100
      };

      for (let i=0; i < fragWords.length; i++) {
        let score = fragScores[i];
        let currentElement = {
          score,
          text: fragWords[i],
          style: {}
        };
        elements.push(currentElement);

        if (score > 0 && score === maxScore) {
          // Highlight word(s) or window(s) with maxScore
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
    let tableStyle = styles.table;
    let ready = this.props.ready;
    let status = this.state.status;
    let statusComponent;

    if (status === 'pending') {
      // Show Progress
      statusComponent = (
        <RefreshIndicator
          size={40}
          left={5}
          top={0}
          status="loading"
          loadingColor={Colors.pinkA200}
          style={styles.refresh}
        />

      );
    } else if (status === 'error') {
      // Show Error
      statusComponent = (<p style={styles.error}>Error using this model with this dataset</p>);
    }
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

    let modelMenuItems = Object.keys(MODELS).map((model, idx) => (
      <MenuItem
        key={`model${idx}`}
        label={MODELS[model].label}
        primaryText={
          <span>
            {MODELS[model].label}
            <br/>
            <span style={styles.modelDescription}>
              {MODELS[model].description}
            </span>
          </span>
        }
        style={styles.modelItem}
        value={model}
        />
    ));

    return (
      <Paper style={styles.content} depth={1}>
        <DropDownMenu style={styles.modelsMenu}
                      underlineStyle={styles.modelsMenuUnderline}
                      value={this.state.model}
                      disabled={!ready}
                      onChange={::this._modelChanged}>
          {modelMenuItems}
        </DropDownMenu>
        {statusComponent}
        <Table
          fixedHeader={true}
          height={styles.table.height}
          onCellClick={this._onTableCellClick.bind(this)}
          ref="results"
          selectable={false}
          style={tableStyle}
        >
          <TableHeader adjustForCheckbox={false} displaySelectAll={false}>
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
