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
import React from 'react';
import Material from 'material-ui';
import SelectDatasetAction from '../actions/dataset-select'
import DatasetStore from '../stores/dataset';
import SearchComponent from './search.jsx';
import SearchResultsComponent from './search-results.jsx';
import ThemeDecorator from 'material-ui/lib/styles/theme-decorator';
import ThemeManager from 'material-ui/lib/styles/theme-manager';
import LightTheme from 'material-ui/lib/styles/raw-themes/light-raw-theme';
import DATASETS from '../constants/datasets';

const {
  Toolbar, ToolbarTitle, ToolbarGroup,
  DropDownMenu, MenuItem,
  IconButton, GridList, GridTile, Paper
} = Material;

@connectToStores([DatasetStore], (context) => ({
  datasets: context.getStore(DatasetStore).getDatasets(),
  currentDataset:  context.getStore(DatasetStore).getCurrent()
}))
@ThemeDecorator(ThemeManager.getMuiTheme(LightTheme)) // eslint-disable-line new-cap
export default class Main extends React.Component {

  static contextTypes = {
    executeAction: React.PropTypes.func,
    muiTheme: React.PropTypes.object
  };

  constructor(props) {
    super(props);
  }

  _getStyles() {
    let theme = this.context.muiTheme.appBar;
    return {
      toolbar: {
        paddingLeft: theme.padding,
        paddingRight: theme.padding,
        backgroundColor: theme.color
      },
      title: {
        lineHeight: `${theme.height}px`,
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        margin: 0,
        paddingTop: 0,
        letterSpacing: 0,
        fontSize: 24,
        fontWeight: theme.titleFontWeight,
        color: theme.textColor
      },
      datasetTitle: {
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        paddingLeft: 0,
        paddingRight: 0,
        letterSpacing: 0,
        fontSize: 18,
        fontWeight: theme.titleFontWeight,
        color: theme.textColor
      },
      menuLabel: {
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        letterSpacing: 0,
        fontSize: 18,
        fontWeight: theme.titleFontWeight,
        color: theme.textColor
      },
      iconStyle: {
        fill: theme.textColor,
        color: theme.textColor
      },
      tile: {
        margin: 1
      }
    };
  }

  _selectDataset(e, index, value) {
    if (value !== this.props.currentDataset) {
      this.context.executeAction(SelectDatasetAction, value);
    }
  }
  render() {
    let styles = this._getStyles();
    let datasetItems = this.props.datasets.map((dataset) => {
      let label = DATASETS[dataset] ? DATASETS[dataset].label : dataset;
      return (<MenuItem key={dataset} value={dataset} primaryText={label}/>);
    });
    return (
      <div>
        <Toolbar style={styles.toolbar}>
          <ToolbarGroup float="left">
            <ToolbarTitle style={styles.title} text="Numenta Imbu"/>
          </ToolbarGroup>
          <ToolbarGroup float="right" lastChild={true}>
            <ToolbarTitle style={styles.datasetTitle} text="Dataset :"/>
            <DropDownMenu value={this.props.currentDataset}
              labelStyle={styles.menuLabel}
              onChange={::this._selectDataset}>
              {datasetItems}
            </DropDownMenu>
            <IconButton href="http://www.numenta.com"
              linkButton={true}
              iconStyle={styles.iconStyle}
              iconClassName="material-icons"
              tooltip="Go to numenta.com">home</IconButton>
          </ToolbarGroup>
        </Toolbar>
        <SearchComponent/>
        <br/>
        <GridList cols={2} cellHeight={800}>
          <GridTile>
            <Paper style={styles.tile}>
              <SearchResultsComponent model="CioDocumentFingerprint"/>
            </Paper>
          </GridTile>
          <GridTile>
            <Paper style={styles.tile}>
              <SearchResultsComponent model="CioWordFingerprint"/>
            </Paper>
          </GridTile>
        </GridList>
      </div>
    );
  }
}
