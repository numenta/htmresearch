import * as d3 from 'd3';
import {arrayOfAxonsChart} from './charts/arrayOfAxonsChart2.js';
import {decodedLocationsChart} from './charts/decodedLocationsChart.js';
import {decodedObjectsChart} from './charts/decodedObjectsChart.js';
import {featureChart} from './charts/featureChart.js';
import {layerOfCellsChart} from './charts/layerOfCellsChart.js';
import {locationModulesChart} from './charts/locationModulesChart.js';
import {motionChart} from './charts/motionChart.js';
import {timelineChart} from './charts/timelineChart2.js';
import {worldChart} from './charts/worldChart2.js';

/**
 *
 * Example timestep:
 * {
 *   bodyWorldLocation: {left: 42.0, top: 12.0},
 *   reset: null,
 *   worldFeatures: {
 *     'Object 1': [
 *       {width: 8, height: 8, top: 0, left: 8, name: 'A'},
 *       {width: 8, height: 8, top: 0, left: 16, name: 'B'}]
 *   },
 *   predictedBodyToSpecificObject: {
 *     modules: [
 *       {activeCells: [],
 *        activeSynapsesByCell: {}},
 *     ]
 *   },
 *   anchoredBodyToSpecificObject: {
 *     modules: [
 *       {activeCells: [],
 *        activeSynapsesByCell: {}},
 *     ]
 *   },
 *   corticalColumns: [{
 *     egocentricLocation: {left: 42.0, top: 12.0},
 *     featureInput: {
 *       inputSize: 150,
 *       activeBits: [],
 *       decodings: []
 *     },
 *     sensorToBody: {
 *       modules: [
 *         {activeCells: []},
 *       ]
 *     },
 *     predictedSensorToSpecificObject: {
 *       modules: [
 *         {activeCells: [],
 *          activeSynapsesByCell: {}},
 *       ]
 *     },
 *     anchoredSensorToSpecificObject: {
 *       modules: [
 *         {activeCells: [],
 *          activeSynapsesByCell: {}},
 *       ]
 *     },
 *     predictedFeatureLocationPair: {
 *       predictedCells: [],
 *       decodings: [],
 *       activeSynapsesByCell: {
 *         42: {
 *           locationLayer: [12, 17, 29],
 *           objectLayer: [42, 45]
 *         }
 *       }
 *     },
 *     featureLocationPair: {
 *       activeCells: [],
 *       decodings: [],
 *       activeSynapsesByCell: {
 *         42: {
 *           locationLayer: [12, 17, 29],
 *           objectLayer: [42, 45]
 *         }
 *       }
 *     },
 *     objectLayer: {
 *       activeCells: [],
 *       decodings: [],
 *       activeSynapsesByCell: {}
 *     }
 *   }]
 * };
 */

function parseData(text) {
  let featureColor = d3.scaleOrdinal(),
      timesteps = [],
      rows = text.split('\n');

  // Row: number of cortical columns
  let numCorticalColumns = parseInt(rows[0]);

  // Row: world dimensions
  let worldDims = JSON.parse(rows[1]);

  // Row: Features and colors
  //   {'A': 'red',
  //    'B': 'blue',
  //    'C': 'gray'}
  let featureColorMapping = JSON.parse(rows[2]);

  var features = [],
      colors = [];

  for (var feature in featureColorMapping) {
    features.push(feature);
    colors.push(featureColorMapping[feature]);
  }

  featureColor
    .domain(features)
    .range(colors);

  // Third row: Objects
  // {
  //   'Object 1': [
  //     {top: 12.0, left: 11.2, width: 5.2, height: 19, name: 'A'},
  //     ...
  //   ],
  //   'Object 2': []
  // };
  let objects = JSON.parse(rows[3]);

  let currentTimestep = null,
      didReset = false,
      worldFeatures = null,
      worldLocationByColumn = null,
      bodyWorldLocation = null;

  // [{cellDimensions: [5,5], moduleMapDimensions: [20.0, 20.0], orientation: 0.2},
  //  ...]
  let configByModule = JSON.parse(rows[4]).map(d => {
    d.dimensions = {rows: d.cellDimensions[0], cols: d.cellDimensions[1]};
    return d;
  }),
      objectLayerConfig = {dimensions: {rows: 16, cols: 256}};

  function endTimestep() {
    if (currentTimestep !== null) {
      currentTimestep.worldFeatures = worldFeatures;

      timesteps.push(currentTimestep);
    }

    currentTimestep = null;
  }

  function beginNewTimestep(type) {
    endTimestep();

    currentTimestep = {
      corticalColumns: d3.range(numCorticalColumns).map(_ => { return {}; }),
      type
    };

    currentTimestep.corticalColumns.forEach((c, iCol) => {
      c.prevObjectLayer = (timesteps.length == 0 || didReset)
        ? Object.assign({cells: [], decodings: []}, objectLayerConfig)
        : timesteps[timesteps.length - 1].corticalColumns[iCol].objectLayer;
    });

    currentTimestep.predictedBodyToSpecificObject =
      (timesteps.length == 0 || didReset)
      ? {modules: configByModule.map(
         c => Object.assign({cells: [], decodings: []}, c))}
      : timesteps[timesteps.length - 1].anchoredBodyToSpecificObject;

    worldLocationByColumn.forEach((location, iCol) => {
      currentTimestep.corticalColumns[iCol].worldLocation =
        {top: location[0], left: location[1]};
    });

    currentTimestep.bodyWorldLocation =
      {top: bodyWorldLocation[0], left: bodyWorldLocation[1]};

    if (didReset) {
      currentTimestep.reset = true;
      didReset = false;
    }
  }

  let i = 5;
  while (i < rows.length) {
    switch (rows[i]) {
    case 'reset':
      didReset = true;
      i++;
      break;
    case 'compute': {
      let timestepType = rows[i+1];
      beginNewTimestep(timestepType);

      let egocentricLocationByColumn = JSON.parse(rows[i+2]),
          featureInputByColumn = JSON.parse(rows[i+3]),
          featureDecodingsByColumn = JSON.parse(rows[i+4]);

      featureInputByColumn.forEach((activeBits, iColumn) => {
        let decodings = featureDecodingsByColumn[iColumn];
        currentTimestep.corticalColumns[iColumn].featureInput = {
          inputSize: 150,
          activeBits,
          decodings
        };
      });

      egocentricLocationByColumn.forEach((location, iColumn) => {
        currentTimestep.corticalColumns[iColumn].egocentricLocation =
          {top: location[0], left: location[1]};
      });

      i += 5;
      break;
    }
    case 'bodyLocationInWorld': {
      bodyWorldLocation = JSON.parse(rows[i+1]);
      i += 2;
      break;
    }
    case 'locationInWorld': {
      worldLocationByColumn = JSON.parse(rows[i+1]);
      i += 2;
      break;
    }
    case 'sensorToBody': {
      let cellsByModuleByColumn = JSON.parse(rows[i+1]);

      cellsByModuleByColumn.forEach((cellsByModule, iCol) => {
        let modules = cellsByModule.map((activeCells, iModule) => {
          let cells = activeCells.map(cell => {
            return {
              cell,
              state: 'predicted-active'
            };
          });

          return Object.assign({cells,
                                activeSynapsesByCell: {}},
                               configByModule[iModule]);
        });

        currentTimestep.corticalColumns[iCol].sensorToBody = {
          modules
        };
      });

      i += 2;
      break;
    }
    case 'anchoredBodyToSpecificObject': {
      let cellsByModule = JSON.parse(rows[i+1]);

      let modules = cellsByModule.map((d, iModule) => {
          let [activeCells, synapsesForActiveCellsBySourceLayer] = d;

        let predictedCells = currentTimestep
            .predictedBodyToSpecificObject
            .modules[iModule]
            .cells
            .map(d => d.cell);

        let cells = activeCells.map(cell => {
          return {
            cell,
            state: predictedCells.indexOf(cell) != -1
              ? 'predicted-active'
              : 'active'
          };
        });

        let activeSynapsesByCell = synapsesForActiveCellsBySourceLayer
            ? getActiveSynapsesByCell(activeCells,
                                       synapsesForActiveCellsBySourceLayer)
            : null;

        return Object.assign({cells,
                              activeSynapsesByCell},
                             configByModule[iModule]);
      });

      currentTimestep.anchoredBodyToSpecificObject = {
        modules
      };

      i += 2;
      break;
    };
    case 'predictedSensorToSpecificObject': {
      let cellsByModuleByColumn = JSON.parse(rows[i+1]),
          decodingsByColumn = JSON.parse(rows[i+2]);

      cellsByModuleByColumn.forEach((cellsByModule, iCol) => {
        let modules = cellsByModule.map((d, iModule) => {
          let [activeCells, synapsesForActiveCellsBySourceLayer] = d;

          let cells = activeCells.map(cell => {
            return {
              cell,
              state: 'predicted-active'
            };
          });

          let activeSynapsesByCell = synapsesForActiveCellsBySourceLayer
              ? getActiveSynapsesByCell(activeCells,
                                         synapsesForActiveCellsBySourceLayer)
              : null;

          return Object.assign({cells,
                                activeSynapsesByCell},
                               configByModule[iModule]);
        });

        let decodings = decodingsByColumn[iCol].map(
          ([objectName, top, left, amountContained]) => {
            return { objectName, top, left, amountContained };
          });

        currentTimestep.corticalColumns[iCol].predictedSensorToSpecificObject = {
          modules, decodings
        };
      });

      i += 3;
      break;
    }
    case 'anchoredSensorToSpecificObject': {
      let cellsByModuleByColumn = JSON.parse(rows[i+1]),
          decodingsByColumn = JSON.parse(rows[i+2]);

      cellsByModuleByColumn.forEach((cellsByModule, iCol) => {
        let modules = cellsByModule.map((d, iModule) => {
          let [activeCells, synapsesForActiveCellsBySourceLayer] = d;

          let predictedCells = currentTimestep
              .corticalColumns[iCol]
              .predictedSensorToSpecificObject
              .modules[iModule]
              .cells
              .map(d => d.cell);

          let cells = activeCells.map(cell => {
            return {
              cell,
              state: predictedCells.indexOf(cell) != -1
                ? 'predicted-active'
                : 'active'
            };
          });

          let activeSynapsesByCell = synapsesForActiveCellsBySourceLayer
              ? getActiveSynapsesByCell(activeCells,
                                         synapsesForActiveCellsBySourceLayer)
              : null;

          return Object.assign({cells, activeSynapsesByCell},
                               configByModule[iModule]);

        });

        let decodings = decodingsByColumn[iCol].map(
          ([objectName, top, left, amountContained]) => {
            return { objectName, top, left, amountContained };
          });

        currentTimestep.corticalColumns[iCol].anchoredSensorToSpecificObject = {
          modules, decodings
        };
      });

      i += 3;
      break;
    }
    case 'predictedFeatureLocationPair': {
      let cellsByColumn = JSON.parse(rows[i+1]);
      let decodingsByColumn = JSON.parse(rows[i+2]);

      cellsByColumn.forEach((d, iCol) => {
        let [predictedCells, segmentsForPredictedCells] = d;

        let cells = predictedCells.map(cell => {
          return {
            cell,
            state: 'predicted'
          };
        });

        let predictedCellDecodings = decodingsByColumn[iCol].map(
          ([objectName, top, left, amountContained]) => {
            return { objectName, top, left, amountContained };
          });

        let synapsesByPredictedCell = {};

        predictedCells.forEach(cell => {
          synapsesByPredictedCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForPredictedCells) {
          segmentsForPredictedCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            synapsesByPredictedCell[predictedCells[ci]][presynapticLayer] = synapses;
          });
        }

        currentTimestep.corticalColumns[iCol].predictedFeatureLocationPair = {
          predictedCells,
          activeSynapsesByCell: synapsesByPredictedCell,
          decodings: predictedCellDecodings,
          cells: cells,
          dimensions: {rows: 32, cols: 150}
        };
      });

      i += 3;
      break;
    }
    case 'featureLocationPair': {
      let cellsByColumn = JSON.parse(rows[i+1]);
      let decodingsByColumn = JSON.parse(rows[i+2]);

      cellsByColumn.forEach((d, iCol) => {
        let [activeCells, synapsesForActiveCellsBySourceLayer] = d;

        let predictedCells = currentTimestep
            .corticalColumns[iCol]
            .predictedFeatureLocationPair
            .cells
            .map(d => d.cell);

        let cells = activeCells.map(cell => {
          return {
            cell,
            state: predictedCells.indexOf(cell) != -1
              ? 'predicted-active'
              : 'active'
          };
        });

        let activeSynapsesByCell = synapsesForActiveCellsBySourceLayer
            ? getActiveSynapsesByCell(activeCells,
                                       synapsesForActiveCellsBySourceLayer)
            : null;

        let activeCellDecodings = decodingsByColumn[iCol].map(
          ([objectName, top, left, amountContained]) => {
            return { objectName, top, left, amountContained };
          });

        currentTimestep.corticalColumns[iCol].featureLocationPair = {
          cells, activeSynapsesByCell,
          decodings: activeCellDecodings,
          dimensions: {rows: 32, cols: 150}
        };

      });

      i += 3;
      break;
    }
    case 'objectLayer': {
      let cellsByColumn = JSON.parse(rows[i+1]),
          decodingsByColumn = JSON.parse(rows[i+2]);

      cellsByColumn.forEach((d, iCol) => {
        let [activeCells, synapsesForActiveCellsBySourceLayer] = d;

        let prevActiveCells = (currentTimestep.reset || timesteps.length == 0)
            ? []
            : timesteps[timesteps.length-1].corticalColumns[iCol].objectLayer.cells.map(d => d.cell);

        let cells = activeCells.map(cell => {
          return {
            cell,
            state: prevActiveCells.indexOf(cell) != -1
              ? 'predicted-active'
              : 'active'
          };
        });

        let activeSynapsesByCell = synapsesForActiveCellsBySourceLayer
            ? getActiveSynapsesByCell(activeCells,
                                       synapsesForActiveCellsBySourceLayer)
            : null;

        let decodings = decodingsByColumn[iCol];
        currentTimestep.corticalColumns[iCol].objectLayer = Object.assign(
          {cells, activeSynapsesByCell, decodings},
          objectLayerConfig);
      });

      i += 3;
      break;
    }
    case 'objectPlacements': {
      let objectPlacements = JSON.parse(rows[i+1]);

      worldFeatures = [];
      for (let objectName in objects) {
        let objectPlacement = objectPlacements[objectName];
        objects[objectName].forEach(({name, top, left, width, height}) => {
          worldFeatures.push({
            name, width, height,
            top: top + objectPlacement[0],
            left: left + objectPlacement[1]
          });
        });
      }

      i += 2;
      break;
    }
    default:
      if (rows[i] == null || rows[i] == '') {
        i++;
      } else {
        throw `Unrecognized: ${rows[i]}`;
      }
    }
  }

  endTimestep();

  return {
    timesteps, worldDims, configByModule, featureColor, objects
  };
}

let rowHeight = 90,
    secondColumnLeft = 180,
    secondRowTop = 12 + rowHeight,
    thirdRowTop = secondRowTop + rowHeight + 10,
    fourthRowTop = thirdRowTop + rowHeight,
    fifthRowTop = fourthRowTop + rowHeight + 34,
    bitsWidth = 90,
    bitsHeight = 90,
    decodingsWidth = 76,
    decodingsTop = 6,
    decodingsLeft = bitsWidth + 6,
    decodingsHeight = rowHeight - 8,
    columnWidth = bitsWidth + (decodingsWidth + 24) + 14;

let layout = {
  smallLabels: [
    {
      html: 'object',
      topRightCorner: {top: 12 + bitsHeight/2, left: 19}
    },
    {
      html: 'feature-location<br />pair',
      topRightCorner: {top: secondRowTop + bitsHeight/2.5, left: 19}
    },
    {
      html: 'feature input',
      topRightCorner: {top: thirdRowTop - 7, left: 19}
    },
    {
      html: 'sensor<br />relative to<br />body',
      topRightCorner: {top: thirdRowTop + bitsHeight/3, left: 19}
    },
    {
      html: 'sensor<br />relative to<br />specific object',
      topRightCorner: {top: fourthRowTop + bitsHeight/3, left: 19}
    },
    {
      html: 'body<br />relative to<br />specific object',
      topRightCorner: {top: fifthRowTop + (bitsHeight + 60)/2.55, left: 40 + 19}
    },
  ],
  mediumLabels: [
    {
      html: 'Neocortex',
      top: -1, left: 5
    },
    {
      html: 'Mystery Population',
      top: fifthRowTop - 13, left: 5 + 20
    },
    {
      html: 'The World',
      top: fifthRowTop - 13, left: 5 + 40 + 60 + 40 + bitsWidth + 138
    }],
  corticalColumn: {
    width: bitsWidth,
    marginRight: columnWidth - bitsWidth,
    firstMarginLeft: 20,

    object: {
      left: 0, top: 12, width: bitsWidth, height: rowHeight
    },
    decodedObject: {
      left: decodingsLeft, top: decodingsTop + 12, width: decodingsWidth,
      height: decodingsHeight
    },

    featureLocationPair: {
      left: 0, top: secondRowTop, width: bitsWidth, height: bitsHeight
    },
    decodedFeatureLocationPair: {
      left: decodingsLeft,
      top: decodingsTop + secondRowTop,
      width: decodingsWidth, height: decodingsHeight
    },

    featureInput: {
      left: 0, top: secondRowTop + bitsHeight, width: bitsWidth, height: 5
    },
    decodedFeatureInput: {
      left: decodingsLeft, top: secondRowTop + bitsHeight,
      height: 10, width: 10
    },

    sensorToBody: {
      left: 0, top: thirdRowTop, width: bitsWidth, height: bitsHeight
    },
    decodedSensorToBody: {
      left: decodingsLeft + decodingsWidth/2 - 5,
      top: decodingsTop + thirdRowTop + decodingsHeight / 2 - 10,
      width: decodingsWidth,
      height: decodingsHeight
    },

    sensorToSpecificObject: {
      left: 0, top: fourthRowTop, width: bitsWidth, height: bitsHeight
    },
    decodedSensorToSpecificObject: {
      left: decodingsLeft, top: decodingsTop + fourthRowTop, width: decodingsWidth,
      height: decodingsHeight
    }
  },

  bodyToSpecificObject: {
    left: 60, top: fifthRowTop, width: bitsWidth + 60,
    height: bitsHeight + 60,
    decodingsLeft: decodingsLeft, decodingsTop: decodingsTop, decodingsWidth: decodingsWidth,
    decodingsHeight: decodingsHeight
  },

  decodedBodyToSpecificObject: {
    left: decodingsLeft, top: decodingsTop, width: decodingsWidth,
    height: decodingsHeight
  },

  world: {
    left: 40 + 60 + 40 + bitsWidth + 138, top: fifthRowTop,
    width: bitsHeight + 60, height: bitsHeight + 60
  }
};

let lines = [
    // Top and bottom of neocortex
    {x1: 0, y1: 12,
     x2: columnWidth*3 - 70, y2: 12},
    {x1: 0, y1: fourthRowTop + rowHeight,
     x2: columnWidth*3 - 70, y2: fourthRowTop + rowHeight},

    // Top and bottom of claustrum
    {x1: layout.bodyToSpecificObject.left - 40,
     y1: layout.bodyToSpecificObject.top,
     x2: layout.bodyToSpecificObject.left + layout.bodyToSpecificObject.width + 40,
     y2: layout.bodyToSpecificObject.top},
    {x1: layout.bodyToSpecificObject.left - 40,
     y1: layout.bodyToSpecificObject.top + layout.bodyToSpecificObject.height,
     x2: layout.bodyToSpecificObject.left + layout.bodyToSpecificObject.width + 40,
     y2: layout.bodyToSpecificObject.top + layout.bodyToSpecificObject.height},

    // Sides of claustrum
    {x1: layout.bodyToSpecificObject.left, y1: layout.bodyToSpecificObject.top,
     x2: layout.bodyToSpecificObject.left, y2: layout.bodyToSpecificObject.top + layout.bodyToSpecificObject.height},
    {x1: layout.bodyToSpecificObject.left + layout.bodyToSpecificObject.width,
     y1: layout.bodyToSpecificObject.top,
     x2: layout.bodyToSpecificObject.left + layout.bodyToSpecificObject.width,
     y2: layout.bodyToSpecificObject.top + layout.bodyToSpecificObject.height}
];

function printRecording(node, text) {
  // Constants
  let margin = {top: 5, right: 5, bottom: 15, left: 5},
      width = 600,
      height = fifthRowTop + rowHeight + 20 + 40,
      parsed = parseData(text);

  // Mutable state
  let iTimestep = 0,
      iTimestepPhase = 0,
      selectedObjectCell = null,
      selectedInputCell = null,
      selectedBodyToSpecificObjectCell = null,
      selectedSensorToSpecificObjectCell = null,
      selectedSensorToBodyCell = null,
      highlightedCellsByLayer = {};

  // Allow a mix of SVG and HTML
  let html = d3.select(node)
        .append('div')
          .style('margin-left', 'auto')
          .style('margin-right', 'auto')
          .style('position', 'relative')
          .style('width', `${width + margin.left + margin.right}px`),
      htmlSelectable = html.append('div')
      .style('margin-left', '-50px')
      .style('padding-left', '50px')
      .style('margin-right', '-20px')
      .style('padding-right', '20px'),
      svg = htmlSelectable.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

  // Add keyboard navigation
  htmlSelectable
    .attr('tabindex', 0)
    .on('keydown', function() {
      switch (d3.event.keyCode) {
      case 37: // Left
        iTimestep--;
        if (iTimestep < 0) {
          iTimestep = parsed.timesteps.length - 1;
        }

        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      case 38: // Up:
       iTimestepPhase--;
        if (iTimestepPhase < 0) {
          iTimestep--;
          iTimestepPhase = 1;

          if (iTimestep < 0) {
            iTimestep = parsed.timesteps.length - 1;
          }
        }
        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      case 40: // Down
        iTimestepPhase++;
        if (iTimestepPhase > 1) {
          iTimestep = (iTimestep+1)%parsed.timesteps.length;
          iTimestepPhase = 0;
        }

        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      }
    });

  // Make the SVG a clickable slideshow
  let slideshow = svg.append('g')
      .on('click', () => {
        iTimestepPhase++;
        if (iTimestepPhase > 1) {
          iTimestep = (iTimestep+1) % parsed.timesteps.length;
          iTimestepPhase = 0;
        }

        onSelectedTimestepChanged();
      });

  slideshow.append('rect')
      .attr('fill', 'transparent')
      .attr('stroke', 'none')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.bottom + margin.top + 10);

  let container = slideshow
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

  // Arrange the boxes
  let bodyToSpecificObjectNode = container.selectAll('.layerBox')
      .data([layout.bodyToSpecificObject]);
  bodyToSpecificObjectNode.exit().remove();
  bodyToSpecificObjectNode = bodyToSpecificObjectNode.enter().append('g')
    .attr('class', 'layer')
    .merge(bodyToSpecificObjectNode)
    .attr('transform', d => `translate(${d.left}, ${d.top})`);

  let corticalColumn = container.selectAll('.corticalColumn')
      .data(d3.range(3));
  corticalColumn.exit().remove();
  corticalColumn = corticalColumn.enter()
    .append('g')
    .attr('class', 'corticalColumn')
    .attr('transform', (_, i) => `translate(${layout.corticalColumn.firstMarginLeft + i*(columnWidth)})`)
    .call(enter => {
      // Neocortex column
      enter.append('line')
        .attr('stroke', 'black')
        .attr('stroke-width', 2)
        .attr('x1', 0)
        .attr('y1', 12)
        .attr('x2', 0)
        .attr('y2', fourthRowTop + rowHeight);

      enter.append('line')
        .attr('stroke', 'black')
        .attr('stroke-width', 2)
        .attr('x1', bitsWidth)
        .attr('y1', 12)
        .attr('x2', bitsWidth)
        .attr('y2', fourthRowTop + rowHeight);

      ['object', 'decodedObject', 'featureLocationPair', 'decodedFeatureLocationPair',
       'featureInput', 'decodedFeatureInput',
       'sensorToBody', 'decodedSensorToBody', 'sensorToSpecificObject',
       'decodedSensorToSpecificObject'].forEach(layer => {
         let lay = layout.corticalColumn[layer];
         enter.append('g')
           .attr('class', layer)
           .attr('transform', `translate(${lay.left},${lay.top})`);
       });
    })
    .merge(corticalColumn);

  let worldNode = container.append('g')
      .attr('transform', `translate(${layout.world.left}, ${layout.world.top})`);

  let timelineNode = htmlSelectable
      .append('div')
      .style('padding-top', '5px')
      .style('padding-left', '17px') // Because it hangs some text off the side.
      .style('padding-right', '17px')
      .style('text-align', 'center');

  let line = container
      .selectAll('.myLine')
      .data(lines);
  line.exit().remove();
  line = line.enter()
    .append('line')
    .attr('stroke', 'black')
    .attr('stroke-width', 2)
    .merge(line)
    .attr('x1', d => d.x1)
    .attr('y1', d => d.y1)
    .attr('x2', d => d.x2)
    .attr('y2', d => d.y2);

  let smallLabel = html.selectAll('.smallLabel')
      .data(layout.smallLabels);
  smallLabel.exit().remove();
  smallLabel = smallLabel.enter()
    .append('div')
    .attr('class', 'smallLabel')
    .style('width', 0)
    .style('position', 'absolute')
    .call(div => {
      div.append('div')
        .attr('class', 'text')
        .style('position', 'absolute')
        .style('text-align', 'right')
        .style('font', '10px Verdana')
        .style('pointer-events', 'none')
        .style('width', '80px')
        .style('right', 0);
    })
    .merge(smallLabel)
    .style('top', d => `${d.topRightCorner.top}px`)
    .style('left', d => `${d.topRightCorner.left}px`);
  smallLabel.select('.text')
    .html(d => d.html);

  let mediumLabel = html.selectAll('.mediumLabel')
      .data(layout.mediumLabels);
  mediumLabel.exit().remove();
  mediumLabel = mediumLabel.enter()
    .append('div')
    .attr('class', 'mediumLabel')
    .style('position', 'absolute')
    .style('text-align', 'left')
    .style('font', '13px Verdana')
    .style('font-weight', 'bold')
    .style('pointer-events', 'none')
    .merge(mediumLabel)
    .style('left', d => `${d.left}px`)
    .style('top', d => `${d.top}px`)
    .html(d => d.html);

  // Configure the charts
  let sensorToBody = locationModulesChart()
        .width(layout.corticalColumn.sensorToBody.width)
        .height(layout.corticalColumn.sensorToBody.height)
        .numRows(4)
        .numCols(4)
        .color(() => "brown")
        .onCellSelected((iModule, cell, layerId) => {
          if (cell !== null) {
            selectedSensorToBodyCell = [layerId, iModule, cell];
          } else {
            selectedSensorToBodyCell = null;
          }
          onLocationCellSelected();
        }),
      sensorToBodyDatum = (column, iCol) => {
        return {
          modules: column.sensorToBody.modules,
          highlightedCells: highlightedCellsByLayer[`${iCol} sensorToBody`] || [],
          id: iCol
        };
      },
      decodedSensorToBody = motionChart()
        .maxLength(40),
      decodedSensorToBodyDatum = column => column.egocentricLocation,
      sensorToSpecificObject = locationModulesChart()
        .width(layout.corticalColumn.sensorToSpecificObject.width)
        .height(layout.corticalColumn.sensorToSpecificObject.height)
        .numRows(4)
        .numCols(4)
      .onCellSelected((iModule, cell, layerId) => {
        selectedSensorToSpecificObjectCell = (cell !== null)
          ? [layerId, iModule, cell]
          : null;
          onLocationCellSelected();
        }),
      sensorToSpecificObjectDatum = (column, iCol) => {
        return {
          modules: (iTimestepPhase == 0)
            ? column.predictedSensorToSpecificObject.modules
            : column.anchoredSensorToSpecificObject.modules,
          highlightedCells: highlightedCellsByLayer[`${iCol} sensorToSpecificObject`] || [],
          id: iCol
        };
      },
      decodedSensorToSpecificObject = decodedLocationsChart()
        .width(layout.corticalColumn.decodedSensorToSpecificObject.width)
        .height(layout.corticalColumn.decodedSensorToSpecificObject.height)
        .perRow(2)
        .color(parsed.featureColor),
      decodedSensorToSpecificObjectDatum = column => {
        return {
          decodings: (iTimestepPhase == 0)
            ? column.predictedSensorToSpecificObject.decodings
            : column.anchoredSensorToSpecificObject.decodings,
          objects: parsed.objects
        };
      },
      featureLocationPair = layerOfCellsChart()
        .width(layout.corticalColumn.featureLocationPair.width)
        .height(layout.corticalColumn.featureLocationPair.height)
        .columnMajorIndexing(true)
        .onCellSelected((cell, layerId) => {
          if (cell !== null) {
            selectedInputCell = [layerId, cell];
          } else {
            selectedInputCell = null;
          }
          onInputCellSelected();
        }),
      featureLocationPairDatum = (column, iCol) => Object.assign(
        {highlightedCells: highlightedCellsByLayer[`${iCol} inputLayer`] || [],
         id: iCol},
        (iTimestepPhase == 0)
          ? column.predictedFeatureLocationPair
          : column.featureLocationPair
      ),
      decodedFeatureLocationPair = decodedLocationsChart()
        .width(layout.corticalColumn.decodedFeatureLocationPair.width)
        .height(layout.corticalColumn.decodedFeatureLocationPair.height)
        .perRow(2)
        .color(parsed.featureColor),
      decodedFeatureLocationPairDatum = column => {
        return {
          decodings: (iTimestepPhase == 0)
            ? column.predictedFeatureLocationPair.decodings
            : column.featureLocationPair.decodings,
          objects: parsed.objects
        };
      },
      objectLayer = layerOfCellsChart()
        .width(layout.corticalColumn.object.width)
        .height(layout.corticalColumn.object.height)
        .onCellSelected((cell, layerId) => {
          if (cell !== null) {
            selectedObjectCell = [layerId, cell];
          } else {
            selectedObjectCell = null;
          }
          onObjectCellSelected();
        }),
      objectDatum = (column, iCol) => Object.assign(
        {highlightedCells: highlightedCellsByLayer[`${iCol} objectLayer`] || [],
         id: iCol},
        (iTimestepPhase) == 0
        ? column.prevObjectLayer
        : column.objectLayer),
      decodedObject = decodedObjectsChart()
        .width(layout.corticalColumn.decodedObject.width)
        .height(layout.corticalColumn.decodedObject.height)
        .perRow(2)
        .color(parsed.featureColor),
      decodedObjectDatum = column => {
        return {
          decodings: (iTimestepPhase) == 0
            ? column.prevObjectLayer.decodings
            : column.objectLayer.decodings,
          objects: parsed.objects
        };
      },
      featureInput = arrayOfAxonsChart()
        .width(layout.corticalColumn.featureInput.width)
        .height(layout.corticalColumn.featureInput.height)
        .rectWidth(1)
        .borderWidth(0),
      featureInputDatum = column => {
        return (iTimestepPhase == 0)
          ? null
          : column.featureInput;
      },
      decodedFeature = featureChart()
        .color(parsed.featureColor)
        .width(layout.corticalColumn.decodedFeatureInput.width)
        .height(layout.corticalColumn.decodedFeatureInput.height),
      decodedFeatureDatum = column => {
        return (iTimestepPhase == 0)
          ? null
          : {name: column.featureInput.decodings[0]};
      },
      bodyToSpecificObject = locationModulesChart()
        .width(layout.bodyToSpecificObject.width)
        .height(layout.bodyToSpecificObject.height)
        .numRows(4)
        .numCols(4)
        .onCellSelected((iModule, cell) => {
          selectedBodyToSpecificObjectCell = (cell !== null)
            ? [iModule, cell]
            : null;
          onLocationCellSelected();
        }),
      bodyToSpecificObjectDatum = () => {
        return {
          modules: (iTimestepPhase == 0)
            ? parsed.timesteps[iTimestep].predictedBodyToSpecificObject.modules
            : parsed.timesteps[iTimestep].anchoredBodyToSpecificObject.modules,
          highlightedCells: highlightedCellsByLayer['bodyToSpecificObject'] || []
        };
      },
      world = worldChart()
        .width(layout.world.width)
        .height(layout.world.height)
        .color(parsed.featureColor),
      worldDatum = () => {
        let [selectedBodyPart,
             selectedLayer,
             iModule,
             selectedCell] = getSelectedCell();

        let selectedLocationModule = null,
            selectedLocationCell = null,
            selectedAnchorLocation = null,
            timestep = parsed.timesteps[iTimestep];

        switch (selectedLayer) {
        case 'sensorToSpecificObject':
          selectedLocationModule = (iTimestepPhase == 0)
            ? timestep.corticalColumns[selectedBodyPart]
            .predictedSensorToSpecificObject
            .modules[iModule]
            : timestep
            .corticalColumns[selectedBodyPart]
            .anchoredSensorToSpecificObject
            .modules[iModule];
          selectedAnchorLocation = timestep.corticalColumns[selectedBodyPart]
            .worldLocation;
          break;
        case 'bodyToSpecificObject':
          selectedLocationModule = (iTimestepPhase == 0)
            ? timestep
            .predictedBodyToSpecificObject
            .modules[iModule]
            : timestep
            .anchoredBodyToSpecificObject
            .modules[iModule];
          selectedAnchorLocation = timestep.bodyWorldLocation;
          break;
        case 'sensorToBody':
          selectedLocationModule = timestep
            .corticalColumns[selectedBodyPart]
            .sensorToBody
            .modules[iModule];
          selectedAnchorLocation = timestep.bodyWorldLocation;
          break;
        }

        return {
          dims: parsed.worldDims,
          bodyLocation: timestep.bodyWorldLocation,
          locations: timestep.corticalColumns.map(c => c.worldLocation),
          features: timestep.worldFeatures,
          selectedLocationModule,
          selectedLocationCell,
          selectedAnchorLocation,
          selectedBodyPart
        };
      },
      timeline = timelineChart().onchange((iTimestepNew, iTimestepPhaseNew) => {
        iTimestep = iTimestepNew;
        iTimestepPhase = iTimestepPhaseNew;
        onSelectedTimestepChanged();
      }),
      timelineDatum = () => {
        return {
          timesteps: parsed.timesteps,
          selectedIndex: iTimestep,
          selectedPhase: iTimestepPhase
        };
      };

  calculateHighlightedCells();
  draw();

  //
  // Lifecycle functions
  //

  function draw(incremental) {
    corticalColumn
      .data(parsed.timesteps[iTimestep].corticalColumns);

    corticalColumn.select(':scope > .object')
      .datum(objectDatum)
      .call(objectLayer);

    corticalColumn.select(':scope > .decodedObject')
      .datum(decodedObjectDatum)
      .call(decodedObject);

    corticalColumn.select(':scope > .featureLocationPair')
      .datum(featureLocationPairDatum)
      .call(featureLocationPair);

    corticalColumn.select(':scope > .decodedFeatureLocationPair')
      .datum(decodedFeatureLocationPairDatum)
      .call(decodedFeatureLocationPair);

    corticalColumn.select(':scope > .sensorToBody')
      .datum(sensorToBodyDatum)
      .call(sensorToBody);

    corticalColumn.select(':scope > .decodedSensorToBody')
      .datum(decodedSensorToBodyDatum)
      .call(decodedSensorToBody);

    corticalColumn.select(':scope > .sensorToSpecificObject')
      .datum(sensorToSpecificObjectDatum)
      .call(sensorToSpecificObject);

    corticalColumn.select(':scope > .decodedSensorToSpecificObject')
      .datum(decodedSensorToSpecificObjectDatum)
      .call(decodedSensorToSpecificObject);

    corticalColumn.select(':scope > .featureInput')
      .datum(featureInputDatum)
      .call(featureInput);

    corticalColumn.select(':scope > .decodedFeatureInput')
      .datum(decodedFeatureDatum)
      .call(decodedFeature);

    bodyToSpecificObjectNode
      .datum(bodyToSpecificObjectDatum)
      .call(bodyToSpecificObject);

    worldNode
      .datum(worldDatum)
      .call(world);

    timelineNode
      .datum(timelineDatum)
      .call(incremental ? timeline.drawSelectedStep : timeline);
  }

  function onSelectedTimestepChanged() {
    calculateHighlightedCells();
    drawHighlightedCells();
    draw(true);
  }

  function onLocationCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();

    worldNode
      .datum(worldDatum)
      .call(world.drawFiringFields)
      .call(world.drawSelectedBodyPart);
  }

  function onInputCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();

    worldNode
      .datum(worldDatum)
      .call(world.drawSelectedBodyPart);
  }

  function onObjectCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();

    worldNode
      .datum(worldDatum)
      .call(world.drawSelectedBodyPart);
  }

  function getSelectedCell() {
    let selectedBodyPart = null,
        selectedLayer = null,
        selectedModule = null,
        selectedCell = null;

   // Selected object cell
    if (selectedObjectCell != null) {
      let [iCol, cell] = selectedObjectCell;
      selectedBodyPart = iCol;
      selectedLayer = 'object';
      selectedCell = cell;
    }

    // Selected input cell
    if (selectedInputCell != null) {
      let [iCol, cell] = selectedInputCell;
      selectedBodyPart = iCol;
      selectedLayer = 'featureLocationPair';
      selectedCell = cell;
    }

    // Selected sensor location cell
    if (selectedSensorToBodyCell != null) {
      let [iCol, iModule, cell] = selectedSensorToBodyCell;
      selectedBodyPart = iCol;
      selectedLayer = 'sensorToBody';
      selectedModule = iModule;
      selectedCell = cell;
    }

    // Selected sensor location cell
    if (selectedSensorToSpecificObjectCell != null) {
      let [iCol, iModule, cell] = selectedSensorToSpecificObjectCell;
      selectedBodyPart = iCol;
      selectedLayer = 'sensorToSpecificObject';
      selectedModule = iModule;
      selectedCell = cell;
    }

    // Selected body location cell
    if (selectedBodyToSpecificObjectCell != null) {
      let [iModule, cell] = selectedBodyToSpecificObjectCell;
      selectedBodyPart = 'body';
      selectedLayer = 'bodyToSpecificObject';
      selectedModule = iModule;
      selectedCell = cell;
    }

    return [selectedBodyPart,
            selectedLayer,
            selectedModule,
            selectedCell];
  }

  function calculateHighlightedCells() {
    highlightedCellsByLayer = {};

    // Selected object cell
    if (selectedObjectCell != null) {
      let [iCol, cell] = selectedObjectCell;
      let layer = (iTimestepPhase == 0 )
          ? parsed.timesteps[iTimestep].corticalColumns[iCol].prevObjectLayer
          : parsed.timesteps[iTimestep].corticalColumns[iCol].objectLayer,
          synapsesByPresynapticLayer =
          layer.activeSynapsesByCell[cell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected input cell
    if (selectedInputCell != null) {
      let [iCol, cell] = selectedInputCell;
      let layer = parsed.timesteps[iTimestep].corticalColumns[iCol].featureLocationPair,
          synapsesByPresynapticLayer =
            layer.activeSynapsesByCell[cell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected sensor location cell
    if (selectedSensorToSpecificObjectCell != null) {
      let [iCol, iModule, cell] = selectedSensorToSpecificObjectCell;
      let module = (iTimestepPhase == 0)
          ? parsed.timesteps[iTimestep].corticalColumns[iCol].predictedSensorToSpecificObject.modules[iModule]
          : parsed.timesteps[iTimestep].corticalColumns[iCol].anchoredSensorToSpecificObject.modules[iModule];

      let synapsesByPresynapticLayer =
          module.activeSynapsesByCell[cell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected body location cell
    if (selectedBodyToSpecificObjectCell != null) {
      let [iModule, cell] = selectedBodyToSpecificObjectCell;
      let module = (iTimestepPhase == 0)
          ? parsed.timesteps[iTimestep].predictedBodyToSpecificObject.modules[iModule]
          : parsed.timesteps[iTimestep].anchoredBodyToSpecificObject.modules[iModule];

      let synapsesByPresynapticLayer =
          module.activeSynapsesByCell[cell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }
  }

  function drawHighlightedCells() {
    corticalColumn.select(':scope > .object')
      .datum(objectDatum)
      .call(objectLayer.drawHighlightedCells);

    corticalColumn.select(':scope > .featureLocationPair')
      .datum(featureLocationPairDatum)
      .call(featureLocationPair.drawHighlightedCells);

    corticalColumn.select(':scope > .sensorToBody')
      .datum(sensorToBodyDatum)
      .call(sensorToBody.drawHighlightedCells);

    corticalColumn.select(':scope > .sensorToSpecificObject')
      .datum(sensorToSpecificObjectDatum)
      .call(sensorToSpecificObject.drawHighlightedCells);

    bodyToSpecificObjectNode
      .datum(bodyToSpecificObjectDatum)
      .call(bodyToSpecificObject.drawHighlightedCells);
  }
}

/**
 * Given data of the format:
 * activeCells:
 *   [42, 67, ...]
 * synapsesForActiveCells:
 *   {presynapticLayer1: [[17, 77, 14, 39], // synapses for cell 42
 *                        [13, 40]          // synapses for cell 67
 *                        ...],
 *    ...}
 *
 * Convert it to
 *   {42: {presynapticLayer1: [17, 77, 14, 39]}
 *    67: {presynapticLayer1: [13, 40]}}
 *
 * The first format is optimized for sending over the network. The
 * second format is much easier to visualize.
 */
function getActiveSynapsesByCell(activeCells, synapsesForActiveCells) {
  let activeSynapsesByCell = {};

  activeCells.forEach(cell => {
    activeSynapsesByCell[cell] = {};
  });

  for (let presynapticLayer in synapsesForActiveCells) {
    synapsesForActiveCells[presynapticLayer].forEach((synapses, ci) => {
      activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
    });
  }

  return activeSynapsesByCell;
}

function printRecordingFromUrl(node, logUrl) {
  d3.text(logUrl,
          (error, contents) =>
          printRecording(node, contents));
}

export {
  printRecording,
  printRecordingFromUrl
};
