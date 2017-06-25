import * as d3 from 'd3';
import {arrayOfAxonsChart} from './charts/arrayOfAxonsChart.js';
import {decodedLocationsChart} from './charts/decodedLocationsChart.js';
import {decodedObjectsChart} from './charts/decodedObjectsChart.js';
import {featureChart} from './charts/featureChart.js';
import {layerOfCellsChart} from './charts/layerOfCellsChart.js';
import {locationModulesChart} from './charts/locationModulesChart.js';
import {motionChart} from './charts/motionChart.js';
import {timelineChart} from './charts/timelineChart.js';
import {worldChart} from './charts/worldChart.js';

/**
 *
 * Example timestep:
 * {
 *   worldLocation: {left: 42.0, top: 12.0},
 *   reset: null,
 *   locationLayer: {
 *     modules: [
 *       {activeCells: [],
 *        activePoints: [],
 *        activeSynapsesByCell: {}},
 *     ]
 *   },
 *   inputLayer: {
 *     activeCells: [],
 *     decodings: [],
 *     activeSynapsesByCell: {
 *       42: {
 *         locationLayer: [12, 17, 29],
 *         objectLayer: [42, 45]
 *       }
 *     }
 *   },
 *   objectLayer: {
 *     activeCells: [],
 *     decodings: [],
 *     activeSynapsesByCell: {}
 *   },
 *   deltaLocationInput: {
 *   },
 *   featureInput: {
 *     inputSize: 150,
 *     activeBits: [],
 *     decodings: []
 *   }
 * };
 */
function parseData(text) {
  let featureColor = d3.scaleOrdinal(),
      timesteps = [],
      rows = text.split('\n');

  // Row: world dimensions
  let worldDims = JSON.parse(rows[0]);

  // Row: Features and colors
  //   {'A': 'red',
  //    'B': 'blue',
  //    'C': 'gray'}
  let featureColorMapping = JSON.parse(rows[1]);

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
  let objects = JSON.parse(rows[2]);

  let currentTimestep = null,
      didReset = false,
      objectPlacements = null,
      worldFeatures = null,
      locationInWorld = null;

  // [{cellDimensions: [5,5], moduleMapDimensions: [20.0, 20.0], orientation: 0.2},
  //  ...]
  let configByModule = JSON.parse(rows[3]).map(d => {
    d.dimensions = {rows: d.cellDimensions[0], cols: d.cellDimensions[1]};
    return d;
  });

  function endTimestep() {
    if (currentTimestep !== null) {
      currentTimestep.objectPlacements = objectPlacements;
      currentTimestep.worldFeatures = worldFeatures;

      if (currentTimestep.type == 'move') {
        currentTimestep.worldLocation = locationInWorld;
        currentTimestep.featureInput = {
          inputSize: 150,
          activeBits: [],
          decodings: []
        };

        // Continue showing the previous object layer.
        for (let i = timesteps.length - 1; i >= 0; i--) {
          if (timesteps[i].type !== 'move') {
            currentTimestep.objectLayer =
              Object.assign({}, timesteps[i].objectLayer);
            currentTimestep.objectLayer.activeSynapsesByCell = {};
            break;
          }
        }
      }

      timesteps.push(currentTimestep);
    }

    currentTimestep = null;
  }

  function beginNewTimestep(type) {
    endTimestep();

    currentTimestep = {
      worldLocation: locationInWorld,
      type
    };

    if (didReset) {
      currentTimestep.reset = true;
      didReset = false;
    }
  }

  let i = 4;
  while (i < rows.length) {
    switch (rows[i]) {
    case 'reset':
      didReset = true;
      i++;
      break;
    case 'sense':
      beginNewTimestep('sense');

      currentTimestep.featureInput = {
        inputSize: 150,
        activeBits: JSON.parse(rows[i+1]),
        decodings: JSON.parse(rows[i+2])
      };

      i += 3;
      break;
    case 'sensoryRepetition':
      beginNewTimestep('repeat');
      currentTimestep.featureInput = timesteps[timesteps.length - 1].featureInput;
      i++;
      break;
    case 'move': {
      beginNewTimestep('move');
      let deltaLocation = JSON.parse(rows[i+1]);

      currentTimestep.deltaLocation = {
        top: deltaLocation[0],
        left: deltaLocation[1]
      };

      i += 2;
      break;
    }
    case 'locationInWorld': {
      let location = JSON.parse(rows[i+1]);

      locationInWorld = {top: location[0], left: location[1]};

      i += 2;
      break;
    }
    case 'shift': {
      let modules = [];
      JSON.parse(rows[i+1]).forEach((activeCells, i) => {
        let cells = activeCells.map(cell => {
          return {
            cell,
            state: 'predicted-active'
          };
        });

        modules.push(Object.assign({cells,
                                    activeSynapsesByCell: {}},
                                   configByModule[i]));
      });

      JSON.parse(rows[i+2]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      let decodings = JSON.parse(rows[i+3]).map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });
      currentTimestep.locationLayer = { modules, decodings };

      i += 4;
      break;
    }
    case 'locationLayer': {
      let modules = [];

      JSON.parse(rows[i+1]).forEach((module, i) => {
        let [activeCells, segmentsForActiveCells] = module;

        let prevActiveCells = (currentTimestep.reset || timesteps.length == 0)
            ? []
            : timesteps[timesteps.length-1].locationLayer.modules[i].cells.map(
              d => d.cell);

        let cells = activeCells.map(cell => {
          return {
            cell,
            state: prevActiveCells.indexOf(cell) != -1
              ? 'predicted-active'
              : 'active'
          };
        });

        let activeSynapsesByCell = {};

        if (segmentsForActiveCells) {

          activeCells.forEach(cell => {
            activeSynapsesByCell[cell] = {};
          });

          for (let presynapticLayer in segmentsForActiveCells) {
            segmentsForActiveCells[presynapticLayer].forEach((segments, ci) => {
              let synapses = [];
              segments.forEach(presynapticCells => {
                synapses = synapses.concat(presynapticCells);
              });

              activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
            });
          }
        }

        modules.push(Object.assign({cells, activeSynapsesByCell},
                                   configByModule[i]));
      });

      JSON.parse(rows[i+2]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      let decodings = JSON.parse(rows[i+3]).map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });
      currentTimestep.locationLayer = { modules, decodings };

      i += 4;
      break;
    }
    case 'inputLayer': {
      let activeSynapsesByCell = {};

      let [activeCells, predictedCells, segmentsForActiveCells,
           segmentsForPredictedCells] = JSON.parse(rows[i+1]);

      let cells = activeCells.map(cell => {
        return {
          cell,
          state: predictedCells.indexOf(cell) != -1
            ? 'predicted-active'
            : 'active'
        };
      });

      if (segmentsForActiveCells) {
        activeCells.forEach(cell => {
          activeSynapsesByCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForActiveCells) {
          segmentsForActiveCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
          });
        }
      }

      let {activeCellDecodings, predictedCellDecodings} = JSON.parse(rows[i+2]);

      let activeCellDecodings2 = activeCellDecodings.map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });
      let predictedCellDecodings2 = predictedCellDecodings.map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });

      currentTimestep.inputLayer = {
        cells, activeSynapsesByCell,
        decodings: activeCellDecodings2,
        dimensions: {rows: 32, cols: 150},
        predictedCells: []
      };

      if (timesteps.length > 0 &&
          timesteps[timesteps.length - 1].type == 'move') {
        let prevTimestep = timesteps[timesteps.length - 1];

        let synapsesByPredictedCell = {};

        let cells2 = predictedCells.map(cell => {
          return {
            cell,
            state: 'predicted'
          };
        });

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

        prevTimestep.inputLayer = {
          predictedCells,
          activeSynapsesByCell: synapsesByPredictedCell,
          decodings: predictedCellDecodings2,
          cells: cells2,
          dimensions: {rows: 32, cols: 150}
        };
      }

      i += 3;
      break;
    }
    case 'objectLayer': {
      let [activeCells, segmentsForActiveCells] = JSON.parse(rows[i+1]);

      let prevActiveCells = (currentTimestep.reset || timesteps.length == 0)
          ? []
          : timesteps[timesteps.length-1].objectLayer.cells.map(d => d.cell);

      let cells = activeCells.map(cell => {
        return {
          cell,
          state: prevActiveCells.indexOf(cell) != -1
            ? 'predicted-active'
            : 'active'
        };
      });

      let activeSynapsesByCell = {};
      if (segmentsForActiveCells) {

        activeCells.forEach(cell => {
          activeSynapsesByCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForActiveCells) {
          segmentsForActiveCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
          });
        }
      }

      let decodings = JSON.parse(rows[i+2]);
      currentTimestep.objectLayer = Object.assign(
        {cells, activeSynapsesByCell, decodings},
        {dimensions: {rows: 16, cols: 256}});
      i += 3;
      break;
    }
    case 'objectPlacements': {
      objectPlacements = JSON.parse(rows[i+1]);

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

let secondColumnLeft = 180,
    secondRowTop = 220,
    thirdRowTop = 428,
    columnWidth = 170;

let boxes = {
  location: {
    left: 0, top: secondRowTop, width: columnWidth, height: 180, text: 'location layer',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 60,
    decodingsLeft: 20, decodingsTop: 90, decodingsWidth: 148, decodingsHeight: 90
  },
  input: {
    left: secondColumnLeft, top: secondRowTop, width: columnWidth, height: 180, text: 'feature-location pair layer',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 60,
    decodingsLeft: 20, decodingsTop: 90, decodingsWidth: 148, decodingsHeight: 90
  },
  object: {
    left: secondColumnLeft, top: 12, width: columnWidth, height: 180, text: 'object layer',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 60,
    decodingsLeft: 20, decodingsTop: 90, decodingsWidth: 148, decodingsHeight: 90
  },
  motion: {
    left: 0, top: thirdRowTop, width: columnWidth, height: 81, text: 'motion input',
    bitsLeft: 0, bitsTop: 0,
    decodingsLeft: 85, decodingsTop: 36,
    secondary: true
  },
  feature: {
    left: secondColumnLeft, top: thirdRowTop, width: columnWidth, height: 81, text: 'feature input',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 5,
    decodingsLeft: 65, decodingsTop: 30,
    secondary: true
  },
  world: {
    left: 370, top: 120, width: 230, height: 230, text: 'the world'
  }
};

function printRecording(node, text) {
  // Constants
  let margin = {top: 5, right: 5, bottom: 15, left: 5},
      width = 600,
      height = 495,
      parsed = parseData(text);

  // Mutable state
  let iTimestep = 0,
      iLocationModule = null,
      selectedLocationCell = null,
      selectedInputCell = null,
      selectedObjectCell = null,
      highlightedCellsByLayer = {};

  // Allow a mix of SVG and HTML
  let html = d3.select(node)
        .append('div')
          .style('margin-left', 'auto')
          .style('margin-right', 'auto')
          .style('position', 'relative')
          .style('width', `${width + margin.left + margin.right}px`),
      svg = html.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

  // Add keyboard navigation
  html
    .attr('tabindex', 0)
    .on('keydown', function() {
      switch (d3.event.keyCode) {
      case 37: // Left
        iTimestep--;
        if (iTimestep < 0) {iTimestep = parsed.timesteps.length - 1;}
        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      }
    });

  // Make the SVG a clickable slideshow
  let slideshow = svg.append('g')
      .on('click', () => {
        iTimestep = (iTimestep + 1) % parsed.timesteps.length;
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
  let box = container.selectAll('.box')
      .data([boxes.location,
             boxes.input,
             boxes.object,
             boxes.feature,
             boxes.motion]);

  box = box.enter().append('g')
    .attr('class', 'layerBox')
    .call(g => {
      g.append('rect')
        .attr('class', 'border')
        .attr('fill', 'none');

      g.append('g')
        .attr('class', 'bits');

      g.append('g')
        .attr('class', 'decodings');
    })
    .merge(box)
    .attr('transform', d => `translate(${d.left}, ${d.top})`);

  box.select('.border')
    .attr('width', d => d.width)
    .attr('height', d => d.height)
    .attr('stroke', d => d.secondary ? 'gray' : 'lightgray')
    .attr('stroke-width', d => d.secondary ? 1 : 3)
    .attr('stroke-dasharray', d => d.secondary ? "5,5" : null);

  let [locationNode,
       inputNode,
       objectNode,
       featureNode,
       _] = box.select('.bits')
        .attr('transform', d => `translate(${d.bitsLeft},${d.bitsTop})`)
        .nodes()
        .map(d3.select);
  let [decodedLocationNode,
       decodedInputNode,
       decodedObjectNode,
       decodedFeatureNode,
       motionNode] = box.select('.decodings')
        .attr('transform', d => `translate(${d.decodingsLeft},${d.decodingsTop})`)
        .nodes()
        .map(d3.select);

  let worldNode = container.append('g')
      .attr('transform', `translate(${boxes.world.left}, ${boxes.world.top})`);
  svg.append('line')
    .attr('stroke', 'gray')
    .attr('stroke-width', 1)
    .attr('x1', boxes.world.left - 5)
    .attr('y1', 10)
    .attr('x2', boxes.world.left - 5)
    .attr('y2', 522);

  let timelineNode = html
      .append('div')
      .style('padding-top', '5px')
      .style('padding-left', '17px') // Because it hangs some text off the side.
      .style('padding-right', '17px')
      .style('text-align', 'center');

  // Label the boxes
  let boxLabel = html.selectAll('.boxLabel')
      .data([boxes.location, boxes.input, boxes.object, boxes.motion,
             boxes.feature, boxes.world]);

  boxLabel.enter()
    .append('div')
      .attr('class', 'boxLabel')
      .style('position', 'absolute')
      .style('text-align', 'left')
      .style('font', '10px Verdana')
      .style('pointer-events', 'none')
    .merge(boxLabel)
      .style('left', d => `${d.left + 7}px`)
      .style('top', d => `${d.top - 9}px`)
      .text(d => d.text);

  // Configure the charts
  let locationModules = locationModulesChart()
        .width(boxes.location.bitsWidth)
        .height(boxes.location.bitsHeight)
        .onCellSelected((iModule, cell) => {
          iLocationModule = iModule;
          selectedLocationCell = cell;
          onLocationCellSelected();
        }),
      decodedLocation = decodedLocationsChart()
        .width(boxes.location.decodingsWidth)
        .height(boxes.location.decodingsHeight)
        .color(parsed.featureColor),
      inputLayer = layerOfCellsChart()
        .width(boxes.input.bitsWidth)
        .height(boxes.input.bitsHeight)
        .columnMajorIndexing(true)
        .onCellSelected(cell => {
          selectedInputCell = cell;
          onInputCellSelected();
        }),
      decodedInput = decodedLocationsChart()
        .width(boxes.input.decodingsWidth)
        .height(boxes.input.decodingsHeight)
        .color(parsed.featureColor),
      objectLayer = layerOfCellsChart()
        .width(boxes.object.bitsWidth)
        .height(boxes.object.bitsHeight)
        .onCellSelected(cell => {
          selectedObjectCell = cell;
          onObjectCellSelected();
        }),
      decodedObject = decodedObjectsChart()
        .width(boxes.object.decodingsWidth)
        .height(boxes.object.decodingsHeight)
        .color(parsed.featureColor),
      featureInput = arrayOfAxonsChart()
        .width(boxes.feature.bitsWidth)
        .height(boxes.feature.bitsHeight),
      decodedFeature = featureChart()
        .color(parsed.featureColor)
        .width(40)
        .height(40),
      motionInput = motionChart(),
      world = worldChart()
        .width(boxes.world.width)
        .height(boxes.world.height)
        .color(parsed.featureColor),
      timeline = timelineChart().onchange(iTimestepNew => {
        iTimestep = iTimestepNew;
        onSelectedTimestepChanged();
      });

  calculateHighlightedCells();
  draw();

  //
  // Lifecycle functions
  //
  function draw(incremental) {
    locationNode.datum({
      modules: parsed.timesteps[iTimestep].locationLayer.modules,
      highlightedCells: highlightedCellsByLayer['locationLayer'] || []
    }).call(locationModules);
    decodedLocationNode.datum({
      decodings: parsed.timesteps[iTimestep].locationLayer.decodings,
      objects: parsed.objects
    }).call(decodedLocation);

    inputNode.datum(
      Object.assign(
        {highlightedCells: highlightedCellsByLayer['inputLayer'] || []},
        parsed.timesteps[iTimestep].inputLayer))
      .call(inputLayer);
    decodedInputNode.datum({
      decodings: parsed.timesteps[iTimestep].inputLayer.decodings,
      objects: parsed.objects
    }).call(decodedInput);

    objectNode.datum(
      Object.assign(
        {highlightedCells: highlightedCellsByLayer['objectLayer'] || []},
        parsed.timesteps[iTimestep].objectLayer))
      .call(objectLayer);
    decodedObjectNode.datum({
      decodings: parsed.timesteps[iTimestep].objectLayer.decodings,
      objects: parsed.objects
    }).call(decodedObject);

    featureNode.datum(parsed.timesteps[iTimestep].featureInput)
      .call(featureInput);
    decodedFeatureNode.datum(
      {name: parsed.timesteps[iTimestep].featureInput.decodings[0]})
      .call(decodedFeature);

    motionNode.datum(parsed.timesteps[iTimestep].deltaLocation)
      .call(motionInput);

    worldNode.datum({
      dims: parsed.worldDims,
      location: parsed.timesteps[iTimestep].worldLocation,
      selectedLocationModule: iLocationModule !== null
        ? parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule]
        : null,
      features: parsed.timesteps[iTimestep].worldFeatures,
      selectedLocationCell
    }).call(world);

    timelineNode.datum({
      timesteps: parsed.timesteps,
      selectedIndex: iTimestep
    }).call(incremental ? timeline.drawSelectedStep : timeline);
  }

  function onSelectedTimestepChanged() {
    calculateHighlightedCells();
    drawHighlightedCells();
    draw(true);
  }

  function onLocationCellSelected() {
    if (iLocationModule != null) {
      let config = parsed.configByModule[iLocationModule],
          module = parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule];

      worldNode.datum(d => {
        d.selectedLocationModule = Object.assign({}, config, module);
        d.selectedLocationCell = selectedLocationCell;
        return d;
      });

      let synapsesByPresynapticLayer =
          module.activeSynapsesByCell[selectedLocationCell];
      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    } else {
      worldNode.datum(d => {
        d.selectedLocationModule = null;
        d.selectedLocationCell = null;
        return d;
      });
    }

    worldNode.call(world.drawFiringFields);

    calculateHighlightedCells();
    drawHighlightedCells();
  }

  function onInputCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();
  }

  function onObjectCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();
  }

  function calculateHighlightedCells() {
    highlightedCellsByLayer = {};

    // Selected location cell
    if (iLocationModule != null) {
      let module = parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule];

      let synapsesByPresynapticLayer =
          module.activeSynapsesByCell[selectedLocationCell];
      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected input cell
    if (selectedInputCell != null) {
      let layer = parsed.timesteps[iTimestep].inputLayer,
          synapsesByPresynapticLayer =
            layer.activeSynapsesByCell[selectedInputCell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected object cell
    if (selectedObjectCell != null) {
      let layer = parsed.timesteps[iTimestep].objectLayer,
          synapsesByPresynapticLayer =
            layer.activeSynapsesByCell[selectedObjectCell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }
  }

  function drawHighlightedCells() {
    locationNode.datum(d => {
      d.highlightedCells = highlightedCellsByLayer['locationLayer'] || [];
      return d;
    }).call(locationModules.drawHighlightedCells);

    inputNode.datum(d => {
      d.highlightedCells = highlightedCellsByLayer['inputLayer'] || [];
      return d;
    }).call(inputLayer.drawHighlightedCells);

    objectNode.datum(d => {
      d.highlightedCells = highlightedCellsByLayer['objectLayer'] || [];
      return d;
    }).call(objectLayer.drawHighlightedCells);
  }
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
