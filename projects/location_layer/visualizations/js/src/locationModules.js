import * as d3 from 'd3';
import {locationModulesChart} from './charts/locationModulesChart.js';
import {motionChart} from './charts/motionChart.js';
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
 *   deltaLocationInput: {
 *   },
 * };
 */
function parseData(text) {
  let timesteps = [],
      rows = text.split('\n');

  // Row: world dimensions
  let worldDims = JSON.parse(rows[0]);

  let currentTimestep = null,
      didReset = false,
      locationInWorld = null;

  // [{cellDimensions: [5,5], moduleMapDimensions: [20.0, 20.0], orientation: 0.2},
  //  ...]
  let configByModule = JSON.parse(rows[1]).map(d => {
    d.dimensions = {rows: d.cellDimensions[0], cols: d.cellDimensions[1]};
    return d;
  });

  function endTimestep() {
    if (currentTimestep !== null) {
      currentTimestep.worldLocation = locationInWorld;
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

  let i = 2;
  while (i < rows.length) {
    switch (rows[i]) {
    case 'reset':
      didReset = true;
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

        modules.push(Object.assign({cells}, configByModule[i]));
      });

      JSON.parse(rows[i+2]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      currentTimestep.locationLayer = { modules };

      i += 3;
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
    timesteps, worldDims, configByModule
  };
}

let boxes = {
  location: {
    left: 0, top: 20, width: 260, height: 140, text: 'location layer',
    bitsLeft: 0, bitsTop: 0, bitsWidth: 260, bitsHeight: 140,
    decodingsLeft: 0, decodingsTop: 0, decodingsWidth: 0, decodingsHeight: 0
  },
  motion: {
    left: 0, top: 190, width: 260, height: 81, text: 'motion input',
    bitsLeft: 0, bitsTop: 0,
    decodingsLeft: 130, decodingsTop: 36,
    secondary: true
  },
  world: {
    left: 280, top: 34, width: 230, height: 230, text: 'the world'
  }
};

function printRecording(node, text) {
  // Constants
  let margin = {top: 5, right: 15, bottom: 15, left: 15},
      width = 510,
      height = 270,
      parsed = parseData(text);

  // Mutable state
  let iTimestep = 0,
      iLocationModule = null,
      selectedLocationCell = null;

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
        draw(true);
        d3.event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        draw(true);
        d3.event.preventDefault();
        break;
      }
    });

  // Make the SVG a clickable slideshow
  let slideshow = svg.append('g')
      .on('click', () => {
        iTimestep = (iTimestep + 1) % parsed.timesteps.length;
        draw(true);
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
       _] = box.select('.bits')
        .attr('transform', d => `translate(${d.bitsLeft},${d.bitsTop})`)
        .nodes()
        .map(d3.select);
  let [decodedLocationNode,
       motionNode] = box.select('.decodings')
        .attr('transform', d => `translate(${d.decodingsLeft},${d.decodingsTop})`)
        .nodes()
        .map(d3.select);

  let worldNode = container.append('g')
      .attr('transform', `translate(${boxes.world.left}, ${boxes.world.top})`);

  // Label the boxes
  let boxLabel = html.selectAll('.boxLabel')
      .data([boxes.location, boxes.motion, boxes.world]);

  boxLabel.enter()
    .append('div')
      .attr('class', 'boxLabel')
      .style('position', 'absolute')
      .style('text-align', 'left')
      .style('font', '10px Verdana')
      .style('pointer-events', 'none')
    .merge(boxLabel)
      .style('left', d => `${d.left + 17}px`)
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
      motionInput = motionChart(),
      world = worldChart()
        .width(boxes.world.width)
        .height(boxes.world.height)
        .color(parsed.featureColor);

  draw();

  //
  // Lifecycle functions
  //
  function draw(incremental) {
    locationNode.datum({
      modules: parsed.timesteps[iTimestep].locationLayer.modules
    }).call(locationModules);

    motionNode.datum(parsed.timesteps[iTimestep].deltaLocation)
      .call(motionInput);

    worldNode.datum({
      dims: parsed.worldDims,
      location: parsed.timesteps[iTimestep].worldLocation,
      selectedLocationModule: iLocationModule !== null
        ? parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule]
        : null,
      features: [],
      selectedLocationCell
    }).call(world);
  }

  function onLocationCellSelected() {
    worldNode.datum(d => {
      d.selectedLocationModule = iLocationModule !== null
        ? Object.assign(
          {},
          parsed.configByModule[iLocationModule],
          parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule])
        : null;
      d.selectedLocationCell = selectedLocationCell;
      return d;
    }).call(world.drawFiringFields);
  }
}

function printRecordingFromUrl(node, csvUrl) {
  d3.text(csvUrl,
          (error, contents) =>
          printRecording(node, contents));
}

export {
  printRecording,
  printRecordingFromUrl
};
