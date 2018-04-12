import * as d3 from 'd3';
import {arrayOfAxonsChart} from './charts/arrayOfAxonsChart.js';
import {decodedLocationsChart} from './charts/decodedLocationsChart.js';
import {decodedObjectsChart} from './charts/decodedObjectsChart.js';
import {featureChart} from './charts/featureChart.js';
import {layerOfCellsChart} from './charts/layerOfCellsChart.js';
import {locationModulesChart} from './charts/locationModulesChart.js';
// import {moduleDisplacementsChart} from './charts/moduleDisplacementsChart.js';
import {motionChart} from './charts/motionChart.js';
import {timelineChart} from './charts/timelineChart3.js';
import {sensorOnObjectChart} from './charts/sensorOnObjectChart.js';

/**
 *
 * Example timestep:
 * {
 *   locationOnObject: {left: 42.0, top: 12.0},
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
 *         locationLayer: [12, 17, 29]
 *       }
 *     }
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
  let featureColor = d3.scaleOrdinal(d3.schemeCategory10),
      timesteps = [],
      rows = text.split('\n');

  let objects = {},
      objectsSummary = {};

  let currentTimestep = null,
      didReset = false,
      objectPlacements = null,
      learnedObjects = null,
      currentObject = null,
      worldFeatures = null,
      locationOnObject = null;

  // First row: Sensory layer info
  let inputLayerConfig = JSON.parse(rows[0]);

  // Second row: Location layer info
  // [{cellDimensions: [5,5], moduleMapDimensions: [20.0, 20.0], orientation: 0.2},
  //  ...]
  let configByModule = JSON.parse(rows[1]).map(d => {
    d.dimensions = {rows: d.cellDimensions[0], cols: d.cellDimensions[1]};
    return d;
  });

  function endTimestep() {
    if (currentTimestep !== null) {
      currentTimestep.objectPlacements = objectPlacements;
      currentTimestep.worldFeatures = worldFeatures;
      currentTimestep.currentObject = currentObject;
      currentTimestep.locationOnObject = locationOnObject;

      timesteps.push(currentTimestep);
    }

    currentTimestep = null;
  }

  function beginNewTimestep(type) {
    endTimestep();

    currentTimestep = {
      predictedLocation: {modules: configByModule.map(
        c => Object.assign({cells: [], decodings: []}, c))},
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
      endTimestep();
      didReset = true;
      i++;
      break;
    case 'initialSensation':
      beginNewTimestep('initial');
      i++;
      break;
    case 'sensoryRepetition':
      beginNewTimestep('settle');
      currentTimestep.featureInput = timesteps[timesteps.length - 1].featureInput;
      currentTimestep.predictedLocation = timesteps[timesteps.length - 1].anchoredLocation;
      i++;
      break;
    case 'featureInput':
      currentTimestep.featureInput = {
        inputSize: inputLayerConfig.numMinicolumns,
        activeBits: JSON.parse(rows[i+1]),
        decodings: JSON.parse(rows[i+2])
      };

      i += 3;
      break;
    case 'locationOnObject': {
      locationOnObject = JSON.parse(rows[i+1]);

      i += 2;
      break;
    }
    case 'shift': {
      beginNewTimestep('move');
      currentTimestep.deltaLocation = JSON.parse(rows[i+1]);

      let modules = [];
      JSON.parse(rows[i+2]).forEach((phaseDisplacement, i) => {
        modules.push(Object.assign({phaseDisplacement: {top: phaseDisplacement[0],
                                                        left: phaseDisplacement[1]},
                                    activeSynapsesByCell: {}},
                                   configByModule[i]));
      });

      JSON.parse(rows[i+3]).forEach((activeCells, i) => {
        let cells = activeCells.map(cell => {
          return {
            cell,
            state: 'active'
          };
        });

        modules[i].cells = cells;
      });

      JSON.parse(rows[i+4]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      let decodings = JSON.parse(rows[i+5]).map(
        ([objectName, iFeature, amountContained]) => {
          let {top, left} = getFeatureCenter(objects[objectName][iFeature]);
          return { objectName, top, left, amountContained };
        });
      currentTimestep.predictedLocation = { modules, decodings };

      i += 6;
      break;
    }
    case 'locationLayer': {
      let modules = [];

      JSON.parse(rows[i+1]).forEach((module, i) => {
        let [activeCells, segmentsForActiveCells] = module;

        let cells = activeCells.map(cell => {
          return {
            cell,
            state: 'active'
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
        ([objectName, iFeature, amountContained]) => {
          let {top, left} = getFeatureCenter(objects[objectName][iFeature]);
          return { objectName, top, left, amountContained };
        });
      currentTimestep.anchoredLocation = { modules, decodings };

      i += 4;
      break;
    }
    case 'predictedFeatureLocationPair': {
      let [predictedCells, segmentsForPredictedCells] = JSON.parse(rows[i+1]),
          decodings = JSON.parse(rows[i+2]);

      let cells = predictedCells.map(cell => {
        return {
          cell,
          state: 'predicted'
        };
      });

      let activeSynapsesByCell = {};
      if (segmentsForPredictedCells) {
        predictedCells.forEach(cell => {
          activeSynapsesByCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForPredictedCells) {
          segmentsForPredictedCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            activeSynapsesByCell[predictedCells[ci]][presynapticLayer] = synapses;
          });
        }
      }

      let predictedCellDecodings = decodings.map(
        ([objectName, iFeature, amountContained]) => {
          let {top, left} = getFeatureCenter(objects[objectName][iFeature]);
          return { objectName, top, left, amountContained };
        });

      currentTimestep.predictedInput = {
        predictedCells,
        activeSynapsesByCell,
        decodings: predictedCellDecodings,
        cells: cells,
        dimensions: {rows: inputLayerConfig.cellsPerColumn,
                     cols: inputLayerConfig.numMinicolumns}
      };

      i += 3;
      break;
    }
    case 'featureLocationPair': {
      let activeSynapsesByCell = {};

      let [activeCells, segmentsForActiveCells] = JSON.parse(rows[i+1]);

      let cells = activeCells.map(cell => {
        return {
          cell,
          state: 'active'
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

      let activeCellDecodings = JSON.parse(rows[i+2]);

      let activeCellDecodings2 = activeCellDecodings.map(
        ([objectName, iFeature, amountContained]) => {
          let {top, left} = getFeatureCenter(objects[objectName][iFeature]);
          return { objectName, top, left, amountContained };
        });

      currentTimestep.inputLayer = {
        cells, activeSynapsesByCell,
        decodings: activeCellDecodings2,
        dimensions: {rows: inputLayerConfig.cellsPerColumn,
                     cols: inputLayerConfig.numMinicolumns},
        predictedCells: []
      };

      i += 3;
      break;
    }
    case 'currentObject': {
      currentObject = JSON.parse(rows[i+1]);
      worldFeatures = [];
      currentObject.features.forEach(f => {
        worldFeatures.push({
            name: f.name,
            width: f.width,
            height: f.height,
            top: f.top,
            left: f.left
        });
      });

      i += 2;
      break;
    }
    case 'learnedObjects': {
      learnedObjects = JSON.parse(rows[i+1]);

      learnedObjects.forEach(({name, top, left, features}) => {
        objects[name] = features;

        let leftMost = Infinity,
            rightMost = -Infinity,
            topMost = Infinity,
            bottomMost = -Infinity;

        features.forEach(f => {
          leftMost = Math.min(leftMost, f.left);
          rightMost = Math.max(rightMost, f.left + f.width);
          topMost = Math.min(topMost, f.top);
          bottomMost = Math.max(bottomMost, f.top + f.height);
        });

        objectsSummary = {leftMost, rightMost, topMost, bottomMost};

      });

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
    timesteps, configByModule, featureColor, objects, objectsSummary
  };
}

function getFeatureCenter(feature) {
  return {top: feature.top + (feature.height / 2),
          left: feature.left + (feature.width / 2)};
}

let firstRowTop = 24,
    layerHeight = 120,
    secondRowTop = firstRowTop + layerHeight + 40,
    thirdRowTop = secondRowTop + 100,
    layerWidth = 200,
    worldLeft = 400,
    worldTop = firstRowTop,
    decodingsWidth = 142, // 76,
    decodingsHeight = layerHeight - 8;

let layout = {
  smallLabels: [
    {html: "sensor's location in object's space",
     topRightCorner: {left: 0, top: secondRowTop + layerHeight/2 - 10}},
    {html: 'sensory input at location',
     topRightCorner: {left: 0, top: firstRowTop + layerHeight/2 - 10}},
    // {html: 'motor displacement<br />(by module)',
    //  topRightCorner: {left: 0, top: thirdRowTop}},
    {html: 'sensory input',
     topRightCorner: {left: 0, top: firstRowTop + layerHeight}}
  ],

  mediumLabels: [
    {
      html: 'Two-Layer Network',
      top: firstRowTop - 16, left: 5
    },
    {
      html: 'An Object And A Sensor',
      left: worldLeft + 5, top: worldTop - 16
    }],

  location: {
    left: 0, top: secondRowTop, width: layerWidth, height: layerHeight
  },
  decodedLocation: {
    left: layerWidth + 10, top: secondRowTop, width: decodingsWidth, height: decodingsHeight
  },

  input: {
    left: 0, top: firstRowTop, width: layerWidth, height: layerHeight
  },
  decodedInput: {
    left: layerWidth + 10, top: firstRowTop, width: decodingsWidth, height: decodingsHeight
  },

  motion: {
    left: layerWidth / 2, top: thirdRowTop + 40, width: layerWidth, height: 81,
    secondary: true
  },
  moduleDisplacements: {
    left: 0, top: thirdRowTop, width: layerWidth, height: 55
  },

  feature: {
    left: 0, top: firstRowTop + layerHeight, width: layerWidth, height: 5,
    secondary: true
  },
  decodedFeature: {
    left: layerWidth + 10, top: firstRowTop + layerHeight - 2, width: 10, height: 10
  },

  world: {
    left: worldLeft, top: worldTop, width: 120, height: 120
  }
};

function printRecording(node, text) {
  // Constants
  let margin = {top: 5, right: 5, bottom: 15, left: 5},
      width = 600,
      height = thirdRowTop + layerHeight/2,
      parsed = parseData(text);

  // Mutable state
  let iTimestep = 0,
      iTimestepPhase = 1,
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
      htmlSelectable = html.append('div')
        .style('margin-left', '-88px')
        .style('padding-left', '103px')
        .style('margin-right', '-20px')
        .style('padding-right', '20px'),
      svg = htmlSelectable.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);


  function goBackOnePhase() {
    for (let i = 0; i < 2; i++) {
      iTimestepPhase--;
      if (iTimestepPhase < 0) {
        iTimestep--;
        iTimestepPhase = 1;

        if (iTimestep < 0) {
          iTimestep = parsed.timesteps.length - 1;
        }
      }

      if (parsed.timesteps[iTimestep].type == 'initial' &&
          iTimestepPhase == 0) {
        // Go back another step if we landed in phase 0 of an initial
        // sense.
        continue;
      }

      break;
    }

    onSelectedTimestepChanged();
  }


  function goForwardOnePhase() {
    for (let i = 0; i < 2; i++) {
      iTimestepPhase++;
      if (iTimestepPhase > 1) {
        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        iTimestepPhase = 0;
      }

      if (parsed.timesteps[iTimestep].type == 'initial' &&
          iTimestepPhase == 0) {
        // Go back another step if we landed in phase 0 of an initial
        // sense.
        continue;
      }

      break;
    }

    onSelectedTimestepChanged();
  }

  // Add keyboard navigation
  htmlSelectable
    .attr('tabindex', 0)
    .on('keydown', function() {
      switch (d3.event.keyCode) {
      case 37: // Left
      case 38: // Up:
        goBackOnePhase();
        d3.event.preventDefault();
        break;

        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        onSelectedTimestepChanged();
        d3.event.preventDefault();
        break;
      case 39: // Right
      case 40: // Down
        goForwardOnePhase();
        d3.event.preventDefault();
        break;
      }
    });

  // Make the SVG a clickable slideshow
  let slideshow = svg.append('g')
      .on('click', () => {
        goForwardOnePhase();
      });

  slideshow.append('rect')
      .attr('fill', 'transparent')
      .attr('stroke', 'none')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.bottom + margin.top + 10);

  let container = slideshow
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

  let [locationNode,
       decodedLocationNode,
       inputNode,
       decodedInputNode,
       // moduleDisplacementsNode,
       motionNode,
       featureNode,
       decodedFeatureNode] =
      ['location', 'decodedLocation',
       'input', 'decodedInput',
       // 'moduleDisplacements',
       'motion',
       'feature', 'decodedFeature'].map(layer => {
         let lay = layout[layer];
         return container.append('g')
           .attr('class', layer)
           .attr('transform', `translate(${lay.left},${lay.top})`);
       });


  let worldNode = container.append('g')
      .attr('transform', `translate(${layout.world.left}, ${layout.world.top})`);

  let timelineNode = htmlSelectable
      .append('div')
      .style('padding-top', '5px')
      .style('margin-left', '-80px')
      // .style('padding-left', '17px') // Because it hangs some text off the side.
      // .style('padding-right', '17px')
      .style('text-align', 'center');

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
        .style('font', '13px Helvetica')
        .style('pointer-events', 'none')
        .style('width', '100px')
        .style('right', 0);
    })
    .merge(smallLabel)
    .style('top', d => `${d.topRightCorner.top}px`)
    .style('left', d => `${15 + d.topRightCorner.left}px`);
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
    .style('font', '13px Helvetica')
    .style('font-weight', 'bold')
    .style('pointer-events', 'none')
    .merge(mediumLabel)
    .style('left', d => `${15 + d.left}px`)
    .style('top', d => `${d.top}px`)
    .html(d => d.html);

  // Configure the charts
  let locationModules = locationModulesChart()
        .width(layout.location.width)
        .height(layout.location.height)
        .color(d3.scaleOrdinal()
               .domain(['active'])
               .range(['black',
                       // 'lightgray'
                      ]))
        .onCellSelected((iModule, cell) => {
          iLocationModule = iModule;
          selectedLocationCell = cell;
          onLocationCellSelected();
        }),
      locationDatum = () => {
        return {
          modules: (iTimestepPhase == 0)
            ? parsed.timesteps[iTimestep].predictedLocation.modules
            : parsed.timesteps[iTimestep].anchoredLocation.modules,
          highlightedCells: highlightedCellsByLayer['locationLayer'] || []
        };
      },
      decodedLocation = decodedLocationsChart()
        .width(layout.decodedLocation.width)
        .height(layout.decodedLocation.height)
        .color(parsed.featureColor),
      decodedLocationDatum = () => {
        return {
          decodings: (iTimestepPhase == 0)
            ? parsed.timesteps[iTimestep].predictedLocation.decodings
            : parsed.timesteps[iTimestep].anchoredLocation.decodings,
          objects: parsed.objects
        };
      },
      inputLayer = layerOfCellsChart()
        .width(layout.input.width)
        .height(layout.input.height)
        .columnMajorIndexing(true)
        .onCellSelected(cell => {
          selectedInputCell = cell;
          onInputCellSelected();
        }),
      inputDatum = () => Object.assign(
        {highlightedCells: highlightedCellsByLayer['inputLayer'] || []},
        (iTimestepPhase == 0)
          ? parsed.timesteps[iTimestep].predictedInput
          : parsed.timesteps[iTimestep].inputLayer),
      decodedInput = decodedLocationsChart()
        .width(layout.decodedInput.width)
        .height(layout.decodedInput.height)
        .color(parsed.featureColor),
      decodedInputDatum = () => {
        return {
          decodings: (iTimestepPhase == 0)
            ? parsed.timesteps[iTimestep].predictedInput.decodings
            : parsed.timesteps[iTimestep].inputLayer.decodings,
          objects: parsed.objects
        };
      },
      featureInput = arrayOfAxonsChart()
        .width(layout.feature.width)
      .height(layout.feature.height),
      featureInputDatum = () => (iTimestepPhase == 1)
        ? parsed.timesteps[iTimestep].featureInput
        : null,
      decodedFeature = featureChart()
        .color(parsed.featureColor)
        .width(layout.decodedFeature.width)
        .height(layout.decodedFeature.height),
      decodedFeatureDatum = () => {
        return (iTimestepPhase == 1)
          ? {name: parsed.timesteps[iTimestep].featureInput.decodings[0]}
          : null;
      },
      // moduleDisplacements = moduleDisplacementsChart()
      //   .width(layout.moduleDisplacements.width)
      //   .height(layout.moduleDisplacements.height),
      // moduleDisplacementsDatum = () => {
      //   return {
      //     modules: (iTimestepPhase == 0)
      //       ? parsed.timesteps[iTimestep].predictedLocation.modules
      //       : parsed.timesteps[iTimestep].anchoredLocation.modules
      //   };
      // },
      motionInput = motionChart(),
      motionDatum = () => (iTimestepPhase == 0)
        ? parsed.timesteps[iTimestep].deltaLocation
        : null,
      world = sensorOnObjectChart()
        .width(layout.world.width)
        .height(layout.world.height)
        .xScale(d3.scaleLinear()
                .domain([parsed.objectsSummary.leftMost,
                         parsed.objectsSummary.rightMost])
                .range([20, layout.world.width - 20]))
        .yScale(d3.scaleLinear()
                .domain([parsed.objectsSummary.topMost,
                         parsed.objectsSummary.bottomMost])
                .range([20, layout.world.height - 20]))
        .color(parsed.featureColor),
      worldDatum = () => {
        return {
          location: parsed.timesteps[iTimestep].locationOnObject,
          selectedLocationModule: iLocationModule !== null
            ? (iTimestepPhase == 0)
              ? parsed.timesteps[iTimestep].predictedLocation.modules[iLocationModule]
              : parsed.timesteps[iTimestep].anchoredLocation.modules[iLocationModule]
            : null,
          features: parsed.timesteps[iTimestep].worldFeatures,
          selectedLocationCell
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
    locationNode.datum(locationDatum)
      .call(locationModules);
    decodedLocationNode.datum(decodedLocationDatum)
      .call(decodedLocation);

    inputNode.datum(inputDatum)
      .call(inputLayer);
    decodedInputNode.datum(decodedInputDatum)
      .call(decodedInput);

    featureNode.datum(featureInputDatum)
      .call(featureInput);
    decodedFeatureNode.datum(decodedFeatureDatum)
      .call(decodedFeature);

    // moduleDisplacementsNode.datum(moduleDisplacementsDatum)
    //   .call(moduleDisplacements);
    motionNode.datum(motionDatum)
      .call(motionInput);

    worldNode.datum(worldDatum).call(world);

    timelineNode.datum(timelineDatum)
      .call(incremental ? timeline.drawSelectedStep : timeline);
  }

  function onSelectedTimestepChanged() {
    calculateHighlightedCells();
    drawHighlightedCells();
    draw(true);
  }

  function onLocationCellSelected() {
    if (iLocationModule != null) {
      let config = parsed.configByModule[iLocationModule],
          module = (iTimestepPhase == 0)
            ? parsed.timesteps[iTimestep].predictedLocation.modules[iLocationModule]
            : parsed.timesteps[iTimestep].anchoredLocation.modules[iLocationModule];

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
      let module = (iTimestepPhase == 0)
          ? parsed.timesteps[iTimestep].predictedLocation.modules[iLocationModule]
          : parsed.timesteps[iTimestep].anchoredLocation.modules[iLocationModule];

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
    locationNode.datum(locationDatum)
      .call(locationModules.drawHighlightedCells);

    inputNode.datum(inputDatum)
      .call(inputLayer.drawHighlightedCells);
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
