import * as d3 from "d3";

function Grid2dLayout(nColumns, nRows, left, top, width, height, padding) {
  this.nColumns = nColumns;
  this.nRows = nRows;
  this.left = left;
  this.top = top;
  this.width = width;
  this.height = height;
  this.columnWidth = width / nColumns;
  this.rowHeight = height / nRows;
  this.padding = padding;
}

Grid2dLayout.prototype.getBitTopLeft = function(bitId) {
  var column = Math.floor(bitId / this.nRows),
      row = bitId % this.nRows;
  return [column * this.columnWidth, row * this.rowHeight];
};

Grid2dLayout.prototype.getBitCenter = function(bitId) {
  var ret = this.getBitTopLeft(bitId);
  ret[0] += this.columnWidth/2;
  ret[1] += this.rowHeight/2;
  return ret;
};

Grid2dLayout.prototype.getBitAtPoint = function(x,y) {
  var column = Math.floor(x / this.columnWidth),
      row = Math.floor(y / this.rowHeight),
      bitId = column*this.nRows + row;

  return bitId;
};

function selectedCellPlot() {
  var distalSegmentWidth = 15,
      distalSegmentHeight = 3,
      selectedCellR = 7,
      layout,
      layouts;

  var chart = function chart (selection) {
    selection.each(function(selectedCells) {
      var hoverContainer = d3.select(this);

      var selectedCell = hoverContainer.selectAll('.selectedCell')
          .data(selectedCells);

      selectedCell.exit()
        .remove();

      selectedCell = selectedCell.enter()
        .append('g')
        .attr('class', 'selectedCell')
        .call(enter => {
          enter
            .append('g')
            .attr('class', 'cellIcon')
            .append('polygon')
            .attr('points', '0,-8 7,8 -7,8')
            .attr('fill', 'black');
        })
        .merge(selectedCell);

      selectedCell
        .select('.cellIcon')
        .attr('transform', d => {
          var pos = layout.getBitCenter(d.cellId);
          return `translate(${layout.left + pos[0]},${layout.top + pos[1]})`;
        });

      var synapsesForCell = hoverContainer.selectAll('.synapsesForCell')
          .data(selectedCells);

      synapsesForCell.exit()
        .remove();

      synapsesForCell = synapsesForCell.enter()
        .append('g')
        .attr('class', 'synapsesForCell')
        .merge(synapsesForCell);

      var presynapticCell = synapsesForCell.selectAll('.presynapticCell')
          .data(function(d, i) {
            var synapses = [];
            d.distalSegments.forEach(function(synapsesByState) {
              ['active'].forEach(function(state) {
                synapsesByState[state].forEach(function(synapse) {
                  synapses.push({
                    state: state,
                    presynapticLayer: synapse[0],
                    presynapticBitId: synapse[1],
                    postsynapticCellId: d.cellId
                  });
                });
              });
            });
            return synapses;
          });

      presynapticCell.exit()
        .remove();

      presynapticCell = presynapticCell.enter()
        .append('g')
        .attr('class', 'presynapticCell')
        .call(enter => {
          enter.append('polygon')
            .attr('points', '0,-6 4,3 -4,3')
            .attr('fill', 'crimson')
            .attr('opacity', 0.5);
        })
        .merge(presynapticCell);

      presynapticCell
        .attr('transform', d => {
          var lay = layouts[d.presynapticLayer];
          var position = lay.getBitCenter(d.presynapticBitId);

          return `translate(${lay.left + position[0]},${lay.top + position[1]})`;
        });


    });
  };

  chart.layout = function(_) {
    if (!arguments.length) return layout;
    layout = _;
    return chart;
  };

  chart.layouts = function(_) {
    if (!arguments.length) return layouts;
    layouts = _;
    return chart;
  };

  return chart;
}


function objectPlot() {
  var rowHeight,
      columnWidth,
      color;

  var chart = function chart (selection) {

    var point = selection.selectAll('.point')
        .data(d => d);

    point.exit()
      .remove();

    point = point.enter()
      .append('g')
      .attr('class', 'point')
      .call(enter => {
        enter.append('rect')
          .attr('class', 'featureColor')
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', columnWidth)
          .attr('height', rowHeight)
          .attr('stroke', 'none');

        enter.append('text')
          .attr('class', 'featureText')
          .attr('text-anchor', 'middle')
          .attr('dy', rowHeight * 0.25)
          .attr('x', columnWidth / 2)
          .attr('y', rowHeight / 2)
          .attr('fill', 'white')
          .style('font', `bold ${rowHeight * 0.8}px monospace`);
      })
      .merge(point)
      .attr('transform', (d, i) =>
            `translate(${columnWidth*d[0][1]},${columnWidth*d[0][0]})`);

    point.select('.featureColor')
      .attr('fill', d => {
        if (d) {
          return color(d[1]);
        } else {
          return 'none';
        }
      });

    point.select('.featureText')
      .text(d => d[1]);
  };

  chart.rowHeight = function(_) {
    if (!arguments.length) return rowHeight;
    rowHeight = _;
    return chart;
  };

  chart.columnWidth = function(_) {
    if (!arguments.length) return columnWidth;
    columnWidth = _;
    return chart;
  };

  chart.color = function(_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };

  return chart;
}


function layerOfCellsPlot() {
  var onCellHover;

  var chart = function chart (selection) {
    selection.each(function(d) {
      var hoveredCell = null,
          timestep = d.timestep,
          layout = d.layout,
          layerName = d.layerName;

      var chartContainer = d3.select(this).selectAll('.chartContainer')
          .data([timestep]);

      chartContainer = chartContainer.enter()
        .append('g')
        .attr('class', 'chartContainer')
        .call(function(enter) {
          enter.append('g')
            .attr('class', 'chartBack');
          enter.append('g')
            .attr('class', 'chartMiddle');
          enter.append('g')
            .attr('class', 'chartFront');
          enter.append('g')
            .attr('class', 'chartFront2');
          enter.append('g')
            .attr('class', 'selectedCellContainer');
          enter.append('rect')
            .attr('class', 'mouseCapture')
            .attr('width', layout.width)
            .attr('height', layout.height)
            .attr('fill', 'transparent')
            .attr('stroke', 'none');
        }).merge(chartContainer);

      var back = chartContainer.select('.chartBack'),
          middle = chartContainer.select('.chartMiddle'),
          front = chartContainer.select('.chartFront'),
          front2 = chartContainer.select('.chartFront2');

      chartContainer.select('.mouseCapture')
        .on('mouseout', function() {
          if (hoveredCell != null) {
            hoveredCell = null;
            onCellHover(layerName, null);
          }
        })
        .on('mousemove', function() {
          var box = chartContainer.select('.mouseCapture').node().getBoundingClientRect(),
          x = d3.event.clientX - box.left,
          y = d3.event.clientY - box.top,
          nearestCell = null,
          nearestD2 = null;

          if (timestep) {
            timestep.activeCells.forEach(function(c) {
              var pos = layout.getBitCenter(c.cellId),
              d2 = Math.pow(pos[1] - y, 2) + Math.pow(pos[0] - x, 2);

              if (nearestD2 == null || d2 < nearestD2) {
                nearestD2 = d2;
                nearestCell = c;
              }
            });
          }

          if (hoveredCell != nearestCell) {
            hoveredCell = nearestCell;
            onCellHover(layerName, hoveredCell);
          }
        });

      if (timestep) {

        var activeCell = front.selectAll('.activeCell')
          .data(timestep['activeCells']);

        activeCell.exit()
          .remove();

        activeCell = activeCell.enter()
          .append('g')
          .attr('class', 'activeCell')
          .call(enter => {
            enter.append('polygon')
              .attr('points', '0,-4 2,2 -2,2')
              .attr('stroke', 'none')
              .attr('fill', 'black');
          })
          .merge(activeCell);

        activeCell.attr('transform', d =>
                        `translate(${layout.getBitCenter(d.cellId).join(',')})`);
      }
    });
  };

  chart.onCellHover = function(_) {
    if (!arguments.length) return onCellHover;
    onCellHover = _;
    return chart;
  };

  return chart;
}

function printRecordingFromUrl(node, csvUrl) {
  d3.text(csvUrl,
          function (error, contents) {
            return printRecording(node, contents);
          });
}

function printRecording(node, csv) {

  // Data from the CSV
  var worldDiameter,
      objects,
      timesteps = [],
      featureColor = d3.scaleOrdinal()
      // .domain([null, 'A', 'B', 'C'])
      // .range(['none', 'red', 'blue', 'gray'])
  ;

  // PARSE THE CSV
  (function() {
    var rows = d3.csvParseRows(csv);

    // First row: diameter
    worldDiameter = parseInt(rows[0]);

    // Second row: Features and colors
    //   {'A': 'red',
    //    'B': 'blue',
    //    'C': 'gray'}
    var featureColorMapping = JSON.parse(rows[1]);

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
    //     [[0,0], 'A'],
    //     [[0,1], 'B'],
    //     [[1,0], 'A'],
    //   ],
    //   'Object 2': []
    // };
    objects = JSON.parse(rows[2]);

    var currentTimestep = null;
    var didReset = false;
    var objectPlacements = null;

    var i = 3;
    while (i < rows.length) {
      switch (rows[i][0]) {
      case "reset":
        didReset = true;
        i++;
        break;
      case "t":
        if (currentTimestep !== null) {
          currentTimestep.objectPlacements = objectPlacements;

          timesteps.push(currentTimestep);
        }
        currentTimestep = {
          layers: {}
        };

        if (didReset) {
          currentTimestep.reset = true;
          didReset = false;
        }
        i++;
        break;
      case "input":
        let inputName = rows[i][1];
        currentTimestep.layers[inputName] = {
          activeBits: JSON.parse(rows[i+1]),
          decodings: JSON.parse(rows[i+2])
        };

        i += 3;
        break;
      case "layer":
        let layerName = rows[i][1];

        currentTimestep.layers[layerName] = {
          activeCells: JSON.parse(rows[i+1]).map(cellAndSegments => {
            return {
              cellId: cellAndSegments[0],
              distalSegments: cellAndSegments[1].map(segment => {
                var synapses = [];
                segment.forEach(presynapticLayerAndIndices => {
                  var presynapticLayer = presynapticLayerAndIndices[0];
                  presynapticLayerAndIndices[1].forEach(presynapticCell => {
                    synapses.push([presynapticLayer, presynapticCell]);
                  });
                });

                return {active: synapses};
              })
            };
          }),
          decodings: JSON.parse(rows[i+2])
        };

        i += 3;
        break;
      case "objectPlacements":
        let objectPlacementsDict = JSON.parse(rows[i+1]);

        objectPlacements = [];
        for (let k in objectPlacementsDict) {
          objectPlacements.push({
            name: k,
            offset: objectPlacementsDict[k]
          });
        }

        i += 2;
        break;
      case "egocentricLocation":
        currentTimestep.egocentricLocation = JSON.parse(rows[i+1]);
        i += 2;
        break;
      default:
        i++;
        break;
      }
    }

    if (currentTimestep !== null) {
      currentTimestep.objectPlacements = objectPlacements;

      timesteps.push(currentTimestep);
    }
  })();

  //
  // CONSTANTS
  //
  var drawSynapses = false,
      inputs = ['newLocation', 'deltaLocation', 'feature'],
      layers = ['location', 'input', 'object'],
      layouts = {
        world: new Grid2dLayout(
          worldDiameter, worldDiameter,
          630, 120,
          288, 288,
          {left: 0, right: 0, top: 0, bottom: 0}),
        object: new Grid2dLayout(
          256, 16,
          375, 12,
          150, 60,
          {top: 10, right: 10, bottom: 110, left: 10}),
        input: new Grid2dLayout(
          150, 32,
          375, 212,
          150, 60,
          {top: 10, right: 10, bottom: 110, left: 10}),
        location: new Grid2dLayout(
          40, 25,
          130, 212,
          150, 60,
          {top: 10, right: 10, bottom: 110, left: 10}),
        deltaLocation: new Grid2dLayout(
          40, 25,
          12, 212,
          40, 25,
          {top: 10, right: 10, bottom: 60, left: 10}),
        newLocation: new Grid2dLayout(
          40, 25,
          165, 412,
          80, 25,
          {top: 10, right: 45, bottom: 90, left: 45}),
        feature: new Grid2dLayout(
          150, 1,
          375, 412,
          150, 1,
          {top: 10, right: 10, bottom: 70, left: 10})
      };

  //
  // SHARED STATE
  //
  var iTimestep = 0,
      onSelectedTimestepChanged = [], // callbacks
      width = 985,
      height = 630,
      brainLeft = 0,
      brainTop = 5,
  html = d3.select(node)
    .append('div')
    .attr('tabindex', 0)
    .style('position', 'relative')
    .style('height', height + 'px')
    .style('width', width + 'px')
    .on('keydown', function() {
      switch (d3.event.keyCode) {
      case 37: // Left
        iTimestep--;
        if (iTimestep < 0) {iTimestep = timesteps.length - 1;}
        onSelectedTimestepChanged.forEach(function(f) { f(); });
        d3.event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%timesteps.length;
        onSelectedTimestepChanged.forEach(function(f) { f(); });
        d3.event.preventDefault();
        break;
      }
    }),
  svg = html.append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('max-width', 'none') // jupyter notebook tries to set this
      .style('max-height', 'none');

  svg.append('defs')
    .append('marker')
      .attr('id', 'arrow')
      .attr('markerWidth', 2)
      .attr('markerHeight', 4)
      .attr('refX', 0.1)
      .attr('refY', 2)
      .attr('orient', 'auto')
      .attr('markerUnits', 'strokeWidth')
    .append('path')
    .attr('d', 'M0,0 V4 L2,2 Z');

  var slideshow = svg
      .append('g')
      .on('click', () => {
        iTimestep = (iTimestep+1)%timesteps.length;
        onSelectedTimestepChanged.forEach(function(f) { f(); });
      });

  slideshow.append('rect')
    .attr('fill', 'transparent')
    .attr('stroke', 'none')
    .attr('width', width)
    .attr('height', 600);

  var brain = slideshow
      .append('g')
      .attr('transform', `translate(${brainLeft},${brainTop})`);


  layouts.decodedLocation = new Grid2dLayout(
    worldDiameter, worldDiameter,
    layouts.location.left + 35,
    layouts.location.top + layouts.location.height + 20,
    80, 80);

  layouts.decodedNewLocation = new Grid2dLayout(
    worldDiameter, worldDiameter,
    layouts.newLocation.left + 10,
    layouts.newLocation.top + layouts.newLocation.height + 20,
    60, 60);

  layouts.decodedInput = new Grid2dLayout(
    worldDiameter, worldDiameter,
    layouts.input.left + 35,
    layouts.input.top + layouts.input.height + 20,
    80, 80);


  //
  // timesteps = [
  //   {
  //     layers: {
  //       location: {
  //         activeCells: [{cellId: 42,
  //                        showSynapses: false,
  //                        distalSegments: [{
  //                          active: [["deltaLocation", 13],
  //                                   ["deltaLocation", 14]]
  //                        }]}],
  //         activeColumns: [10]
  //       }
  //     },
  //     senses: {
  //       deltaLocation: {
  //         activeBits: [42, 43]
  //       }
  //     }
  //   }
  // ]



  //
  // LAYERS
  //
  (function() {
    let chart = layerOfCellsPlot()
      .onCellHover(drawHoveredCell);

    function draw() {
      var layer = brain.selectAll('.layer')
          .data(layers.map(layerName => {
            return {
              layerName: layerName,
              layout: layouts[layerName],
              timestep: timesteps[iTimestep].layers[layerName]
            };
          }));

      layer = layer.enter()
        .append('g')
        .attr('class', 'layer')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'node')
            .attr('fill', 'none')
            .attr('stroke', 'lightgray')
            .attr('stroke-width', 3)
            .attr('x', d => d.layout.left - d.layout.padding.left)
            .attr('y', d => d.layout.top - d.layout.padding.top)
            .attr('width', d => d.layout.width + d.layout.padding.left + d.layout.padding.right)
            .attr('height', d => d.layout.height + d.layout.padding.top + d.layout.padding.bottom);

          enter
            .append('g')
              .attr('class', 'layerOfCells')
              .attr('transform', d =>
                  `translate(${d.layout.left},${d.layout.top})`)
            .append('rect')
              .attr('x', -2)
              .attr('y', -2)
              .attr('width', d => d.layout.width + 4)
              .attr('height', d => d.layout.height + 4)
              .attr('fill', 'none')
              .attr('stroke', 'black');
        })
        .merge(layer);

      layer.select('.layerOfCells')
        .call(chart);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();


  //
  // INPUTS
  //
  (function() {

    function draw() {
      var input = brain.selectAll('.input')
          .data(inputs.map(inputName => {
            return {
              layerName: inputName,
              layout: layouts[inputName],
              timestep: timesteps[iTimestep].layers[inputName]
            };
          }));

      input = input.enter()
        .append('g')
        .attr('class', 'input')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'node')
            .attr('fill', 'none')
            .attr('stroke', 'gray')
            .attr('stroke-dasharray', "5, 5")
            .attr('x', d => d.layout.left - d.layout.padding.left)
            .attr('y', d => d.layout.top - d.layout.padding.top)
            .attr('width', d => d.layout.width + d.layout.padding.left + d.layout.padding.right)
            .attr('height', d => d.layout.height + d.layout.padding.top + d.layout.padding.bottom);

          enter
            .append('g')
              .attr('class', 'inputAxons')
              .attr('transform', d =>
                    `translate(${d.layout.left},${d.layout.top})`)
            .append('rect')
              .attr('x', -2)
              .attr('y', -2)
              .attr('width', d => d.layout.width + 4)
              .attr('height', d => d.layout.height + 4)
              .attr('fill', 'none')
              .attr('stroke', 'black');
        })
        .merge(input);

      var activeAxon = input.select('.inputAxons')
          .selectAll('.activeAxon')
          .data(d => d.timestep.activeBits.map(bit => {
            return {
              layout: d.layout, bit: bit
            };
          }));

      activeAxon.exit()
        .remove();

      activeAxon = activeAxon.enter()
        .append('g')
        .attr('class', 'activeAxon')
        .call(enter => {
          enter.append('circle')
            .attr('r', 1.5)
            .attr('stroke', 'none')
            .attr('fill', 'black');
        })
        .merge(activeAxon)
        .attr('transform', d =>
              `translate(${d.layout.getBitCenter(d.bit).join(',')})`);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();


  //
  // DECODED FEATURE
  //
  (function() {

    function draw() {
      var decodedFeatures = brain
          .selectAll('.decodedFeatures')
          .data(['feature'].map(name => {
            return {
              name: name,
              sourceLayout: layouts[name],
              timestep: timesteps[iTimestep].layers[name]
            };
          }));

      decodedFeatures.exit()
        .remove();

      decodedFeatures = decodedFeatures.enter()
        .append('g')
        .attr('class', 'decodedFeatures')
        .merge(decodedFeatures)
        .attr('transform', d =>
              `translate(${d.sourceLayout.left},${d.sourceLayout.top + d.sourceLayout.height + 20})`);

      var decodedFeature = decodedFeatures.selectAll('.decodedFeature')
          .data(d => [d.timestep.decodings[0]]);

      decodedFeature = decodedFeature.enter()
        .append('g')
        .attr('class', 'decodedFeature')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'featureColor')
            .attr('width', d => 40)
            .attr('height', d => 40)
            .attr('fill', 'none')
            .attr('stroke', 'none');

          enter.append('text')
            .attr('class', 'featureText')
            .attr('text-anchor', 'middle')
            .attr('dy', 8)
            .attr('x', 20)
            .attr('y', 20)
            .attr('fill', 'white')
            .style('font', 'bold 26px monospace');
        })
        .merge(decodedFeature)
        .attr('transform', 'translate(55,0)');

      decodedFeature.select('.featureColor')
        .attr('fill', d => featureColor(d));

      decodedFeature.select('.featureText')
          .text(d => d);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();

  //
  // DECODED OBJECT
  //
  (function() {

    function draw() {
      var decodedObjects = brain
          .selectAll('.decodedObjects')
          .data(['object'].map(name => {
            return {
              name: name,
              sourceLayout: layouts[name],
              timestep: timesteps[iTimestep].layers[name]
            };
          }));

      decodedObjects.exit()
        .remove();

      decodedObjects = decodedObjects.enter()
        .append('g')
        .attr('class', 'decodedObjects')
        .merge(decodedObjects)
        .attr('transform', d =>
              `translate(${d.sourceLayout.left + 10},${d.sourceLayout.top + d.sourceLayout.height + 20})`);

      var decodedObjectRow = decodedObjects.selectAll('.decodedObjectRow')
          .data(d => {
            let decodings = d.timestep.decodings;

            var rows = [];
            var i = 0;
            for (; i + 3 < decodings.length; i += 3) {
              rows.push([decodings[0], decodings[1], decodings[2]]);
            }

            if (i < decodings.length) {
              let lastRow = [];
              for (; i < decodings.length; i++) {
                lastRow.push(decodings[i]);
              }
              rows.push(lastRow);
            }

            return rows;
          });

      decodedObjectRow.exit()
        .remove();

      decodedObjectRow = decodedObjectRow.enter()
        .append('g')
        .attr('class', 'decodedObjectRow')
        .merge(decodedObjectRow)
        .attr('transform', (d, i) => {
          var y = (i == 0) ? 0 : 30*i + 10;
          return `translate(0,${y})`;
        });

      var decodedObject = decodedObjectRow.selectAll('.decodedObject')
          .data(d => d);

      decodedObject.exit()
        .remove();

      decodedObject = decodedObject.enter()
        .append('g')
        .attr('class', 'decodedObject')
        .merge(decodedObject)
        .attr('transform', (d, i) => `translate(${i*50},0)`);

      decodedObject
        .datum(d => objects[d])
        .call(objectPlot()
              .rowHeight(10)
              .columnWidth(10)
              .color(featureColor));
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();

  //
  // DECODED DELTA LOCATION
  //
  (function() {

    function draw() {
      var decodedDeltaLocations = brain
          .selectAll('.decodedDeltaLocations')
          .data(['deltaLocation'].map(name => {
            return {
              name: name,
              sourceLayout: layouts[name],
              timestep: timesteps[iTimestep].layers[name]
            };
          }));

      decodedDeltaLocations.exit()
        .remove();

      decodedDeltaLocations = decodedDeltaLocations.enter()
        .append('g')
        .attr('class', 'decodedDeltaLocations')
        .merge(decodedDeltaLocations)
        .attr('transform', d =>
              `translate(${d.sourceLayout.left},${d.sourceLayout.top + d.sourceLayout.height + 20})`);

      var decodedDeltaLocation = decodedDeltaLocations.selectAll('.decodedDeltaLocation')
          .data(d => d.timestep.decodings);

      decodedDeltaLocation.exit()
        .remove();

      decodedDeltaLocation = decodedDeltaLocation.enter()
        .append('g')
        .attr('class', 'decodedDeltaLocation')
        .call(enter => {
          enter.append('g')
            .attr('class', 'arrowTransform')
            .append('line')
            .attr('class', 'deltaLocationArrow')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', 20)
            .attr('y2', 0)
            .attr('stroke', '#000')
            .attr('stroke-width', 5)
            .attr('marker-end', 'url(#arrow)');
        })
        .merge(decodedDeltaLocation)
        .attr('transform', 'translate(12,12)');

      decodedDeltaLocation
        .select('.arrowTransform')
        .attr('transform', d => {
          var radians = Math.atan(-d[0] / d[1]);
          var degrees = radians * 180 / Math.PI;
          if (d[1] < 0) {
            degrees += 180;
          }
          return `rotate(${-degrees} 10 0)`;
        });
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();


  //
  // DECODED LOCATIONS
  //
  (function() {

    function draw() {
      var decodedLocations = brain
        .selectAll('.decodedLocations')
          .data([['location', 'decodedLocation'],
                 ['newLocation', 'decodedNewLocation'],
                 ['input', 'decodedInput']].map(names => {
                   let layerName = names[0];
                   let name = names[1];
                   return {
                     name: name,
                     timestep: timesteps[iTimestep].layers[layerName],
                     layout: layouts[name]
                   };
                 }));

      decodedLocations = decodedLocations.enter()
        .append('g')
        .attr('class', 'decodedLocations')
            .call(enter => {
              enter.append('rect')
                .attr('width', d => d.layout.width)
                .attr('height', d => d.layout.height)
                .attr('fill', 'none')
                .attr('stroke', 'lightgray')
                .attr('stroke-width', 1);

              enter.selectAll('.verticalLine')
                .data(d => d3.range(d.layout.nColumns).map(i => {
                  return {
                    layout: d.layout,
                    i: i
                  };
                }))
                .call(function(verticalLine) {
                  verticalLine.enter()
                    .append('line')
                      .attr('class', 'verticalLine')
                      .attr('x1', function(d, i) { return d.i*d.layout.columnWidth; })
                      .attr('y1', 0)
                      .attr('x2', function(d, i) { return d.i*d.layout.columnWidth; })
                      .attr('y2', d => d.layout.height)
                      .attr('stroke', 'lightgray')
                      .attr('stroke-width', 1);

                  verticalLine.exit()
                    .remove();
                });

              enter.selectAll('.horizontalLine')
                .data(d => d3.range(d.layout.nRows).map(i => {
                  return {
                    layout: d.layout,
                    i: i
                  };
                }))
                .call(function(horizontalLine) {
                  horizontalLine.enter()
                    .append('line')
                    .attr('class', 'horizontalLine')
                    .attr('x1', 0)
                    .attr('y1', function(d, i) { return d.i*d.layout.rowHeight; })
                    .attr('x2', d => d.layout.width)
                    .attr('y2', function(d, i) { return d.i*d.layout.rowHeight; })
                    .attr('stroke', 'lightgray')
                    .attr('stroke-width', 1);

              horizontalLine.exit()
                .remove();
            });
        })
        .merge(decodedLocations)
        .attr('transform', d =>
              `translate(${d.layout.left},${d.layout.top})`);

      var currentLocation = decodedLocations.selectAll('.currentLocation')
          .data(d => d.timestep.decodings.map(decoding => {
            if (d.name == "decodedInput") {
              return {
                location: decoding[1],
                feature: decoding[0],
                layout: d.layout
              };
            } else {
              return {
                location: decoding,
                layout: d.layout
              };
            }
          }));

      currentLocation.exit()
        .remove();

      currentLocation = currentLocation.enter()
        .append('g')
        .attr('class', 'currentLocation')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'featureColor')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', d => d.layout.columnWidth)
            .attr('height',d => d.layout.rowHeight)
            .attr('stroke', 'none');

          enter.append('text')
            .attr('class', 'featureText')
            .attr('text-anchor', 'middle')
            .attr('dy', d => d.layout.rowHeight * 0.35)
            .attr('x', d => d.layout.columnWidth/2)
            .attr('y', d => d.layout.rowHeight/2)
            .attr('fill', 'white')
            .style('font', 'bold 8px monospace');
        })
        .merge(currentLocation);

      currentLocation.attr('transform', d => {
        var position = [d.layout.columnWidth*d.location[1],
                        d.layout.rowHeight*d.location[0]];
        return `translate(${position[0]}, ${position[1]})`;
      });

      currentLocation.select('.featureColor')
        .attr('fill', d => {
          if (d.feature) {
            return featureColor(d.feature);
          } else {
            return 'black';
          }
        });

      currentLocation.select('.featureText')
        .text(d => d.feature);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();

  var hoveredCellContainer = brain.append('g')
      .attr('class', 'hoveredCellContainer')
      .style('pointer-events', 'none');

  function drawHoveredCell(layerName, hoveredCell) {
    hoveredCellContainer
      .datum(hoveredCell ? [hoveredCell] : [])
      .call(selectedCellPlot().layout(layouts[layerName]).layouts(layouts));
  }

  //
  // LABELS
  //

  [{layout: layouts.object,
    // decodedLayout: layouts.decodedLocation,
    text: 'object layer'},
   {layout: layouts.input,
    text: 'feature-location pair layer'},
   {layout: layouts.location,
    text: 'location layer'},
   {layout: layouts.deltaLocation,
    text: 'motor input'},
   {layout: layouts.world,
    text: 'egocentric space'},
   {layout: layouts.newLocation,
    text: 'egocentric location input'},
   {layout: layouts.feature,
    text: 'feature input'}].forEach(function(d) {
      html.append('div')
        .style('position', 'absolute')
        .style('width', '50px')
        .style('left', `${brainLeft + d.layout.left + d.layout.width + d.layout.padding.right + 6}px`)
        .style('top', `${d.layout.top - d.layout.padding.top}px`)
        .style('text-align', 'left')
        .style('font', '10px Verdana')
        .style('pointer-events', 'none')
        .text(d.text);
    });

  (function() {

    var xOffset = 0;

    timesteps.forEach((timestep, i) => {
      if (i > 0 && timestep.reset) {
        xOffset += 4;
      }

      timestep.xOffset = xOffset;

      xOffset += 12;
    });

    var time = svg.append('g')
        .attr('class', 'time')
        .attr('transform', `translate(${width/2 - xOffset/2}, 600)`);

    function drawTime() {

      var timestepMarker = time.selectAll('.timestepMarker')
          .data(timesteps);

      timestepMarker.exit()
        .remove();

      timestepMarker = timestepMarker.enter()
        .append('g')
        .attr('transform', (d, i) => `translate(${d.xOffset},0)`)
        .attr('class', 'timestepMarker')
        .call(enter => {
          enter.append('circle')
            .attr('class', 'regular')
            .attr('r', 5)
            .attr('cx', 5)
            .attr('cy', 5)
            .attr('stroke', 'none')
            .style('cursor', 'pointer')
            .on('click', function(d, i) {
              iTimestep = i;
              onSelectedTimestepChanged.forEach(function(f) { f(); });
            });
        })
        .merge(timestepMarker)
        .attr('fill', (d, i) => i == iTimestep ? 'black' : 'lightgray');

      let resets = [];
      timesteps.forEach((timestep, i) => {
        if (i > 0 && timestep.reset) {
          resets.push(i);
        }
      });

      var resetMarker = time.selectAll('.resetMarker')
          .data(resets);

      resetMarker = resetMarker.enter()
        .append('rect')
        .attr('class', 'resetMarker')
        .attr('height', 14)
        .attr('width', 2)
        .attr('y', -2)
        .attr('fill', 'gray')
        .merge(resetMarker)
        .attr('x', d => timesteps[d].xOffset - 4);
    }

    onSelectedTimestepChanged.push(drawTime);
      drawTime();

  })();

  //
  // THE WORLD
  //
  (function() {
    let worldLayout = layouts.world;

    svg.append('line')
      .attr('stroke', 'gray')
      .attr('stroke-width', 1)
      .attr('x1', worldLayout.left - 25)
      .attr('y1', 10)
      .attr('x2', worldLayout.left - 25)
      .attr('y2', 550);

    let world = slideshow.append('g')
        .attr('transform', `translate(${worldLayout.left},${worldLayout.top})`);

    world.append('rect')
      .attr('width', d => worldLayout.width)
      .attr('height', d => worldLayout.height)
      .attr('fill', 'none')
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);

    world.selectAll('.verticalLine')
      .data(d3.range(worldDiameter))
      .enter()
      .append('line')
      .attr('class', 'verticalLine')
      .attr('x1', i => i * worldLayout.columnWidth)
      .attr('y1', 0)
      .attr('x2', i => i * worldLayout.columnWidth)
      .attr('y2', worldLayout.height)
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);

    world.selectAll('.horizontalLine')
      .data(d3.range(worldDiameter))
      .enter()
      .append('line')
      .attr('class', 'horizontalLine')
      .attr('x1', 0)
      .attr('y1', i => i * worldLayout.rowHeight)
      .attr('x2', d => worldLayout.width)
      .attr('y2', i => i * worldLayout.rowHeight)
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);


    function draw() {
      var objectPlacements = timesteps[iTimestep].objectPlacements || [];
      var egocentricLocation = timesteps[iTimestep].egocentricLocation;

      var placedObject = world
          .selectAll('.placedObject')
          .data(objectPlacements);

      placedObject = placedObject
        .enter()
        .append('g')
        .attr('class', 'placedObject')
        .merge(placedObject);

      placedObject
        .attr('transform', d => {
          var position = [worldLayout.columnWidth*d.offset[1],
                          worldLayout.rowHeight*d.offset[0]];
          return `translate(${position[0]}, ${position[1]})`;
        })
        .datum(d => objects[d.name])
        .call(objectPlot()
              .rowHeight(worldLayout.rowHeight)
              .columnWidth(worldLayout.columnWidth)
              .color(featureColor));

      var currentLocation = world
          .selectAll('.currentLocation')
          .data([egocentricLocation]);

      currentLocation = currentLocation.enter()
        .append('g')
        .attr('class', 'currentLocation')
        .call(enter => {
          enter.append('rect')
            .attr('stroke', 'gold')
            .attr('stroke-width', 5)
            .attr('fill', 'none')
            .attr('width', worldLayout.columnWidth)
            .attr('height', worldLayout.rowHeight);
        })
        .merge(currentLocation)
        .attr('transform', d => `translate(${worldLayout.columnWidth*d[1]}, ${worldLayout.rowHeight*d[0]})`);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();
}

export {
  printRecording,
  printRecordingFromUrl
};
