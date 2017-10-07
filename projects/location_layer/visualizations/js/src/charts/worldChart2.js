import * as d3 from 'd3';
import {featureChart} from './featureChart.js';
import {plusShape} from './../shapes/plusShape.js';

/**
 * Data:
 * {locations: [[12.0, 16.0], [11.0, 6.0]],
 *  features: [{name: 'A', location: {}, width: 12.2, height: 7.2}],
 *  selectedBodyPart: 'body',
 *  selectedAnchorLocation: {top: 12.0, 16.0},
 *  selectedLocationCell: 42,
 *  selectedLocationModule: {
 *    cellDimensions: [5, 5],
 *    moduleMapDimensions: [20.0, 20.0],
 *    orientation: 0.834
 *    cells: [{cell: 42}, ...],
 *  }}
 */
function worldChart() {
  var width,
      height,
      color,
      t = 0;

  let drawSelectedBodyPart = function(selection) {
    selection.each(function(worldData) {
      d3.select(this)
        .selectAll('.currentLocation')
        .select('path')
        .attr('fill', (d, i) => worldData.selectedBodyPart == i
              ? 'goldenrod'
              : 'white');

      d3.select(this)
        .select('.bodyLocation')
        .select('circle')
        .attr('fill', (worldData.selectedBodyPart == 'body')
              ? 'goldenrod'
              : 'white');
    });
  };

  let drawFiringFields = function(selection) {
    selection.each(function(worldData) {
      let worldBackground = d3.select(this).select('.worldBackground'),
          firingFields = worldBackground.select('.firingFields');

      let xScale = d3.scaleLinear()
            .domain([0, worldData.dims.width])
            .range([0, width]),
          x = location => xScale(location.left),
          yScale =  d3.scaleLinear()
            .domain([0, worldData.dims.height])
            .range([0, height]),
          y = location => yScale(location.top);

      let pattern = worldBackground.select('.myDefs')
          .selectAll('pattern')
          .data(worldData.selectedLocationModule
                ? [worldData.selectedLocationModule.cells.map(c => c.cell)]
                : []);

      pattern.exit().remove();

      if (worldData.selectedLocationModule) {
        let config = worldData.selectedLocationModule,
            distancePerCell = [config.moduleMapDimensions[0] / config.cellDimensions[0],
                               config.moduleMapDimensions[1] / config.cellDimensions[1]],
            pixelsPerCell = [height * (distancePerCell[0] / worldData.dims.height),
                             width * (distancePerCell[1] / worldData.dims.width)];

        let selectedCellFieldOrigin = [
          Math.floor(worldData.selectedLocationCell / config.cellDimensions[1]),
          worldData.selectedLocationCell % config.cellDimensions[1]];

        pattern.enter().append('pattern')
          .attr('id', 'FiringField')
          .attr('patternUnits', 'userSpaceOnUse')
          .call(enter => {
            enter.append('path')
              .attr('fill', 'black')
              .attr('fill-opacity', 0.6)
              .attr('stroke', 'none');
          })
          .merge(pattern)
          .attr('width', config.cellDimensions[1])
          .attr('height', config.cellDimensions[0])
          .call(p => {
            p.select('path')
              .attr('d', cells => {
                let squares = [];

                for (let i = -1; i < 2; i++) {
                  for (let j = -1; j < 2; j++) {
                    cells.forEach(cell => {
                      let cellFieldOrigin = [
                        Math.floor(cell / config.cellDimensions[1]),
                        cell % config.cellDimensions[1]];

                      let top = i*config.cellDimensions[1] + (selectedCellFieldOrigin[0] - cellFieldOrigin[0]),
                          left = j*config.cellDimensions[0] + (selectedCellFieldOrigin[1] - cellFieldOrigin[1]);

                      squares.push(`M ${left} ${top} l 1 0 l 0 1 l -1 0 Z`);

                    });
                  }
                }

                return squares.join(' ');
              });
          })
          .attr('patternTransform', point => {
            return `translate(${x(worldData.selectedAnchorLocation)},${y(worldData.selectedAnchorLocation)}) `
              + `rotate(${180 * config.orientation / Math.PI}, 0, 0) `
              + `scale(${pixelsPerCell[1]},${pixelsPerCell[0]})`;
          });
      }

      let firingField = firingFields.selectAll('.firingField')
          .data(worldData.selectedLocationModule
                ? [worldData.selectedLocationModule.cells.map(c => c.cell)]
                : []);

      firingField.enter().append('rect')
        .attr('class', 'firingField')
        .attr('fill', (d,i) => `url(#FiringField)`)
        .attr('stroke', 'none')
        .attr('width', width)
        .attr('height', height);

      firingField.exit().remove();
    });
  };

  var chart = function(selection) {
    selection.each(function(worldData) {
      let worldNode = d3.select(this);

      let xScale = d3.scaleLinear()
            .domain([0, worldData.dims.width])
            .range([0, width]),
          x = location => xScale(location.left),
          yScale =  d3.scaleLinear()
            .domain([0, worldData.dims.height])
            .range([0, height]),
          y = location => yScale(location.top);


      let features = worldNode.selectAll('.features')
        .data(d => [d]);

      features = features.enter().append('g')
        .attr('class', 'features')
        .merge(features);

      let feature = features.selectAll('feature')
          .data(d => d.features ? d.features : []);

      feature.exit().remove();

      feature.enter()
        .append('g')
        .attr('class', 'feature')
        .merge(feature)
        .attr('transform', d => `translate(${x(d)},${y(d)})`)
        .each(function(featureData) {
          d3.select(this)
            .call(featureChart()
                  .width(xScale(featureData.width))
                  .height(yScale(featureData.height))
                  .color(color));
        });


      let appendage = d3.select(this).selectAll('.appendage')
          .data(worldData.locations);

      appendage.exit().remove();

      appendage.enter().append('line')
          .attr('class', 'appendage')
          .attr('stroke', 'black')
          .attr('stroke-width', 1)
        .merge(appendage)
        .attr('x1', d => x(d))
        .attr('y1', d => y(d))
        .attr('x2', x(worldData.bodyLocation))
        .attr('y2', y(worldData.bodyLocation));

      let currentLocation = d3.select(this).selectAll('.currentLocation')
          .data(worldData.locations);

      currentLocation.exit().remove();

      currentLocation.enter().append('g')
          .attr('class', 'currentLocation')
          .call(enter => {
            enter.append('path')
              .attr('stroke', 'black')
              .attr('stroke-width', 1)
              .attr('d', plusShape().radius(7).innerRadius(2));
          })
        .merge(currentLocation)
          .attr('transform', d => `translate(${x(d)},${y(d)})`);


      let bodyLocation = d3.select(this).selectAll('.bodyLocation')
          .data([worldData.bodyLocation]);

      bodyLocation.exit().remove();

      bodyLocation.enter().append('g')
          .attr('class', 'bodyLocation')
          .call(enter => {
            enter
              .append('circle')
              .attr('fill', 'white')
              .attr('stroke', 'black')
              .attr('stroke-width', 2)
              .attr('r', 5);
          })
        .merge(bodyLocation)
          .attr('transform', d => `translate(${x(d)},${y(d)})`);

      let worldBackground = worldNode.selectAll('.worldBackground')
        .data(d => [d]);

      worldBackground = worldBackground
        .enter().append('g')
          .attr('class', 'worldBackground')
        .call(enter => {
          enter.append('defs')
              .attr('class', 'myDefs');

          enter.append('rect').attr('fill', 'none')
            .attr('width', width)
            .attr('height', height)
            .attr('stroke', 'black')
            .attr('stroke-dasharray', '5,5')
            .attr('stroke-width', 2);

          enter.append('g').attr('class', 'firingFields');
        })
        .merge(worldBackground);
    });

    drawFiringFields(selection);
    drawSelectedBodyPart(selection);
  };

  chart.drawFiringFields = drawFiringFields;
  chart.drawSelectedBodyPart = drawSelectedBodyPart;

  chart.width = function(_) {
    if (!arguments.length) return width;
    width = _;
    return chart;
  };

  chart.height = function(_) {
    if (!arguments.length) return height;
    height = _;
    return chart;
  };

  chart.color = function(_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };


  return chart;
}

export {worldChart};
