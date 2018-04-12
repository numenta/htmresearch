import * as d3 from 'd3';
import {featureChart} from './featureChart.js';

const CURSOR_URL = 'data:image/png;base64,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';

/**
 * Data:
 * {location: [12.0, 16.0],
 *  features: [{name: 'A', location: {}, width: 12.2, height: 7.2}],
 *  selectedLocationCell: 42,
 *  selectedLocationModule: {
 *    cellDimensions: [5, 5],
 *    moduleMapDimensions: [20.0, 20.0],
 *    orientation: 0.834
 *    activeCells: [42],
 *    activePoints: [[11.0, 12.0]]
 *  }}
 */
function sensorOnObjectChart() {
  var width,
      height,
      color,
      xScale,
      yScale;

  let drawFiringFields = function(selection) {
    selection.each(function(worldData) {
      let worldBackground = d3.select(this).select('.worldBackground'),
          firingFields = worldBackground.select('.firingFields');

      let x = location => xScale(location.left),
          y = location => yScale(location.top);

      let pattern = worldBackground.select('.myDefs')
          .selectAll('pattern')
          .data(worldData.selectedLocationModule
                ? [worldData.selectedLocationModule.activePoints]
                : []);

      pattern.exit().remove();

      if (worldData.selectedLocationModule) {
        let config = worldData.selectedLocationModule,
            distancePerCell = [config.moduleMapDimensions[0] / config.cellDimensions[0],
                               config.moduleMapDimensions[1] / config.cellDimensions[1]],
            pixelsPerCell = [height * (distancePerCell[0] /
                                       (yScale.domain()[1] - yScale.domain()[0])),
                             width * (distancePerCell[1] /
                                      (xScale.domain()[1] - xScale.domain()[0]))];

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
              .attr('d', points => {

                let squares = [];

                for (let i = -1; i < 2; i++) {
                  for (let j = -1; j < 2; j++) {
                    points.forEach(point => {

                      let cellFieldOrigin = [
                        Math.floor(worldData.selectedLocationCell / config.cellDimensions[1]),
                        worldData.selectedLocationCell % config.cellDimensions[1]];

                      let top = i*config.cellDimensions[1] + (cellFieldOrigin[0] - point[0]),
                          left = j*config.cellDimensions[0] + (cellFieldOrigin[1] - point[1]);

                      squares.push(`M ${left} ${top} l 1 0 l 0 1 l -1 0 Z`);

                    });
                  }
                }

                return squares.join(' ');
              });
          })
          .attr('patternTransform', point => {
            let translateLocation = {
              top: worldData.location.top,
              left: worldData.location.left
            };

            return `translate(${x(translateLocation)},${y(translateLocation)}) `
              + `rotate(${180 * config.orientation / Math.PI}, 0, 0) `
              + `scale(${pixelsPerCell[1]},${pixelsPerCell[0]})`;
          });
      }

      let firingField = firingFields.selectAll('.firingField')
          .data(worldData.selectedLocationModule
                ? [worldData.selectedLocationModule.activePoints]
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

      let x = location => xScale(location.left),
          y = location => yScale(location.top);


      let features = worldNode.selectAll('.features')
          .data(d => [d]);

      features = features.enter().append('g')
        .attr('class', 'features')
        .merge(features);

      let feature = features.selectAll('.feature')
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
                  .width(xScale(featureData.width) - xScale(0))
                  .height(yScale(featureData.height) - yScale(0))
                  .color(color));
        });

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

      let currentLocation = d3.select(this).selectAll('.currentLocation')
          .data([null]);

      currentLocation.enter().append('g')
        .attr('class', 'currentLocation')
        .call(enter => {
          enter.append('image')
            .attr('x', -21)
            .attr('y', -10)
            .attr('width', 50)
            .attr('height', 53)
            .attr('xlink:href', CURSOR_URL);
        })
        .merge(currentLocation)
        .attr('transform', `translate(${x(worldData.location)},${y(worldData.location)})`);
    });

    drawFiringFields(selection);
  };

  chart.drawFiringFields = drawFiringFields;

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

  chart.xScale = function(_) {
    if (!arguments.length) return xScale;
    xScale = _;
    return chart;
  };

  chart.yScale = function(_) {
    if (!arguments.length) return yScale;
    yScale = _;
    return chart;
  };

  return chart;
}

export {sensorOnObjectChart};
