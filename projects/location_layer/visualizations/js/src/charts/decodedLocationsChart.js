import * as d3 from "d3";
import {featureChart} from './featureChart.js';

function partition(arr, n) {
  let result = [];

  for (let i = 0; i < arr.length; i += n) {
    result.push(arr.slice(i, i+n));
  }

  return result;
}

/**
 * Example data:
 *    {decodings: [{objectName: 'Object 1', top: 42.0, left: 17.2, amountContained: 0.95}],
 *     objects: {'Object 1': [{name: 'A', left: 11.2, top:12.0, width: 12.2, height: 7.2}],
 *               'Object 2': []}}
 */
function decodedLocationsChart() {
  let width,
      height,
      color,
      perRow = 3,
      minimumMatch = 0.25;

  let chart = function(selection) {

    selection.each(function(decodedLocationsData) {
      let decodingsByObject = {};
      decodedLocationsData.decodings.forEach(d => {
        if (!decodingsByObject.hasOwnProperty(d.objectName)) {
          decodingsByObject[d.objectName] = [];
        }

        decodingsByObject[d.objectName].push(d);
      });

      let decodings = [],
          maxWidth = 0,
          maxHeight = 0,
          pxPerRow = 4 + width / perRow;
      for (let objectName in decodingsByObject) {
        if (decodingsByObject[objectName].some(
          d => d.amountContained >= minimumMatch)) {
          decodings.push([objectName, decodingsByObject[objectName]]);

          decodedLocationsData.objects[objectName].forEach(d => {
            maxWidth = Math.max(maxWidth, d.left + d.width);
            maxHeight = Math.max(maxHeight, d.top + d.height);
          });
        }
      }

      // Sort by object name.
      decodings.sort((a,b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);

      let rows = partition(decodings, perRow);

      let decodedObjectRow = d3.select(this).selectAll('.decodedObjectRow')
          .data(rows);

      decodedObjectRow.exit().remove();

      decodedObjectRow = decodedObjectRow.enter().append('g')
        .attr('class', 'decodedObjectRow')
        .attr('transform', (d,i) => `translate(0,${i == 0 ? 0 : i*height/2})`)
        .merge(decodedObjectRow);

      let decodedObject = decodedObjectRow.selectAll('.decodedObject')
          .data(d => d);

      decodedObject.exit().remove();

      decodedObject = decodedObject.enter().append('g')
        .attr('class', 'decodedObject')
        .attr('transform', (d, i) => `translate(${i*pxPerRow},0)`)
        .call(enter => {
          enter.append('g')
            .attr('class', 'features');
          enter.append('g')
              .attr('class', 'points')
            .append('rect')
              .attr('width', width/perRow)
              .attr('height', height/2)
              .attr('fill', 'white')
              .attr('fill-opacity', 0.7);
        })
        .merge(decodedObject);

      decodedObject.each(function([objectName, decodedLocations]) {

        let cmMax = Math.max(maxWidth, maxHeight);
        let pxMax = Math.min(width/perRow, height/2 - 4);
        let x = d3.scaleLinear()
            .domain([0, cmMax])
            .range([0, pxMax]);
        let y = d3.scaleLinear()
            .domain([0, cmMax])
            .range([0, pxMax]);

        let feature = d3.select(this).select('.features').selectAll('.feature')
            .data(decodedLocationsData.objects[objectName]);

        feature.exit().remove();

        feature = feature.enter()
          .append('g')
          .attr('class', 'feature')
          .merge(feature)
          .attr('transform', d => `translate(${x(d.left)},${y(d.top)})`)
          .each(function(featureData) {
            d3.select(this)
              .call(featureChart()
                    .width(x(featureData.width))
                    .height(y(featureData.height))
                    .color(color)
                    .includeText(false));
          });

        let point = d3.select(this).select('.points').selectAll('.point')
            .data(decodedLocations);

        point.exit().remove();

        point = point.enter().append('g')
            .attr('class', 'point')
          .call(enter => {
            enter.append('circle')
              .attr('r', 4)
              .attr('fill', 'white')
              .attr('stroke', 'none');

            enter.append('path')
              .attr('fill', 'black')
              .attr('stroke', 'none');
          }).merge(point)
            .attr('transform', d => `translate(${x(d.left)},${y(d.top)})`);

        point.select('path')
          .attr('d', d3.arc()
                .innerRadius(0)
                .outerRadius(4)
                .startAngle(0)
                .endAngle(d => d.amountContained / 1.0 * 2 * Math.PI));
      });
    });
  };

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

  chart.perRow = function(_) {
    if (!arguments.length) return perRow;
    perRow = _;
    return chart;
  };

  chart.color = function(_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };

  chart.minimumMatch = function(_) {
    if (!arguments.length) return minimumMatch;
    minimumMatch = _;
    return chart;
  };

  return chart;
}

export {decodedLocationsChart};
