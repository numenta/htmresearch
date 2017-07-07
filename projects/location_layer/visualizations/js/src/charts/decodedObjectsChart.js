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
 *    {decodings: ['Object 1', 'Object 2'],
 *     objects: {'Object 1': [{name: 'A', top: 42.0, left: 17.2, width: 12.2, height: 7.2}],
 *               'Object 2': []}}
 */
function decodedObjectsChart() {
  let width,
      height,
      color;

  let chart = function(selection) {

    selection.each(function(decodedObjectsData) {

      let maxWidth = 0,
          maxHeight = 0;
      decodedObjectsData.decodings.forEach(objectName => {
        decodedObjectsData.objects[objectName].forEach(d => {
          maxWidth = Math.max(maxWidth, d.left + d.width);
          maxHeight = Math.max(maxHeight, d.top + d.height);
        });
      });

      let decodings = decodedObjectsData.decodings.slice();

      // Sort by object name.
      decodings.sort();

      let rows = partition(decodings, 3);

      let decodedObjectRow = d3.select(this).selectAll('.decodedObjectRow')
          .data(rows);

      decodedObjectRow.exit().remove();

      decodedObjectRow = decodedObjectRow.enter().append('g')
        .attr('class', 'decodedObjectRow')
        .attr('transform', (d,i) => `translate(0,${i == 0 ? 0 : i*height/3 + 10})`)
        .merge(decodedObjectRow);

      let decodedObject = decodedObjectRow.selectAll('.decodedObject')
          .data(d => d);

      decodedObject.exit().remove();

      decodedObject = decodedObject.enter().append('g')
        .attr('class', 'decodedObject')
        .attr('transform', (d, i) => `translate(${i*width/3},0)`)
        .merge(decodedObject);

      decodedObject.each(function(objectName) {
        let cmMax = Math.max(maxWidth, maxHeight);
        let pxMax = Math.min(width/3, height/3);
        let x = d3.scaleLinear()
            .domain([0, cmMax])
            .range([0, pxMax]);
        let y = d3.scaleLinear()
            .domain([0, cmMax])
            .range([0, pxMax]);

        let feature = d3.select(this).selectAll('.feature')
            .data(decodedObjectsData.objects[objectName]);

        feature.exit().remove();

        feature.enter()
          .append('g')
          .attr('class', 'feature')
          .merge(feature)
          .attr('transform', d => `translate(${x(d.left)},${y(d.top)})`)
          .each(function(featureData) {
            d3.select(this)
              .call(featureChart()
                    .width(x(featureData.width))
                    .height(y(featureData.height))
                    .color(color));
          });
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

  chart.color = function(_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };

  return chart;
}

export {decodedObjectsChart};
