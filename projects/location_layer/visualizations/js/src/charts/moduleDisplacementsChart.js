import * as d3 from 'd3';
import {arrowShape} from './../shapes/arrowShape.js';

/**
 * Data:
 * {modules: [{dimensions: {rows: 5, cols: 5},
 *             scale: 3.2,
 *             orientation: 0.834,
 *             cells: [{cell: 2, state: 'active'}],
 *            ...],
 *  highlightedCells: [10, 46]}
 */
function moduleDisplacementsChart() {
  var width,
      height,
      numRows = 3,
      numCols = 6;

  var chart = function(selection) {
    let modules = selection.selectAll('.modules')
        .data(d => [d]);

    modules = modules.enter()
      .append('g')
      .attr('class', 'modules')
      .merge(modules);

    selection.selectAll('.boundary')
      .data([null])
      .enter().append('rect')
      .attr('class', 'boundary')
      .attr('fill', 'none')
      .attr('stroke', 'black')
      .attr('width', width)
      .attr('height', height);

    modules.each(function(moduleArrayData) {
      let moduleWidth = width / numCols,
          moduleHeight = height / numRows;

      let module = d3.select(this)
          .selectAll('.module')
          .data(moduleArrayData.modules);

      module.exit().remove();

      module = module.enter()
        .append('g')
        .attr('class', 'module')
        .call(enter => {
          enter.append('path');
        })
        .merge(module)
        .attr('transform',
              (d, i) => `translate(${Math.floor(i/numRows) * moduleWidth},${(i%numRows)*moduleHeight})`);

      module.select('path')
        .each(function(d, i) {
          if (d.phaseDisplacement) {
            let displacement = d.phaseDisplacement,
                radians = Math.atan(displacement.top / displacement.left),
                degrees = radians * 180 / Math.PI;
            if (displacement.left < 0) {
              degrees += 180;
            }

            d3.select(this)
              .attr('transform', `translate(12, 9) rotate(${degrees})`)
              .attr('visibility', 'visible')
              .attr('d', arrowShape()
                    .arrowLength(10)
                    .arrowWidth(2)
                    .markerLength(5)
                    .markerWidth(6));
          } else {
            d3.select(this)
              .attr('visibility', 'hidden');
          }

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

  chart.numRows = function(_) {
    if (!arguments.length) return numRows;
    numRows = _;
    return chart;
  };

  chart.numCols = function(_) {
    if (!arguments.length) return numCols;
    numCols = _;
    return chart;
  };

  return chart;
}

export {moduleDisplacementsChart};
