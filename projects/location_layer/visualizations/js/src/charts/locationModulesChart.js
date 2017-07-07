import * as d3 from 'd3';
import {layerOfCellsChart} from './layerOfCellsChart.js';

/**
 * Data:
 * {modules: [{dimensions: {rows: 5, cols: 5},
 *             scale: 3.2,
 *             orientation: 0.834,
 *             cells: [{cell: 2, state: 'active'}],
 *            ...],
 *  highlightedCells: [10, 46]}
 */
function locationModulesChart() {
  var width,
      height,
      onCellSelected = (iModule, selectedCell) => {};

  let drawHighlightedCells = function(selection) {
    selection.each(function(moduleArrayData) {
      let moduleWidth = width / 6,
          moduleHeight = height / 3,
          highlightedCellsByModule = [];

      let base = 0;
      moduleArrayData.modules.forEach(module => {
        let end = base + module.dimensions.rows*module.dimensions.cols;

        let highlightedInModule = [];
        moduleArrayData.highlightedCells.forEach(cell => {
          if (cell >= base && cell < end) {
            highlightedInModule.push(cell - base);
          }
        });

        highlightedCellsByModule.push(highlightedInModule);

        base = end;
      });

      let module = d3.select(this)
          .selectAll('.module')
          .datum((d, i) => {
            d.highlightedCells = highlightedCellsByModule[i];
            return d;
          })
          .each(function(d, i) {
            d3.select(this).call(
              layerOfCellsChart()
                .width(moduleWidth)
                .height(moduleHeight)
                .stroke('lightgray')
                .onCellSelected(
                  cell => onCellSelected(cell !== null ? i : null,
                                         cell))
                .drawHighlightedCells);
          });
    });
  };

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
      // TODO: stop hardcoding 18 modules
      let moduleWidth = width / 6,
          moduleHeight = height / 3;

      let module = d3.select(this)
          .selectAll('.module')
          .data(moduleArrayData.modules.map(m => {
            return Object.assign({highlightedCells: []}, m);
          }));

      module.exit().remove();

      module = module.enter().append('g')
        .attr('class', 'module')
        .merge(module)
        .attr('transform',
              (d, i) => `translate(${Math.floor(i/3) * moduleWidth},${(i%3)*moduleHeight})`)
        .each(function(d, i) {
          d3.select(this)
            .call(layerOfCellsChart()
                  .width(moduleWidth)
                  .height(moduleHeight)
                  .stroke('lightgray')
                  .onCellSelected(
                    cell => onCellSelected(cell !== null ? i : null,
                                           cell)));
        });
    });
  };

  chart.drawHighlightedCells = drawHighlightedCells;

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

  chart.onCellSelected = function(_) {
    if (!arguments.length) return onCellSelected;
    onCellSelected = _;
    return chart;
  };

  return chart;
}

export {locationModulesChart};
