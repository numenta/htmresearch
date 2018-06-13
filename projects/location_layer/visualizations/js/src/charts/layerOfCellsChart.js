import * as d3 from "d3";

/**
 * Example params:
 * {
 *   dimensions: {rows: 5, cols: 5},
 *   cells: [{cell: 42, state: 'active'},
 *           {cell: 43, state: 'predicted'},
 *           {cell: 44, state: 'predicted-active'}]
 *   highlightedCells: [42]
 * }
 */
function layerOfCellsChart() {
  let width,
      height,
      color = d3.scaleOrdinal()
        .domain(['active', 'predicted', 'predicted-active', 'inhibited'])
        .range(['black', 'rgba(0, 127, 255, 0.498)', 'black', 'darkseagreen']),
      stroke = "black",
      onCellSelected = (selectedCell, id) => {},
      columnMajorIndexing = false;

  let drawHighlightedCells = function(selection) {
    selection.each(function(layerData) {
      let xScale = d3.scaleLinear()
          .domain([0, layerData.dimensions.cols - 1])
          .range([5, width - 5]),
          yScale = d3.scaleLinear()
          .domain([0, layerData.dimensions.rows - 1])
          .range([5, height - 5]);

      let x, y;
      if (columnMajorIndexing) {
        x = cell => xScale(Math.floor(cell / layerData.dimensions.rows));
        y = cell => yScale(cell % layerData.dimensions.rows);
      } else {
        x = cell => xScale(cell % layerData.dimensions.cols);
        y = cell => yScale(Math.floor(cell / layerData.dimensions.cols));
      }

      let highlightedCell = d3.select(this)
        .select('.front')
        .selectAll('.highlightedCell')
        .data(layerData.highlightedCells);
      highlightedCell.exit().remove();
      highlightedCell = highlightedCell.enter()
        .append('polygon')
          .attr('class', 'highlightedCell')
          .attr('points', '0,-4 2,2 -2,2')
          .attr('stroke', 'goldenrod')
          .attr('stroke-width', 2)
        .merge(highlightedCell)
          .attr('transform', d => `translate(${x(d)},${y(d)})`);
    });
  };

  let chart = function(selection) {
    selection.each(function(layerData) {
      let layerNode = this,
          layer = d3.select(layerNode),
          xScale = d3.scaleLinear()
            .domain([0, layerData.dimensions.cols - 1])
            .range([5, width - 5]),
          yScale = d3.scaleLinear()
            .domain([0, layerData.dimensions.rows - 1])
            .range([5, height - 5]);

      let x, y;
      if (columnMajorIndexing) {
        x = d => xScale(Math.floor(d.cell / layerData.dimensions.rows));
        y = d => yScale(d.cell % layerData.dimensions.rows);
      } else {
        x = d => xScale(d.cell % layerData.dimensions.cols);
        y = d => yScale(Math.floor(d.cell / layerData.dimensions.cols));
      }

      // For each layer, keep:
      // - the selected cell, so that we fire events only when the selection
      //   changes.
      // - the mouse position from the most recent mousemove, so that we can
      //   reevaluate the selected cell when the data changes.
      if (layerNode._selectedCell === undefined) {
        layerNode._selectedCell = null;
      }
      if (layerNode._mousePosition === undefined) {
        layerNode._mousePosition = null;
      }

      layer.selectAll('.border')
        .data([null])
        .enter().append('rect')
          .attr('class', 'border')
          .attr('stroke', stroke)
          .attr('stroke-width', 1)
          .attr('fill', 'none')
          .attr('width', width)
          .attr('height', height);

      let main = layer.selectAll(':scope > .main')
          .data([null]);
      main.exit().remove();
      main = main.enter()
        .append('g')
        .attr('class', 'main')
        .merge(main);

      layer.selectAll(':scope > .front')
        .data([null])
        .enter()
        .append('g')
        .attr('class', 'front');

      let cells = main.selectAll('.cells')
          .data([layerData.cells]);

      cells = cells.enter()
        .append('g')
          .attr('class', 'cells')
        .merge(cells);

      let cell = cells.selectAll('.cell')
          .data(d => d);

      cell.exit().remove();

      cell = cell.enter()
        .append('polygon')
          .attr('class', 'cell')
        .merge(cell)
          .attr('fill', d => color(d.state))
          .attr('stroke-width', d =>
                layerData.highlightedCells.indexOf(d.cell) != -1
                ? 2
                : 0);

      // Enable fast lookup of the cell nearest to the cursor.
      let quadtree = d3.quadtree()
          .extent([[0, 0], [width, height]])
          .x(x)
          .y(y)
          .addAll(layerData.cells);

      let mouseEvents = layer.selectAll('.mouseEvents')
          .data([null]);

      mouseEvents.enter()
        .append('rect')
          .attr('class', 'mouseEvents')
          .attr('stroke', 'transparent')
          .attr('fill', 'transparent')
          .attr('width', width)
          .attr('height', height)
        .merge(mouseEvents)
        .on('mousemove', function() {
          layerNode._mousePosition = d3.mouse(this);

          let p = quadtree.find(layerNode._mousePosition[0],
                                layerNode._mousePosition[1]);

          if (p !== layerNode._selectedCell) {
            layerNode._selectedCell = p;
            draw();
            onCellSelected(p ? p.cell : null, layerData.id);
          }
        })
        .on('mouseleave', () => {
          layerNode._mousePosition = null;

          if (layerNode._selectedCell !== null) {
            layerNode._selectedCell = null;
            draw();
            onCellSelected(null, layerData.id);
          }
        });

      // If we're rerendering, check if it has caused the nearest cell to
      // change.
      if (layerNode._mousePosition !== null) {
        let p = quadtree.find(layerNode._mousePosition[0],
                              layerNode._mousePosition[1]);
        if (p !== layerNode._selectedCell) {
          layerNode._selectedCell = p;
          onCellSelected(p ? p.cell : null, layerData.id);
        }
      }

      draw();


      function draw() {
        cell
          .attr('transform', d => `translate(${x(d)},${y(d)})`)
          .attr('points', cell =>
                cell == layerNode._selectedCell
                ? '0,-6 5,6 -5,6'
                : (cell.state == 'predicted')
                ? '0,-3 1.5,1.5 -1.5,1.5'
                : '0,-4 2,2 -2,2');
      }
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

  chart.stroke = function(_) {
    if (!arguments.length) return stroke;
    stroke = _;
    return chart;
  };

  chart.onCellSelected = function(_) {
    if (!arguments.length) return onCellSelected;
    onCellSelected = _;
    return chart;
  };

  chart.columnMajorIndexing = function(_) {
    if (!arguments.length) return columnMajorIndexing;
    columnMajorIndexing = _;
    return chart;
  };

  chart.color = function(_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };

  return chart;
}

export {layerOfCellsChart};
