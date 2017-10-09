import * as d3 from "d3";

/**
 * Example params:
 * {
 *   inputSize: 100,
 *   activeBits: [42, 45]
 * }
 */
function arrayOfAxonsChart() {
  let width,
      height,
      borderWidth = 1,
      rectWidth = 2;

  let chart = function(selection) {
    selection.each(function(axonsData) {
      if (!axonsData) {
        d3.select(this).selectAll('.activeAxon')
          .data(d => [])
          .exit()
          .remove();
        return;
      }

      let x = d3.scaleLinear()
          .domain([0, axonsData.inputSize])
          .range([4, width - 4]);

      d3.select(this)
        .selectAll('.border')
        .data([null])
        .enter().append('rect')
        .attr('class', 'border')
        .attr('fill', 'none')
        .attr('stroke', 'black')
        .attr('stroke-width', borderWidth)
        .attr('width', width)
        .attr('height', height);

      let activeAxon = d3.select(this).selectAll('.activeAxon')
          .data(d => d.activeBits);

      activeAxon.enter().append('g')
        .attr('class', 'activeAxon')
        .call(enter => {
          enter.append('rect')
            .attr('x', -rectWidth / 2)
            .attr('width', rectWidth)
            .attr('height', 5)
            .attr('stroke', 'none')
            .attr('fill', 'brown');
        })
        .merge(activeAxon)
        .attr('transform', cell => `translate(${x(cell)}, 2.5)`);

      activeAxon.exit().remove();
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

  chart.rectWidth = function(_) {
    if (!arguments.length) return rectWidth;
    rectWidth = _;
    return chart;
  };

  chart.borderWidth = function(_) {
    if (!arguments.length) return borderWidth;
    borderWidth = _;
    return chart;
  };

  return chart;
}

export {arrayOfAxonsChart};
