import * as d3 from "d3";

/**
 * Example data:
 *    {name: 'A'}
 */
function featureChart() {
  let width,
      height,
      color,
      includeText = true;

  let chart = function(selection) {
    let featureColor = selection.selectAll('.featureColor')
        .data(d => (d && d.name != null) ? [d.name] : []);

    featureColor.exit().remove();

    featureColor.enter()
      .append('rect')
      .attr('class', 'featureColor')
      .attr('width', width)
      .attr('height', height)
      .attr('stroke', 'none')
      .merge(featureColor)
      .attr('fill', d => color(d));

    let featureText = selection.selectAll('.featureText')
        .data(d => (includeText && d && d.name != null) ? [d.name] : []);

    featureText.exit().remove();

    featureText.enter()
      .append('text')
      .attr('class', 'featureText')
      .attr('text-anchor', 'middle')
      .attr('dy', height * 0.25)
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('fill', 'white')
      .style('font', `bold ${height * 0.8}px monospace`)
      .merge(featureText)
      .text(d => d);
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

  chart.includeText = function(_) {
    if (!arguments.length) return includeText;
    includeText = _;
    return chart;
  };

  return chart;
}

export {featureChart};
