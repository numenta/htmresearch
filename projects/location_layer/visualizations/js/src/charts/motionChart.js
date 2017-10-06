import * as d3 from 'd3';
import {arrowShape} from './../shapes/arrowShape.js';

function motionChart() {
  let maxLength = 900;

  let chart = function(selection) {
    selection.each(function(deltaLocation) {
      let data = deltaLocation != null ? [deltaLocation] : [];

      let arrow = d3.select(this).selectAll('.arrow')
          .data(data);

      arrow.exit().remove();

      let arrowLength,
          correctedArrowLength,
          correctionFactor;
      if (deltaLocation != null) {
        arrowLength = Math.sqrt(Math.pow(deltaLocation.top, 2) +
                                Math.pow(deltaLocation.left, 2));
        correctedArrowLength = Math.min(maxLength, Math.max(15, arrowLength));
        correctionFactor = correctedArrowLength / arrowLength;
      }

      arrow.enter().append('g')
          .attr('class', 'arrow')
          .call(enter => {
            enter.append('path');
          })
        .merge(arrow)
          .attr('transform', d => {
            let radians = Math.atan(d.top / d.left),
                degrees = radians * 180 / Math.PI;
            if (d.left < 0) {
              degrees += 180;
            }
            return `rotate(${degrees})`;
          })
        .select('path')
        .attr('d', d => arrowShape()
            .arrowLength(correctedArrowLength)
            .arrowWidth(3)
            .markerLength(10)
            .markerWidth(8)());

      let leftLabel = d3.select(this).selectAll('.leftLabel')
          .data(deltaLocation == null || deltaLocation.left == 0
                ? []
                : [deltaLocation]);

      leftLabel.exit().remove();

      leftLabel = leftLabel.enter().append('g')
          .attr('class', 'leftLabel')
          .call(enter => {
            enter.append('line')
              .attr('class', 'line')
              .attr('stroke', 'gray')
              .attr('stroke-width', 1);

            let r = 2;

            enter.append('line')
              .attr('class', 'beginCap')
              .attr('stroke', 'gray')
              .attr('stroke-width', 1)
              .attr('y1', -r)
              .attr('y2', r);

            enter.append('line')
              .attr('class', 'endCap')
              .attr('stroke', 'gray')
              .attr('stroke-width', 1)
              .attr('y1', -r)
              .attr('y2', r);

            enter.append('text')
              .attr('text-anchor', 'middle')
              .attr('dy', 12)
              .style('font', '10px Verdana');
          })
        .merge(leftLabel)
        .attr('transform', d => {
          return `translate(0,${Math.abs(correctionFactor*d.top/2) + 10})`;
        });

      leftLabel.select('text')
        .text(d => d3.format('.0f')(Math.abs(d.left)));

      leftLabel.select('.beginCap')
        .attr('x1', d => -correctionFactor*d.left/2)
        .attr('x2', d => -correctionFactor*d.left/2);

      leftLabel.select('.endCap')
        .attr('x1', d => correctionFactor*d.left/2)
        .attr('x2', d => correctionFactor*d.left/2);

      leftLabel.select('.line')
        .attr('x1', d => -correctionFactor*d.left/2)
        .attr('x2', d => correctionFactor*d.left/2);

      let topLabel = d3.select(this).selectAll('.topLabel')
          .data(deltaLocation == null || deltaLocation.top == 0
                ? []
                : [deltaLocation]);

      topLabel.exit().remove();

      topLabel = topLabel.enter().append('g')
          .attr('class', 'topLabel')
          .call(enter => {
            enter.append('line')
              .attr('class', 'line')
              .attr('stroke', 'gray')
              .attr('stroke-width', 1);

            let r = 2;

            enter.append('line')
              .attr('class', 'beginCap')
              .attr('stroke', 'gray')
              .attr('stroke-width', 1)
              .attr('x1', -r)
              .attr('x2', r);

            enter.append('line')
              .attr('class', 'endCap')
              .attr('stroke', 'gray')
              .attr('stroke-width', 1)
              .attr('x1', -r)
              .attr('x2', r);

            enter.append('text')
              .attr('dx', 4)
              .attr('dy', 4)
              .style('font', '10px Verdana');
          })
        .merge(topLabel)
        .attr('transform', d => {
          return `translate(${Math.abs(correctionFactor*d.left/2) + 8},0)`;
        });

      topLabel.select('text')
        .text(d => d3.format('.0f')(Math.abs(d.top)));

      topLabel.select('.beginCap')
        .attr('y1', d => -correctionFactor*d.top/2)
        .attr('y2', d => -correctionFactor*d.top/2);

      topLabel.select('.endCap')
        .attr('y1', d => correctionFactor*d.top/2)
        .attr('y2', d => correctionFactor*d.top/2);

      topLabel.select('.line')
        .attr('y1', d => -correctionFactor*d.top/2)
        .attr('y2', d => correctionFactor*d.top/2);
    });
  };

  chart.maxLength = function(_) {
    if (!arguments.length) return maxLength;
    maxLength = _;
    return chart;
  };

  return chart;
};

export {motionChart};
