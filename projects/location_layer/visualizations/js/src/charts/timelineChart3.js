import * as d3 from 'd3';
import {arrowShape} from './../shapes/arrowShape.js';

/**
 * Data example:
 * [{timesteps: [{}, {}, {reset: true}, {}, {}]
 *   selectedIndex: 2},
 *  ...]
 *
 * "reset: true" implies that a reset happened before that timestep.
 */
function timelineChart() {
  var onchange,
      senseWidth = 16,
      moveWidth = 16,
      resetWidth = 6,
      repeatOffset = 10,
      betweenRepeatOffset = 6;

  let drawSelectedStep = function(selection) {
    selection.each(function(timelineData) {
      d3.select(this).selectAll('.colorWithSelection')
        .attr('fill', d => (d.iTimestep == timelineData.selectedIndex &&
                            d.iTimestepPhase == timelineData.selectedPhase)
              ? 'black'
              : 'lightgray');

      d3.select(this).selectAll('.move.colorWithSelection')
        .attr('stroke', d => (d.iTimestep == timelineData.selectedIndex &&
                              d.iTimestepPhase == timelineData.selectedPhase)
              ? 'black'
              : 'lightgray');

      d3.select(this).selectAll('.selectedText')
        .style('visibility', d => (d.iTimestep == timelineData.selectedIndex &&
                                   d.iTimestepPhase == timelineData.selectedPhase)
               ? 'visible'
               : 'hidden');
    });
  };

  var chart = function(selection) {
    let timeline = selection.selectAll('.timeline')
        .data(d => [d]);

    timeline = timeline.enter()
      .append('g')
      .attr('class', 'timeline')
      .merge(timeline);

    timeline.each(function(timelineData) {
      let timelineNode = d3.select(this);

      let shapes = [];

      let touchNumber = 0;


      timelineData.timesteps.forEach((timestep, iTimestep) => {

        if (timestep.reset && iTimestep !== 0) {
          shapes.push({type: 'reset'});
          touchNumber = 0;
        }

        let o = Object.assign({iTimestep}, timestep);

        switch (o.type) {
        case 'initial':
          touchNumber++;

          o.text = `Touch ${touchNumber}`;

          shapes.push({
            type: 'sense',
            timesteps: [Object.assign({iTimestepPhase: 1}, o)]
          });

          break;
        case 'move':
          shapes.push(
            Object.assign({text: 'Move',
                           iTimestepPhase: 0}, o));

          touchNumber++;
          shapes.push({
            type: 'sense',
            timesteps: [Object.assign({text: `Touch ${touchNumber}`,
                                       iTimestepPhase: 1}, o)]
          });
          break;
        case 'settle': {
          let touchData = shapes[shapes.length-1];
          o.text = 'Settle';
          touchData.timesteps.push(Object.assign({iTimestepPhase: 0}, o));
          touchData.timesteps.push(Object.assign({iTimestepPhase: 1}, o));
          break;
        }
        default:
          throw `Unrecognized ${o.type}`;
        }
      });

      let onchangeFn = onchange
          ? d => onchange(d.iTimestep, d.iTimestepPhase)
          : null;

      let verticalElement = timelineNode.selectAll('.verticalElement')
          .data(shapes);

      verticalElement.exit().remove();

      // Clean up updating nodes.
      // shape.filter(d => d.type == 'reset')
      //   .selectAll(':scope > :not(.reset)')
      //   .remove();
      // shape.filter(d => d.type == 'move')
      //   .selectAll(':scope > :not(.move)')
      //   .remove();
      // shape.filter(d => d.type == 'sense')
      //   .selectAll(':scope > :not(.sense)')
      //   .remove();

      verticalElement = verticalElement.enter()
        .append('div')
        .attr('class', 'shape')
        .style('position', 'relative')
        .style('display', 'inline-block')
        .style('margin-top', '15px')
        .call(enter => {
          enter.append('svg')
            .attr('height', 30);
        })
        .merge(verticalElement);

      verticalElement.filter(d => d.type == 'reset')
          .style('width', `${resetWidth}px`)
          .style('top', '-2px')
          .select('svg')
        .attr('width', resetWidth)
        .selectAll('.reset')
        .data([null])
        .enter().append('rect')
          .attr('class', 'reset')
          .attr('height', 15)
          .attr('width', 2)
        .attr('fill', 'gray');


      let move = verticalElement.filter(d => d.type == 'move')
          .style('width', `${moveWidth}px`)
          .call(m => {
            let selectedText = m.selectAll('.selectedText')
                .data(d => [d]);

            selectedText.enter().append('div')
              .attr('class', 'selectedText')
              .style('position', 'absolute')
              .style('width', '50px')
              .style('left', '-18px')
              .style('top', '-16px')
              .style('font', '10px Verdana')
              .merge(selectedText)
              .text(d => d.text);
          })
          .select('svg')
          .attr('width', moveWidth)
          .selectAll('.move')
          .data(d => [d]);

      move.enter()
        .append('g')
          .attr('class', 'move colorWithSelection')
          .attr('cursor', onchange ? 'pointer' : null)
          .call(enter => {
            enter.append('path')
              .attr('d', arrowShape()
                    .arrowLength(12)
                    .arrowWidth(3)
                    .markerLength(5)
                    .markerWidth(8));
          })
        .merge(move)
        .attr('transform', d => {
          let radians = Math.atan(d.deltaLocation.top / d.deltaLocation.left),
              degrees = radians * 180 / Math.PI;
          if (d.deltaLocation.left < 0) {
            degrees += 180;
          }
          return `translate(6, 6) rotate(${degrees})`;
        })
        .on('click', onchangeFn);

      let sense = verticalElement.filter(d => d.type == 'sense')
            .style('width', `${senseWidth}px`)
          .call(s => {

            let selectedText = s.selectAll('.selectedText')
                .data(d => d.timesteps);

            selectedText.exit().remove();

            selectedText.enter().append('div')
              .attr('class', 'selectedText')
              .style('position', 'absolute')
              .style('width', '50px')
              .style('left', '-18px')
              .style('top', '-16px')
              .style('font', '10px Verdana')
              .merge(selectedText)
              .text(d => d.text);
          })
          .select('svg')
          .attr('width', senseWidth)
          .selectAll('.sense')
          .data(d => [d]);

      sense = sense.enter()
        .append('g')
          .attr('class', 'sense')
        .merge(sense);

      let repeat = sense.selectAll('.repeat')
          .data(d => d.timesteps);

      repeat.exit().remove();

      repeat.enter().append('circle')
          .attr('class', 'repeat colorWithSelection')
          .attr('stroke', 'none')
          .attr('r', (d, i) => i == 0 ? 6 : 2)
          .attr('cx', 6)
          .attr('cy', (d, i) => 6 + (i == 0
                                     ? 0
                                     : repeatOffset + (i-1)*betweenRepeatOffset))
          .attr('cursor', onchange ? 'pointer' : null)
        .merge(repeat)
          .on('click', onchangeFn);
    });

    drawSelectedStep(selection);
  };

  chart.drawSelectedStep = drawSelectedStep;

  chart.onchange = function(_) {
    if (!arguments.length) return onchange;
    onchange = _;
    return chart;
  };

  return chart;
}

export {timelineChart};
