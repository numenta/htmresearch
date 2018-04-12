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
      resetWidth = 6,
      repeatOffset = 10,
      betweenRepeatOffset = 6;

  let drawSelectedStep = function(selection) {
    selection.each(function(timelineData) {
      d3.select(this).selectAll('.colorWithSelection')
        .attr('fill', d =>
              (d.timestep.iTimestep == timelineData.selectedIndex &&
               (d.phase === null ||
                d.phase === timelineData.selectedPhase))
              ? 'black'
              : 'lightgray');

      d3.select(this).selectAll('.selectedText')
        .style('visibility', d => d.iTimestep == timelineData.selectedIndex
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
        case 'sense':
          touchNumber++;

          o.text = `Touch ${touchNumber}`;

          shapes.push({
            type: 'sense',
            timestep: o
          });

          break;
        case 'repeat': {
          o.text = 'Settle';
          shapes.push({
            type: 'repeat',
            timestep: o
          });
          break;
        }
        default:
          throw `Unrecognized ${o.type}`;
        }
      });

      let predictionPhaseFn = onchange
          ? d => onchange(d.timestep.iTimestep, 0)
          : null,
          sensePhaseFn = onchange
          ? d => onchange(d.timestep.iTimestep, 1)
          : null;

      let verticalElement = timelineNode.selectAll('.verticalElement')
          .data(shapes);

      verticalElement.exit().remove();

      verticalElement = verticalElement.enter()
        .append('div')
        .attr('class', 'shape')
        .style('position', 'relative')
        .style('display', 'inline-block')
        .style('margin-top', '15px')
        .call(enter => {
          enter.append('svg')
            .attr('height', 40);
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

      let timestep = verticalElement.filter(d => d.type != 'reset')
          .call(s => {
            let selectedText = s.selectAll('.selectedText')
                .data(d => [d.timestep]);

            selectedText.exit().remove();

            selectedText.enter().append('div')
                .attr('class', 'selectedText')
                .style('position', 'absolute')
                .style('width', '50px')
                .style('left', '-18px')
                .style('top', '-16px')
                .style('font', '10px Helvetica')
              .merge(selectedText)
                .text(d => d.text);
          })
          .select('svg')
          .attr('width', senseWidth)
          .selectAll('.timestep')
          .data(d => [d]);

      timestep = timestep.enter()
        .append('g')
          .attr('class', 'timestep')
        .merge(timestep);

      timestep.exit().remove();

      let circle = timestep.selectAll('.nophase')
          .data(d => [Object.assign({phase: null}, d)]);

      circle.exit().remove();

      circle.enter().append('circle')
          .attr('class', 'nophase colorWithSelection')
          .attr('stroke', 'none')
          .attr('cx', 6)
          .attr('cy', 6)
          .attr('cursor', onchange ? 'pointer' : null)
        .merge(circle)
          .attr('r', (d, i) => d.type == 'repeat' ? 3 : 5)
        .on('click', predictionPhaseFn);

      let phase0 = timestep.selectAll('.phase0')
          .data(d => [Object.assign({phase: 0}, d)]);

      phase0.exit().remove();

      phase0.enter().append('text')
          .attr('class', 'repeat colorWithSelection')
          .attr('stroke', 'none')
          .attr('x', 6)
          .attr('y', 25)
          .attr('cursor', onchange ? 'pointer' : null)
          .attr('text-anchor', 'middle')
          .style('font', `bold 13px Helvetica`)
          .text('?')
        .merge(phase0)
          .on('click', predictionPhaseFn);

      let phase1 = timestep.selectAll('.phase1')
          .data(d => [Object.assign({phase: 1}, d)]);

      phase1.exit().remove();

      phase1.enter().append('text')
          .attr('class', 'phase1 colorWithSelection')
          .attr('stroke', 'none')
          .attr('x', 6)
          .attr('y', 40)
          .attr('cursor', onchange ? 'pointer' : null)
          .attr('text-anchor', 'middle')
          .style('font', `bold 13px Helvetica`)
          .text('!')
        .merge(phase1)
          .on('click', sensePhaseFn);
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
