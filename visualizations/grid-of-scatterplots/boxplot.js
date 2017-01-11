function boxPlotLeft() {
  var y,
      radius = 3;

  function chart(selection) {
    var boxPlot = selection.selectAll(".boxPlot")
        .data(quartiles => [quartiles]);

    boxPlot = boxPlot.enter()
      .append("g")
      .attr("class", "boxPlot")
      .call(entering => {

        entering.append("line")
          .attr("class", "upperQuartile")
          .attr("stroke", "black")
          .attr("x1", 0)
          .attr("x2", 0);

        entering.append("line")
          .attr("class", "median")
          .attr("stroke", "black")
          .attr("x1", -radius)
          .attr("x2", radius);

        entering.append("line")
          .attr("class", "lowerQuartile")
          .attr("stroke", "black")
          .attr("x1", 0)
          .attr("x2", 0);

        entering.append("rect")
          .attr("class", "box")
          .attr("x", -radius)
          .attr("width", 2*radius)
          .attr("stroke", "black")
          .attr("fill", "none");


        var xPos = -radius - 3;

        entering.append("text")
          .attr("class", "label0")
          .attr("dominant-baseline", "central")
          .attr("text-anchor", "end")
          .attr("x", xPos);

        entering.append("text")
          .attr("class", "label1")
          .attr("dominant-baseline", "central")
          .attr("text-anchor", "end")
          .attr("x", xPos);

        entering.append("text")
          .attr("class", "label2")
          .attr("dominant-baseline", "central")
          .attr("text-anchor", "end")
          .attr("x", xPos);

        entering.append("text")
          .attr("class", "label3")
          .attr("dominant-baseline", "central")
          .attr("text-anchor", "end")
          .attr("x", xPos);

        entering.append("text")
          .attr("class", "label4")
          .attr("dominant-baseline", "central")
          .attr("text-anchor", "end")
          .attr("x", xPos);
      })
      .merge(boxPlot);

    boxPlot.select(".lowerQuartile")
      .attr("y1", d => y(d[0]))
      .attr("y2", d => y(d[1]));

    boxPlot.select(".median")
      .attr("y1", d => y(d[2]))
      .attr("y2", d => y(d[2]));

    boxPlot.select(".upperQuartile")
      .attr("y1", d => y(d[3]))
      .attr("y2", d => y(d[4]));

    boxPlot.select(".box")
      .attr("y", d => y(d[3]))
      .attr("height", d => y(d[1]) - y(d[3]));

    var verticalLabelSpace = 8;

    boxPlot.select(".label2")
      .attr("y", d => y(d[2]))
      .text(d => Math.round(d[2]));

    boxPlot.select(".label0")
      .attr("y", d => y(d[0]))
      .text(d => {
        var yPos = y(d[0]);
        var dRight = yPos - y(d[2]);

        if (dRight < verticalLabelSpace) {
          return "";
        } else {
          return Math.round(d[0]);
        }
      });

    boxPlot.select(".label4")
      .attr("y", d => y(d[4]))
      .text(d => {
        var yPos = y(d[4]);
        var dLeft = y(d[2]) - yPos;

        if (dLeft < verticalLabelSpace) {
          return "";
        } else {
          return Math.round(d[4]);
        }
      });

    boxPlot.select(".label1")
      .attr("y", d => y(d[1]))
      .text(d => {
        var yPos = y(d[1]);
        var dLeft = y(d[0]) - yPos;
        var dRight = yPos - y(d[2]);

        if (dLeft < verticalLabelSpace ||
            dRight < verticalLabelSpace) {
          return "";
        } else {
          return Math.round(d[1]);
        }
      });

    boxPlot.select(".label3")
      .attr("y", d => y(d[3]))
      .text(d => {
        var yPos = y(d[3]);
        var dLeft = y(d[2]) - yPos;
        var dRight = yPos - y(d[4]);

        if (dLeft < verticalLabelSpace || dRight < verticalLabelSpace) {
          return "";
        } else {
          return Math.round(d[3]);
        }
      });
  };

  chart.y = function(_) {
    if (!arguments.length) return y;
    y = _;
    return chart;
  };

  chart.radius = function(_) {
    if (!arguments.length) return radius;
    radius = _;
    return chart;
  };

  return chart;
}

function boxPlotBottom() {
  var x,
      radius = 3;

  function chart(selection) {
    var boxPlot = selection.selectAll(".boxPlot")
        .data(quartiles => [quartiles]);

    boxPlot = boxPlot.enter()
      .append("g")
      .attr("class", "boxPlot")
      .call(entering => {

        entering.append("line")
          .attr("class", "upperQuartile")
          .attr("stroke", "black")
          .attr("y1", 0)
          .attr("y2", 0);

        entering.append("line")
          .attr("class", "median")
          .attr("stroke", "black")
          .attr("y1", -radius)
          .attr("y2", radius);

        entering.append("line")
          .attr("class", "lowerQuartile")
          .attr("stroke", "black")
          .attr("y1", 0)
          .attr("y2", 0);

        entering.append("rect")
          .attr("class", "box")
          .attr("y", -radius)
          .attr("height", 2*radius)
          .attr("stroke", "black")
          .attr("fill", "none");

        var yPos = radius + 11;

        entering.append("text")
          .attr("class", "label0")
          .attr("text-anchor", "middle")
          .attr("y", yPos);

        entering.append("text")
          .attr("class", "label1")
          .attr("text-anchor", "middle")
          .attr("y", yPos);

        entering.append("text")
          .attr("class", "label2")
          .attr("text-anchor", "middle")
          .attr("y", yPos);

        entering.append("text")
          .attr("class", "label3")
          .attr("text-anchor", "middle")
          .attr("y", yPos);

        entering.append("text")
          .attr("class", "label4")
          .attr("text-anchor", "middle")
          .attr("y", yPos);
      })
      .merge(boxPlot);

    boxPlot.select(".lowerQuartile")
      .attr("x1", d => x(d[0]))
      .attr("x2", d => x(d[1]));

    boxPlot.select(".median")
      .attr("x1", d => x(d[2]))
      .attr("x2", d => x(d[2]));

    boxPlot.select(".upperQuartile")
      .attr("x1", d => x(d[3]))
      .attr("x2", d => x(d[4]));

    boxPlot.select(".box")
      .attr("x", d => x(d[1]))
      .attr("width", d => x(d[3]) - x(d[1]));


    var horizontalLabelSpace = 16;

    boxPlot.select(".label2")
      .attr("x", d => x(d[2]))
      .text(d => Math.round(d[2]));

    boxPlot.select(".label0")
      .attr("x", d => x(d[0]))
      .text(d => {
        var xPos = x(d[0]);
        var dRight = x(d[2]) - xPos;

        if (dRight < horizontalLabelSpace) {
          return "";
        } else {
          return Math.round(d[0]);
        }
      });

    boxPlot.select(".label4")
      .attr("x", d => x(d[4]))
      .text(d => {
        var xPos = x(d[4]);
        var dLeft = xPos - x(d[2]);

        if (dLeft < horizontalLabelSpace) {
          return "";
        } else {
          return Math.round(d[4]);
        }
      });

    boxPlot.select(".label1")
      .attr("x", d => x(d[1]))
      .text(d => {
        var xPos = x(d[1]);
        var dLeft = xPos - x(d[0]);
        var dRight = x(d[2]) - xPos;

        if (dLeft < horizontalLabelSpace || dRight < horizontalLabelSpace) {
          return "";
        } else {
          return Math.round(d[1]);
        }
      });

    boxPlot.select(".label3")
      .attr("x", d => x(d[3]))
      .text(d => {
        var xPos = x(d[3]);
        var dLeft = xPos - x(d[2]);
        var dRight = x(d[4]) - xPos;

        if (dLeft < horizontalLabelSpace || dRight < horizontalLabelSpace) {
          return "";
        } else {
          return Math.round(d[3]);
        }
      });
  };

  chart.x = function(_) {
    if (!arguments.length) return x;
    x = _;
    return chart;
  };

  return chart;
}
