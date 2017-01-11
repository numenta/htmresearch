/**
 * @param experiments
 * Parameters and results. Each result is a pair:
 * [# correctly active cells, # incorrectly active cells]
 *
 * Example:
 *
 * [{numColumns: 1:, noiseLevel: 0.10, results: [[40,  0], [40,  0], [30, 20]]},
 *  {numColumns: 1:, noiseLevel: 0.20, results: [[40, 10], [40, 12], [28, 20]]}
 */
function insertScatterplotGrid_numColumns_noiseLevel(experiments) {

  // Constants not dependent on the data
  let padding = {top: 50, left: 5, right: 160, bottom: 100},

      // How far the dot area of each chart is from the sides of the rectangle
      chartPadding = {top: 5, right: 10, left: 35, bottom: 27},

      pixelsPerUnit = 0.75,
      axisLeftOffset = 7,
      axisBottomOffset = 7;

  let allNoiseLevels = sortedUnique(experiments, e => e.noiseLevel),
      allColumnCounts = sortedUnique(experiments, e => e.numColumns),
      maxCorrect = d3.max(experiments, e => d3.max(e.results, d => d[0])),
      maxIncorrect = d3.max(experiments, e => d3.max(e.results, d => d[1])),

      // The width and height of the area for dots
      scatterplotWidth = maxCorrect * pixelsPerUnit,
      scatterplotHeight = maxIncorrect * pixelsPerUnit,

      // The width and height of each rectangle in the grid
      chartWidth = chartPadding.left + scatterplotWidth + chartPadding.right,
      chartHeight = chartPadding.top + scatterplotHeight + chartPadding.bottom,

      // The width and height of the entire grid
      gridWidth = allNoiseLevels.length * chartWidth,
      gridHeight = allColumnCounts.length * chartHeight;

  var gridX = d3.scaleOrdinal()
      .domain(allNoiseLevels)
      .range(d3.range(0, gridWidth, chartWidth));

  var gridY = d3.scaleOrdinal()
      .domain(allColumnCounts)
      .range(d3.range(0, gridHeight, chartHeight));

  var svg = d3.select("#graphics").append("svg")
      .attr("width", padding.left + gridWidth + padding.right)
      .attr("height", padding.top + gridHeight + padding.bottom);

  var container = svg.append("g")
      .attr("transform", `translate(${padding.left},${padding.top})`);

  var x = d3.scaleLinear()
      .domain([0, maxCorrect])
      .range([0, scatterplotWidth]);

  var y = d3.scaleLinear()
      .domain([0, maxIncorrect])
      .range([scatterplotHeight, 0]);

  var chart = container.selectAll(".chart").data(experiments);
  chart = chart.enter()
    .append("g")
      .attr("class", "chart")
    .merge(chart)
    .attr("transform", (experiment, i) =>
          `translate(${chartPadding.left + gridX(experiment.noiseLevel)},` +
          `${chartPadding.top + gridY(experiment.numColumns)})`);

  var point = chart.append("g")
      .attr("class", "points")
    .selectAll(".point")
    .data(d => d.results);

  point = point.enter()
    .append("circle")
      .attr("class", "point")
      .attr("r", 1)
      .attr("fill", "black")
      .attr("fill-opacity", 0.4)
    .merge(point)
      .attr("cx", d => x(d[0]))
      .attr("cy", d => y(d[1]));

  chart
    .append("g")
    .attr("class", "incorrectActiveBoxPlot")
    .attr("transform", `translate(${-axisLeftOffset}, 0)`)
    .datum(data => {
      var sorted =  data.results.map(d => d[1]).sort((a,b) => a - b);
      var quartiles = [0, .25, .5, .75, 1.0]
          .map(p => d3.quantile(sorted, p));
      return quartiles;
    })
    .call(boxPlotLeft()
          .y(y));

  chart
    .append("g")
    .attr("class", "correctActiveBoxPlot")
    .attr("transform", `translate(0,${scatterplotHeight + axisBottomOffset})`)
    .datum(data => {
      var sorted = data.results.map(d => d[0]).sort((a,b) => a - b);
      var quartiles = [0, .25, .5, .75, 1.0]
          .map(p => d3.quantile(sorted, p));
      return quartiles;
    })
    .call(boxPlotBottom()
          .x(x));

  container.append("text")
    .attr("class", "gridAxisTopLabel")
    .attr("x", chartWidth * allNoiseLevels.length / 2)
    .attr("y", -30)
    .attr("text-anchor", "middle")
    .text("Cell activity with n% noise");

  var legendY = allColumnCounts.length * chartHeight;
  var legendX = chartWidth * allNoiseLevels.length + 90;

  container.append("line")
    .attr("x1", legendX)
    .attr("x2", legendX)
    .attr("y1", legendY)
    .attr("y2", legendY + 40)
    .attr("stroke", "black");

  container.append("line")
    .attr("x1", legendX)
    .attr("x2", legendX + 30)
    .attr("y1", legendY + 40)
    .attr("y2", legendY + 40)
    .attr("stroke", "black");

  container.append("text")
    .attr("class", "axisLeftLabel")
    .attr("y", legendY + 5)
    .attr("text-anchor", "end")
    .call(text => {
      text.append("tspan")
        .attr("x", legendX - 5)
        .attr("dy", ".6em")
        .text("Incorrect");
      text.append("tspan")
        .attr("x", legendX - 5)
        .attr("dy", "1.2em")
        .text("active cells");
    });

  container.append("text")
    .attr("class", "axisBottomLabel")
    .attr("y", legendY + 45)
    .attr("text-anchor", "beginning")
    .call(text => {
      text.append("tspan")
        .attr("x", legendX)
        .attr("dy", ".6em")
        .text("Correct");
      text.append("tspan")
        .attr("x", legendX)
        .attr("dy", "1.2em")
        .text("active cells");
    });

  var noiseLabel = container.selectAll(".noiseLabel")
      .data(allNoiseLevels);
  noiseLabel = noiseLabel.enter()
    .append("text")
    .attr("class", "noiseLabel")
    .attr("text-anchor", "middle")
    .attr("y", -5)
    .merge(noiseLabel)
    .attr("x", d => gridX(d) + chartWidth/2)
    .text(d => d3.format(".0%")(d));

  var numColumnsLabel = container.selectAll(".numColumnsLabel")
      .data(allColumnCounts);
  numColumnsLabel = numColumnsLabel.enter()
    .append("text")
    .attr("class", "numColumnsLabel")
    .merge(numColumnsLabel)
    .attr("x", d => allNoiseLevels.length * chartWidth)
    .attr("y", d => gridY(d) + chartHeight / 2)
    .text(d => `${d} column${d != 1 ? "s" : ""}`);

  container.selectAll(".horizontalLine")
    .data(d3.range(allColumnCounts.length - 1))
    .enter()
    .append("line")
      .attr("class", "horizontalLine")
      .attr("stroke", "lightgray")
      .attr("stroke-width", 1)
      .attr("y1", (d, i) => (i+1)*chartHeight)
      .attr("y2", (d, i) => (i+1)*chartHeight)
      .attr("x1", 0)
      .attr("x2", chartWidth*allNoiseLevels.length);

  container.selectAll(".verticalLine")
    .data(d3.range(allNoiseLevels.length - 1))
    .enter()
    .append("line")
      .attr("class", "verticalLine")
      .attr("stroke", "lightgray")
      .attr("stroke-width", 1)
      .attr("x1", (d, i) => (i+1)*chartWidth)
      .attr("x2", (d, i) => (i+1)*chartWidth)
      .attr("y1", 0)
      .attr("y2", chartHeight*allColumnCounts.length);

}

function handleFile(file) {
  var fileReader = new FileReader();
  fileReader.onload = function (e) {
    insertScatterplotGrid_numColumns_noiseLevel(JSON.parse(fileReader.result));
  };
  fileReader.onerror = function (e) {
    throw 'Error reading CSV file';
  };

  fileReader.readAsText(file);
}

function handleFiles(files) {
  for (let i = 0; i < files.length; i++) {
    handleFile(files[i]);
  }
}

function sortedUnique(arr, getValue, compare) {
  if (!getValue) {
    getValue = x => x;
  }

  if (!compare) {
    compare = (a, b) => a - b;
  }

  let values = [];
  arr.forEach(d => {
    let v = getValue(d);
    if (values.indexOf(v) == -1) {
      values.push(v);
    }
  });

  values.sort(compare);

  return values;
}

document.addEventListener("dragover", evt => {
  evt.preventDefault();
});

document.addEventListener("drop", evt => {
  evt.preventDefault();
  handleFiles(evt.dataTransfer.files);
});
