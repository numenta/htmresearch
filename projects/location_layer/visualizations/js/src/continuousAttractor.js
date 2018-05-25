import * as d3 from "d3";


function groupedCells() {

  let color,
      groupDiameter = 16,
      cellOffset = 4,
      cellR = 1.5,
      padding = {top: cellR + 2, left: cellR + 2};

  function chart(selection) {

    let svg = selection.selectAll('svg.groupedCells')
      .data(rows => [rows]);

    svg.exit()
      .remove();

    svg = svg.enter()
      .append('svg')
        .attr('class', 'groupedCells')
        .attr('width', rows => padding.left + rows.length*groupDiameter)
        .attr('height', rows => padding.top + rows.length*groupDiameter)
      .call(enter => {
        enter.append('g')
          .attr('class', 'container')
          .attr('transform', `translate(${padding.left},${padding.top})`);
      })
      .merge(svg);

    let row = svg.select('g.container')
        .selectAll('.row')
        .data(rows => rows);

    row.exit()
      .remove();

    row = row.enter()
      .append('g')
      .attr('class', 'row')
      .attr('transform', (d, i) => `translate(0,${groupDiameter*i})`)
      .merge(row);

    let col = row.selectAll('.col')
      .data(d => d);

    col.exit()
      .remove();

    col = col.enter()
      .append('g')
      .attr('class', 'col')
      .attr('transform', (d, i) => `translate(${groupDiameter*i},0)`)
      .call(enter => {
        [{k: "n", cx: cellOffset, cy: 0},
         {k: "e", cx: 2*cellOffset, cy: cellOffset},
         {k: "s", cx: cellOffset, cy: 2*cellOffset},
         {k: "w", cx: 0, cy: cellOffset}].forEach(direction => {
           enter.append('circle')
             .attr('class', direction.k)
             .attr('cx', direction.cx)
             .attr('cy', direction.cy)
             .attr('r', cellR);
         });
      })
      .merge(col);

    ['n', 'e', 's', 'w'].forEach(k => {
      col.select(`.${k}`)
        .attr('fill', d => {
          return color(d[k]);
        });
    });

  }

  chart.color = function(_) {
    if (!arguments.length) return color;
    color = _;
    return chart;
  };

  return chart;
}


/**
 * @param jsonText
 *
 * {dimensions: [16, 16],
 *  rates: {'n': [0.12, 2.3, ...],
 *          'e': [42.0, 3.7, ...]}}
 */
function insertSpikeRatesSnapshot(node, jsonText) {

  var jsonParsed = JSON.parse(jsonText);

  // Create a 2D array of elements that look like:
  //   {n: 0.12, e: 42.3, s: 0, w: 7.2}
  var rows = [];

  var maxRate = 0;

  d3.range(jsonParsed.dimensions[0]).forEach(row => {
    const base = row * jsonParsed.dimensions[1];

    let rowData = [];

    d3.range(jsonParsed.dimensions[1]).forEach(col => {
      const i = base + col;

      let colData = {};

      for (const k in jsonParsed.rates) {
        colData[k] = jsonParsed.rates[k][i];
        maxRate = Math.max(maxRate, jsonParsed.rates[k][i]);
      }

      rowData.push(colData);
    });

    rows.push(rowData);
  });

  let color = d3.scaleLinear()
      .domain([0, maxRate])
      .range(['white', 'black']);

  d3.select(node)
    .datum(rows)
    .call(groupedCells()
          .color(color));
}

function insertSpikeRatesSnapshotFromUrl(node, jsonUrl) {
  d3.text(jsonUrl, (error, contents) => {
    insertSpikeRatesSnapshot(node, contents);
  });
}

/**
 * @param jsonText
 *
 * {dimensions: [16, 16],
 *  timeResolution: 0.0005,
 *  timesteps: [{'n': [0.12, 2.3, ...],
 *               'e': [42.0, 3.7, ...]},
 *              {'n': [0.12, 2.3, ...],
 *               'e': [42.0, 3.7, ...]}]
 */
function insertSpikeRatesTimeline(node, jsonText) {

  let jsonParsed = JSON.parse(jsonText);

  let timesteps = [];
  let maxRate = 0;

  jsonParsed.timesteps.forEach(timestep => {
    // Create a 2D array of elements that look like:
    //   {n: 0.12, e: 42.3, s: 0, w: 7.2}
    var rows = [];

    d3.range(jsonParsed.dimensions[0]).forEach(row => {
      const base = row * jsonParsed.dimensions[1];

      let rowData = [];

      d3.range(jsonParsed.dimensions[1]).forEach(col => {
        const i = base + col;

        let colData = {};

        for (const k in timestep) {
          colData[k] = timestep[k][i];
          maxRate = Math.max(maxRate, timestep[k][i]);
        }

        rowData.push(colData);
      });

      rows.push(rowData);
    });

    timesteps.push(rows);
  });


  if (timesteps.length == 0) {
    return;
  }

  let cells = [];

  d3.range(jsonParsed.dimensions[0]).forEach(row => {
    d3.range(jsonParsed.dimensions[1]).forEach(col => {
      ['n', 'e', 's', 'w'].forEach(k => {
        cells.push(
          {row, col, k, scalar: null}
        );
      });
    });
  });

  let t = 0;

  let color = d3.scaleLinear()
      .domain([0, maxRate])
      .range(['white', 'black']);

  const groupDiameter = 12;
  const cellOffset = 3;
  const cellR = 2;
  let padding = {top: cellR + 2, left: cellR + 2};

  function x(d) {
    const base = padding.left + d.col*groupDiameter;

    switch (d.k) {
    case 'n':
    case 's':
      return base + cellOffset;
    case 'e':
      return base + 2*cellOffset;
    case 'w':
      return base;
    default:
      throw `Unrecognized k ${d.k}`;
    }
  }

  function y(d) {
    const base = padding.top + d.row*groupDiameter;

    switch (d.k) {
    case 'e':
    case 'w':
      return base + cellOffset;
    case 's':
      return base + 2*cellOffset;
    case 'n':
      return base;
    default:
      throw `Unrecognized k ${d.k}`;
    }
  }

  let width = jsonParsed.dimensions[1]*groupDiameter + padding.left,
      height = jsonParsed.dimensions[0]*groupDiameter + padding.top;

  let svg = d3.select(node).append("svg")
      .attr('width', width)
      .attr('height', height);

  let cell = svg.selectAll('.cell')
      .data(cells);

  cell.exit()
    .remove();

  cell = cell.enter()
    .append('circle')
    .attr('class', 'cell')
    .attr('stroke', 'lightgray')
    .attr('stroke-width', 0.5)
    .attr('r', cellR)
    .merge(cell);

  cell
    .attr('cx', x)
    .attr('cy', y);


  function draw() {
    cell.each(d => {
      d.scalar = timesteps[t][d.row][d.col][d.k];
    });

    cell.attr('fill', d => color(d.scalar));

    setTimeout(() => {
      t = (t + 1) % timesteps.length;

      draw();
    }, 100);
  }

  draw();
}

function insertSpikeRatesTimelineFromUrl(node, jsonUrl) {
  d3.text(jsonUrl, (error, contents) => {
    insertSpikeRatesTimeline(node, contents);
  });
}

function values(object) {
  let out = [];
  for (let k in object) {
    out.push(object[k]);
  }
  return out;
}

/**
 * Draw the cells. When a cell is selected, that cell's row in the matrix
 * is used as weights for all of the cells.
 *
 * @param matrix
 * This maps cells to cells, but each cell is indexed as a group-number + k pair.
 * An entry in this matrix would be indexed with:
 *   matrix[selectedCellGroup][selectedCellK][otherGroup][otherK]
 *
 * where each k is one of: 'n', 'e', 's', 'w'.
 */
function insertWeights(node, matrix, dimensions) {

  let cells = [];

  d3.range(dimensions[0]).forEach(row => {
    d3.range(dimensions[1]).forEach(col => {
      ['n', 'e', 's', 'w'].forEach(k => {
        cells.push(
          {row, col, k, scalar: null}
        );
      });
    });
  });

  const groupDiameter = 30;
  const cellOffset = 7;
  const cellR = 4;
  let padding = {top: cellR + 2, left: cellR + 2};

  function x(d) {
    const base = padding.left + d.col*groupDiameter;

    switch (d.k) {
    case 'n':
    case 's':
      return base + cellOffset;
    case 'e':
      return base + 2*cellOffset;
    case 'w':
      return base;
    default:
      throw `Unrecognized k ${d.k}`;
    }
  }

  function y(d) {
    const base = padding.top + d.row*groupDiameter;

    switch (d.k) {
    case 'e':
    case 'w':
      return base + cellOffset;
    case 's':
      return base + 2*cellOffset;
    case 'n':
      return base;
    default:
      throw `Unrecognized k ${d.k}`;
    }
  }

  let quadtree = d3.quadtree()
      .extent([[0, 0],
               [dimensions[1]*groupDiameter, dimensions[0]*groupDiameter]])
      .x(x)
      .y(y)
      .addAll(cells);

  let dSelected = null;

  let min = Math.min(0, d3.min(matrix, groups => d3.max(values(groups), row => d3.min(row, v => d3.min(values(v))))));
  let max = Math.max(0, d3.max(matrix, groups => d3.max(values(groups), row => d3.max(row, v => d3.max(values(v))))));
  let color = d3.scaleLinear()
      .domain([min, max])
      .range(['red', 'white']);

  let width = dimensions[1]*groupDiameter + padding.left,
      height = dimensions[0]*groupDiameter + padding.top;

  let svg = d3.select(node).append("svg")
      .attr('width', width)
      .attr('height', height)
      .on('mousemove', mousemoved)
      .on('mouseleave', mouseleave);

  let cell = svg.selectAll('.cell')
      .data(cells);

  cell.exit()
    .remove();

  cell = cell.enter()
    .append('circle')
    .attr('class', 'cell')
    .attr('fill', 'white')
    .attr('stroke', 'lightgray')
    .attr('stroke-width', 1)
    .attr('r', cellR)
    .merge(cell);

  cell
    .attr('cx', x)
    .attr('cy', y);

  function draw() {
    if (dSelected === null) {
      cell.each(d => {
        d.scalar = null;
      });
    } else {
      let iSelected = dSelected.row*dimensions[1] + dSelected.col;
      let weights = matrix[iSelected][dSelected.k];
      cell.each(d => {
        let j = d.row*dimensions[1] + d.col;
        d.scalar = weights[j][d.k];
      });
    }

    cell.attr('stroke-width', d => d === dSelected ? 2 : 1);
    cell.attr('stroke', d => d === dSelected ? 'black' : 'lightgray');
    cell.attr('fill', d => d.scalar !== null ? color(d.scalar) : 'white');
  }

  dSelected = cells[42];
  draw();

  function mousemoved() {
    let m = d3.mouse(this);
    let p = quadtree.find(m[0], m[1]);

    if (p !== dSelected) {
      dSelected = p;
      draw();
    }
  }

  function mouseleave() {
    cell.attr('stroke-width', 1);
    cell.attr('stroke', 'lightgray');
    cell.attr('fill', 'white');
  }
}

/**
 * @param jsonText
 * {dimensions: [64, 64],
 *  repeatedWeights: {n: [], e: [], s: [], w: []}
 */
function insertInputWeights(node, jsonText) {
  let jsonParsed = JSON.parse(jsonText);

  let dimensions = jsonParsed.dimensions;
  let numCells = dimensions.reduce((acc, val) => acc * val);

  let ks = ['n', 'e', 's', 'w'];

  let inputMatrix = d3.range(numCells).map(postsynaptic => {

    let weights = d3.range(numCells).map(presynaptic => {

      let presynapticGroup = {};

      ks.forEach(k => {
        presynapticGroup[k] =
          jsonParsed.inputMatrices[k][postsynaptic][presynaptic];
      });

      return presynapticGroup;
    });

    // Built-in assumption in this JSON: The input depends only on the
    // presynaptic cell's k, not the postsynaptic cell's k.
    let postsynapticGroup = {};
    ks.forEach(k => {
      postsynapticGroup[k] = weights;
    });

    return postsynapticGroup;
  });

  insertWeights(node, inputMatrix, jsonParsed.dimensions);
}

function insertInputWeightsFromUrl(node, jsonUrl) {
  d3.text(jsonUrl, (error, contents) => {
    insertInputWeights(node, contents);
  });
}

/**
 * @param jsonText
 * {dimensions: [64, 64],
 *  repeatedWeights: {n: [], e: [], s: [], w: []}
 */
function insertOutputWeights(node, jsonText) {
  let jsonParsed = JSON.parse(jsonText);

  let dimensions = jsonParsed.dimensions;
  let numCells = dimensions.reduce((acc, val) => acc * val);

  let ks = ['n', 'e', 's', 'w'];

  let outputMatrix = d3.range(numCells).map(presynaptic => {

    let presynapticGroup = {};

    ks.forEach(k => {

      let weights = d3.range(numCells).map(postsynaptic => {
        // Built-in assumption in this JSON: The output depends only on the
        // presynaptic cell's k, not the postsynaptic cell's k.
        let postsynapticGroup = {};
        ks.forEach(k2 => {
          postsynapticGroup[k2] =
            jsonParsed.inputMatrices[k][postsynaptic][presynaptic];
        });

        return postsynapticGroup;
      });

      presynapticGroup[k] = weights;
    });

    return presynapticGroup;
  });

  insertWeights(node, outputMatrix, jsonParsed.dimensions);
}

function insertOutputWeightsFromUrl(node, jsonUrl) {
  d3.text(jsonUrl, (error, contents) => {
    insertOutputWeights(node, contents);
  });
}

export {insertSpikeRatesSnapshot, insertSpikeRatesSnapshotFromUrl,
        insertSpikeRatesTimeline, insertSpikeRatesTimelineFromUrl,
        insertInputWeights, insertInputWeightsFromUrl,
        insertOutputWeights, insertOutputWeightsFromUrl};
