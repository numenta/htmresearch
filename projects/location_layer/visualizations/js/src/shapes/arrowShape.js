function arrowShape() {
  var length,
      width,
      markerLength,
      markerWidth;

  let shape = function(_) {
    let start = `M ${-length/2} ${-width/2}`;

    let markerStart = length/2 - markerLength;

    let n1 = `L ${markerStart} ${-width/2}`;
    let n2 = `L ${markerStart} ${-markerWidth/2}`;
    let n3 = `L ${length/2} 0`;
    let n4 = `L ${markerStart} ${markerWidth/2}`;
    let n5 = `L ${markerStart} ${width/2}`;
    let n6 = `L ${-length/2} ${width/2}`;

    let end = 'Z';

    return [start, n1, n2, n3, n4, n5, n6, end].join(' ');
  };

  shape.arrowLength = function(_) {
    if (!arguments.length) return length;
    length = _;
    return shape;
  };

  shape.arrowWidth = function(_) {
    if (!arguments.length) return width;
    width = _;
    return shape;
  };

  shape.markerLength = function(_) {
    if (!arguments.length) return markerLength;
    markerLength = _;
    return shape;
  };

  shape.markerWidth = function(_) {
    if (!arguments.length) return markerWidth;
    markerWidth = _;
    return shape;
  };

  return shape;
}

export {arrowShape};
