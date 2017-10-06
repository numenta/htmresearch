function plusShape() {
  var radius,
      innerRadius;

  let shape = function(_) {
    let diff = radius - innerRadius,
        innerWidth = innerRadius*2,
        innerHeight = innerWidth;

    let innerRight = `M ${innerRadius} ${-innerRadius}`,
        right1 = `l ${diff} 0`,
        down1 = `l 0 ${innerHeight}`,
        left1 = `l ${-diff} 0`,
        down2 = `l 0 ${diff}`,
        left2 = `l ${-innerWidth} 0`,
        up1 = `l 0 ${-diff}`,
        left3 = `l ${-diff} 0`,
        up2 = `l 0 ${-innerHeight}`,
        right2 = `l ${diff} 0`,
        up3 = `l 0 ${-diff}`,
        right3 = `l ${innerWidth} 0`,
        end = 'Z';

    return [innerRight, right1, down1, left1, down2, left2, up1, left3, up2, right2, up3, right3, end].join(' ');
  };

  shape.innerRadius = function(_) {
    if (!arguments.length) return innerRadius;
    innerRadius = _;
    return shape;
  };

  shape.radius = function(_) {
    if (!arguments.length) return radius;
    radius = _;
    return shape;
  };

  return shape;
}

export {plusShape};
