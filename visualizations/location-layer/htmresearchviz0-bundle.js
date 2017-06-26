(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
	typeof define === 'function' && define.amd ? define(['exports'], factory) :
	(factory((global.htmresearchviz0 = global.htmresearchviz0 || {})));
}(this, (function (exports) { 'use strict';

var ascending = function(a, b) {
  return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
};

var bisector = function(compare) {
  if (compare.length === 1) compare = ascendingComparator(compare);
  return {
    left: function(a, x, lo, hi) {
      if (lo == null) lo = 0;
      if (hi == null) hi = a.length;
      while (lo < hi) {
        var mid = lo + hi >>> 1;
        if (compare(a[mid], x) < 0) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    },
    right: function(a, x, lo, hi) {
      if (lo == null) lo = 0;
      if (hi == null) hi = a.length;
      while (lo < hi) {
        var mid = lo + hi >>> 1;
        if (compare(a[mid], x) > 0) hi = mid;
        else lo = mid + 1;
      }
      return lo;
    }
  };
};

function ascendingComparator(f) {
  return function(d, x) {
    return ascending(f(d), x);
  };
}

var ascendingBisect = bisector(ascending);
var bisectRight = ascendingBisect.right;

function pair(a, b) {
  return [a, b];
}

var number = function(x) {
  return x === null ? NaN : +x;
};

var extent = function(values, valueof) {
  var n = values.length,
      i = -1,
      value,
      min,
      max;

  if (valueof == null) {
    while (++i < n) { // Find the first comparable value.
      if ((value = values[i]) != null && value >= value) {
        min = max = value;
        while (++i < n) { // Compare the remaining values.
          if ((value = values[i]) != null) {
            if (min > value) min = value;
            if (max < value) max = value;
          }
        }
      }
    }
  }

  else {
    while (++i < n) { // Find the first comparable value.
      if ((value = valueof(values[i], i, values)) != null && value >= value) {
        min = max = value;
        while (++i < n) { // Compare the remaining values.
          if ((value = valueof(values[i], i, values)) != null) {
            if (min > value) min = value;
            if (max < value) max = value;
          }
        }
      }
    }
  }

  return [min, max];
};

var identity = function(x) {
  return x;
};

var sequence = function(start, stop, step) {
  start = +start, stop = +stop, step = (n = arguments.length) < 2 ? (stop = start, start = 0, 1) : n < 3 ? 1 : +step;

  var i = -1,
      n = Math.max(0, Math.ceil((stop - start) / step)) | 0,
      range = new Array(n);

  while (++i < n) {
    range[i] = start + i * step;
  }

  return range;
};

var e10 = Math.sqrt(50);
var e5 = Math.sqrt(10);
var e2 = Math.sqrt(2);

var ticks = function(start, stop, count) {
  var reverse = stop < start,
      i = -1,
      n,
      ticks,
      step;

  if (reverse) n = start, start = stop, stop = n;

  if ((step = tickIncrement(start, stop, count)) === 0 || !isFinite(step)) return [];

  if (step > 0) {
    start = Math.ceil(start / step);
    stop = Math.floor(stop / step);
    ticks = new Array(n = Math.ceil(stop - start + 1));
    while (++i < n) ticks[i] = (start + i) * step;
  } else {
    start = Math.floor(start * step);
    stop = Math.ceil(stop * step);
    ticks = new Array(n = Math.ceil(start - stop + 1));
    while (++i < n) ticks[i] = (start - i) / step;
  }

  if (reverse) ticks.reverse();

  return ticks;
};

function tickIncrement(start, stop, count) {
  var step = (stop - start) / Math.max(0, count),
      power = Math.floor(Math.log(step) / Math.LN10),
      error = step / Math.pow(10, power);
  return power >= 0
      ? (error >= e10 ? 10 : error >= e5 ? 5 : error >= e2 ? 2 : 1) * Math.pow(10, power)
      : -Math.pow(10, -power) / (error >= e10 ? 10 : error >= e5 ? 5 : error >= e2 ? 2 : 1);
}

function tickStep(start, stop, count) {
  var step0 = Math.abs(stop - start) / Math.max(0, count),
      step1 = Math.pow(10, Math.floor(Math.log(step0) / Math.LN10)),
      error = step0 / step1;
  if (error >= e10) step1 *= 10;
  else if (error >= e5) step1 *= 5;
  else if (error >= e2) step1 *= 2;
  return stop < start ? -step1 : step1;
}

var sturges = function(values) {
  return Math.ceil(Math.log(values.length) / Math.LN2) + 1;
};

var threshold = function(values, p, valueof) {
  if (valueof == null) valueof = number;
  if (!(n = values.length)) return;
  if ((p = +p) <= 0 || n < 2) return +valueof(values[0], 0, values);
  if (p >= 1) return +valueof(values[n - 1], n - 1, values);
  var n,
      i = (n - 1) * p,
      i0 = Math.floor(i),
      value0 = +valueof(values[i0], i0, values),
      value1 = +valueof(values[i0 + 1], i0 + 1, values);
  return value0 + (value1 - value0) * (i - i0);
};

var merge = function(arrays) {
  var n = arrays.length,
      m,
      i = -1,
      j = 0,
      merged,
      array;

  while (++i < n) j += arrays[i].length;
  merged = new Array(j);

  while (--n >= 0) {
    array = arrays[n];
    m = array.length;
    while (--m >= 0) {
      merged[--j] = array[m];
    }
  }

  return merged;
};

var min = function(values, valueof) {
  var n = values.length,
      i = -1,
      value,
      min;

  if (valueof == null) {
    while (++i < n) { // Find the first comparable value.
      if ((value = values[i]) != null && value >= value) {
        min = value;
        while (++i < n) { // Compare the remaining values.
          if ((value = values[i]) != null && min > value) {
            min = value;
          }
        }
      }
    }
  }

  else {
    while (++i < n) { // Find the first comparable value.
      if ((value = valueof(values[i], i, values)) != null && value >= value) {
        min = value;
        while (++i < n) { // Compare the remaining values.
          if ((value = valueof(values[i], i, values)) != null && min > value) {
            min = value;
          }
        }
      }
    }
  }

  return min;
};

function length(d) {
  return d.length;
}

var noop = {value: function() {}};

function dispatch() {
  for (var i = 0, n = arguments.length, _ = {}, t; i < n; ++i) {
    if (!(t = arguments[i] + "") || (t in _)) throw new Error("illegal type: " + t);
    _[t] = [];
  }
  return new Dispatch(_);
}

function Dispatch(_) {
  this._ = _;
}

function parseTypenames(typenames, types) {
  return typenames.trim().split(/^|\s+/).map(function(t) {
    var name = "", i = t.indexOf(".");
    if (i >= 0) name = t.slice(i + 1), t = t.slice(0, i);
    if (t && !types.hasOwnProperty(t)) throw new Error("unknown type: " + t);
    return {type: t, name: name};
  });
}

Dispatch.prototype = dispatch.prototype = {
  constructor: Dispatch,
  on: function(typename, callback) {
    var _ = this._,
        T = parseTypenames(typename + "", _),
        t,
        i = -1,
        n = T.length;

    // If no callback was specified, return the callback of the given type and name.
    if (arguments.length < 2) {
      while (++i < n) if ((t = (typename = T[i]).type) && (t = get(_[t], typename.name))) return t;
      return;
    }

    // If a type was specified, set the callback for the given type and name.
    // Otherwise, if a null callback was specified, remove callbacks of the given name.
    if (callback != null && typeof callback !== "function") throw new Error("invalid callback: " + callback);
    while (++i < n) {
      if (t = (typename = T[i]).type) _[t] = set(_[t], typename.name, callback);
      else if (callback == null) for (t in _) _[t] = set(_[t], typename.name, null);
    }

    return this;
  },
  copy: function() {
    var copy = {}, _ = this._;
    for (var t in _) copy[t] = _[t].slice();
    return new Dispatch(copy);
  },
  call: function(type, that) {
    if ((n = arguments.length - 2) > 0) for (var args = new Array(n), i = 0, n, t; i < n; ++i) args[i] = arguments[i + 2];
    if (!this._.hasOwnProperty(type)) throw new Error("unknown type: " + type);
    for (t = this._[type], i = 0, n = t.length; i < n; ++i) t[i].value.apply(that, args);
  },
  apply: function(type, that, args) {
    if (!this._.hasOwnProperty(type)) throw new Error("unknown type: " + type);
    for (var t = this._[type], i = 0, n = t.length; i < n; ++i) t[i].value.apply(that, args);
  }
};

function get(type, name) {
  for (var i = 0, n = type.length, c; i < n; ++i) {
    if ((c = type[i]).name === name) {
      return c.value;
    }
  }
}

function set(type, name, callback) {
  for (var i = 0, n = type.length; i < n; ++i) {
    if (type[i].name === name) {
      type[i] = noop, type = type.slice(0, i).concat(type.slice(i + 1));
      break;
    }
  }
  if (callback != null) type.push({name: name, value: callback});
  return type;
}

var xhtml = "http://www.w3.org/1999/xhtml";

var namespaces = {
  svg: "http://www.w3.org/2000/svg",
  xhtml: xhtml,
  xlink: "http://www.w3.org/1999/xlink",
  xml: "http://www.w3.org/XML/1998/namespace",
  xmlns: "http://www.w3.org/2000/xmlns/"
};

var namespace = function(name) {
  var prefix = name += "", i = prefix.indexOf(":");
  if (i >= 0 && (prefix = name.slice(0, i)) !== "xmlns") name = name.slice(i + 1);
  return namespaces.hasOwnProperty(prefix) ? {space: namespaces[prefix], local: name} : name;
};

function creatorInherit(name) {
  return function() {
    var document = this.ownerDocument,
        uri = this.namespaceURI;
    return uri === xhtml && document.documentElement.namespaceURI === xhtml
        ? document.createElement(name)
        : document.createElementNS(uri, name);
  };
}

function creatorFixed(fullname) {
  return function() {
    return this.ownerDocument.createElementNS(fullname.space, fullname.local);
  };
}

var creator = function(name) {
  var fullname = namespace(name);
  return (fullname.local
      ? creatorFixed
      : creatorInherit)(fullname);
};

var matcher = function(selector) {
  return function() {
    return this.matches(selector);
  };
};

if (typeof document !== "undefined") {
  var element = document.documentElement;
  if (!element.matches) {
    var vendorMatches = element.webkitMatchesSelector
        || element.msMatchesSelector
        || element.mozMatchesSelector
        || element.oMatchesSelector;
    matcher = function(selector) {
      return function() {
        return vendorMatches.call(this, selector);
      };
    };
  }
}

var matcher$1 = matcher;

var filterEvents = {};

var event = null;

if (typeof document !== "undefined") {
  var element$1 = document.documentElement;
  if (!("onmouseenter" in element$1)) {
    filterEvents = {mouseenter: "mouseover", mouseleave: "mouseout"};
  }
}

function filterContextListener(listener, index, group) {
  listener = contextListener(listener, index, group);
  return function(event) {
    var related = event.relatedTarget;
    if (!related || (related !== this && !(related.compareDocumentPosition(this) & 8))) {
      listener.call(this, event);
    }
  };
}

function contextListener(listener, index, group) {
  return function(event1) {
    var event0 = event; // Events can be reentrant (e.g., focus).
    event = event1;
    try {
      listener.call(this, this.__data__, index, group);
    } finally {
      event = event0;
    }
  };
}

function parseTypenames$1(typenames) {
  return typenames.trim().split(/^|\s+/).map(function(t) {
    var name = "", i = t.indexOf(".");
    if (i >= 0) name = t.slice(i + 1), t = t.slice(0, i);
    return {type: t, name: name};
  });
}

function onRemove(typename) {
  return function() {
    var on = this.__on;
    if (!on) return;
    for (var j = 0, i = -1, m = on.length, o; j < m; ++j) {
      if (o = on[j], (!typename.type || o.type === typename.type) && o.name === typename.name) {
        this.removeEventListener(o.type, o.listener, o.capture);
      } else {
        on[++i] = o;
      }
    }
    if (++i) on.length = i;
    else delete this.__on;
  };
}

function onAdd(typename, value, capture) {
  var wrap = filterEvents.hasOwnProperty(typename.type) ? filterContextListener : contextListener;
  return function(d, i, group) {
    var on = this.__on, o, listener = wrap(value, i, group);
    if (on) for (var j = 0, m = on.length; j < m; ++j) {
      if ((o = on[j]).type === typename.type && o.name === typename.name) {
        this.removeEventListener(o.type, o.listener, o.capture);
        this.addEventListener(o.type, o.listener = listener, o.capture = capture);
        o.value = value;
        return;
      }
    }
    this.addEventListener(typename.type, listener, capture);
    o = {type: typename.type, name: typename.name, value: value, listener: listener, capture: capture};
    if (!on) this.__on = [o];
    else on.push(o);
  };
}

var selection_on = function(typename, value, capture) {
  var typenames = parseTypenames$1(typename + ""), i, n = typenames.length, t;

  if (arguments.length < 2) {
    var on = this.node().__on;
    if (on) for (var j = 0, m = on.length, o; j < m; ++j) {
      for (i = 0, o = on[j]; i < n; ++i) {
        if ((t = typenames[i]).type === o.type && t.name === o.name) {
          return o.value;
        }
      }
    }
    return;
  }

  on = value ? onAdd : onRemove;
  if (capture == null) capture = false;
  for (i = 0; i < n; ++i) this.each(on(typenames[i], value, capture));
  return this;
};

var sourceEvent = function() {
  var current = event, source;
  while (source = current.sourceEvent) current = source;
  return current;
};

var point = function(node, event) {
  var svg = node.ownerSVGElement || node;

  if (svg.createSVGPoint) {
    var point = svg.createSVGPoint();
    point.x = event.clientX, point.y = event.clientY;
    point = point.matrixTransform(node.getScreenCTM().inverse());
    return [point.x, point.y];
  }

  var rect = node.getBoundingClientRect();
  return [event.clientX - rect.left - node.clientLeft, event.clientY - rect.top - node.clientTop];
};

var mouse = function(node) {
  var event = sourceEvent();
  if (event.changedTouches) event = event.changedTouches[0];
  return point(node, event);
};

function none() {}

var selector = function(selector) {
  return selector == null ? none : function() {
    return this.querySelector(selector);
  };
};

var selection_select = function(select) {
  if (typeof select !== "function") select = selector(select);

  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, subgroup = subgroups[j] = new Array(n), node, subnode, i = 0; i < n; ++i) {
      if ((node = group[i]) && (subnode = select.call(node, node.__data__, i, group))) {
        if ("__data__" in node) subnode.__data__ = node.__data__;
        subgroup[i] = subnode;
      }
    }
  }

  return new Selection(subgroups, this._parents);
};

function empty$1() {
  return [];
}

var selectorAll = function(selector) {
  return selector == null ? empty$1 : function() {
    return this.querySelectorAll(selector);
  };
};

var selection_selectAll = function(select) {
  if (typeof select !== "function") select = selectorAll(select);

  for (var groups = this._groups, m = groups.length, subgroups = [], parents = [], j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        subgroups.push(select.call(node, node.__data__, i, group));
        parents.push(node);
      }
    }
  }

  return new Selection(subgroups, parents);
};

var selection_filter = function(match) {
  if (typeof match !== "function") match = matcher$1(match);

  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, subgroup = subgroups[j] = [], node, i = 0; i < n; ++i) {
      if ((node = group[i]) && match.call(node, node.__data__, i, group)) {
        subgroup.push(node);
      }
    }
  }

  return new Selection(subgroups, this._parents);
};

var sparse = function(update) {
  return new Array(update.length);
};

var selection_enter = function() {
  return new Selection(this._enter || this._groups.map(sparse), this._parents);
};

function EnterNode(parent, datum) {
  this.ownerDocument = parent.ownerDocument;
  this.namespaceURI = parent.namespaceURI;
  this._next = null;
  this._parent = parent;
  this.__data__ = datum;
}

EnterNode.prototype = {
  constructor: EnterNode,
  appendChild: function(child) { return this._parent.insertBefore(child, this._next); },
  insertBefore: function(child, next) { return this._parent.insertBefore(child, next); },
  querySelector: function(selector) { return this._parent.querySelector(selector); },
  querySelectorAll: function(selector) { return this._parent.querySelectorAll(selector); }
};

var constant$1 = function(x) {
  return function() {
    return x;
  };
};

var keyPrefix = "$"; // Protect against keys like “__proto__”.

function bindIndex(parent, group, enter, update, exit, data) {
  var i = 0,
      node,
      groupLength = group.length,
      dataLength = data.length;

  // Put any non-null nodes that fit into update.
  // Put any null nodes into enter.
  // Put any remaining data into enter.
  for (; i < dataLength; ++i) {
    if (node = group[i]) {
      node.__data__ = data[i];
      update[i] = node;
    } else {
      enter[i] = new EnterNode(parent, data[i]);
    }
  }

  // Put any non-null nodes that don’t fit into exit.
  for (; i < groupLength; ++i) {
    if (node = group[i]) {
      exit[i] = node;
    }
  }
}

function bindKey(parent, group, enter, update, exit, data, key) {
  var i,
      node,
      nodeByKeyValue = {},
      groupLength = group.length,
      dataLength = data.length,
      keyValues = new Array(groupLength),
      keyValue;

  // Compute the key for each node.
  // If multiple nodes have the same key, the duplicates are added to exit.
  for (i = 0; i < groupLength; ++i) {
    if (node = group[i]) {
      keyValues[i] = keyValue = keyPrefix + key.call(node, node.__data__, i, group);
      if (keyValue in nodeByKeyValue) {
        exit[i] = node;
      } else {
        nodeByKeyValue[keyValue] = node;
      }
    }
  }

  // Compute the key for each datum.
  // If there a node associated with this key, join and add it to update.
  // If there is not (or the key is a duplicate), add it to enter.
  for (i = 0; i < dataLength; ++i) {
    keyValue = keyPrefix + key.call(parent, data[i], i, data);
    if (node = nodeByKeyValue[keyValue]) {
      update[i] = node;
      node.__data__ = data[i];
      nodeByKeyValue[keyValue] = null;
    } else {
      enter[i] = new EnterNode(parent, data[i]);
    }
  }

  // Add any remaining nodes that were not bound to data to exit.
  for (i = 0; i < groupLength; ++i) {
    if ((node = group[i]) && (nodeByKeyValue[keyValues[i]] === node)) {
      exit[i] = node;
    }
  }
}

var selection_data = function(value, key) {
  if (!value) {
    data = new Array(this.size()), j = -1;
    this.each(function(d) { data[++j] = d; });
    return data;
  }

  var bind = key ? bindKey : bindIndex,
      parents = this._parents,
      groups = this._groups;

  if (typeof value !== "function") value = constant$1(value);

  for (var m = groups.length, update = new Array(m), enter = new Array(m), exit = new Array(m), j = 0; j < m; ++j) {
    var parent = parents[j],
        group = groups[j],
        groupLength = group.length,
        data = value.call(parent, parent && parent.__data__, j, parents),
        dataLength = data.length,
        enterGroup = enter[j] = new Array(dataLength),
        updateGroup = update[j] = new Array(dataLength),
        exitGroup = exit[j] = new Array(groupLength);

    bind(parent, group, enterGroup, updateGroup, exitGroup, data, key);

    // Now connect the enter nodes to their following update node, such that
    // appendChild can insert the materialized enter node before this node,
    // rather than at the end of the parent node.
    for (var i0 = 0, i1 = 0, previous, next; i0 < dataLength; ++i0) {
      if (previous = enterGroup[i0]) {
        if (i0 >= i1) i1 = i0 + 1;
        while (!(next = updateGroup[i1]) && ++i1 < dataLength);
        previous._next = next || null;
      }
    }
  }

  update = new Selection(update, parents);
  update._enter = enter;
  update._exit = exit;
  return update;
};

var selection_exit = function() {
  return new Selection(this._exit || this._groups.map(sparse), this._parents);
};

var selection_merge = function(selection) {

  for (var groups0 = this._groups, groups1 = selection._groups, m0 = groups0.length, m1 = groups1.length, m = Math.min(m0, m1), merges = new Array(m0), j = 0; j < m; ++j) {
    for (var group0 = groups0[j], group1 = groups1[j], n = group0.length, merge = merges[j] = new Array(n), node, i = 0; i < n; ++i) {
      if (node = group0[i] || group1[i]) {
        merge[i] = node;
      }
    }
  }

  for (; j < m0; ++j) {
    merges[j] = groups0[j];
  }

  return new Selection(merges, this._parents);
};

var selection_order = function() {

  for (var groups = this._groups, j = -1, m = groups.length; ++j < m;) {
    for (var group = groups[j], i = group.length - 1, next = group[i], node; --i >= 0;) {
      if (node = group[i]) {
        if (next && next !== node.nextSibling) next.parentNode.insertBefore(node, next);
        next = node;
      }
    }
  }

  return this;
};

var selection_sort = function(compare) {
  if (!compare) compare = ascending$1;

  function compareNode(a, b) {
    return a && b ? compare(a.__data__, b.__data__) : !a - !b;
  }

  for (var groups = this._groups, m = groups.length, sortgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, sortgroup = sortgroups[j] = new Array(n), node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        sortgroup[i] = node;
      }
    }
    sortgroup.sort(compareNode);
  }

  return new Selection(sortgroups, this._parents).order();
};

function ascending$1(a, b) {
  return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
}

var selection_call = function() {
  var callback = arguments[0];
  arguments[0] = this;
  callback.apply(null, arguments);
  return this;
};

var selection_nodes = function() {
  var nodes = new Array(this.size()), i = -1;
  this.each(function() { nodes[++i] = this; });
  return nodes;
};

var selection_node = function() {

  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group = groups[j], i = 0, n = group.length; i < n; ++i) {
      var node = group[i];
      if (node) return node;
    }
  }

  return null;
};

var selection_size = function() {
  var size = 0;
  this.each(function() { ++size; });
  return size;
};

var selection_empty = function() {
  return !this.node();
};

var selection_each = function(callback) {

  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group = groups[j], i = 0, n = group.length, node; i < n; ++i) {
      if (node = group[i]) callback.call(node, node.__data__, i, group);
    }
  }

  return this;
};

function attrRemove(name) {
  return function() {
    this.removeAttribute(name);
  };
}

function attrRemoveNS(fullname) {
  return function() {
    this.removeAttributeNS(fullname.space, fullname.local);
  };
}

function attrConstant(name, value) {
  return function() {
    this.setAttribute(name, value);
  };
}

function attrConstantNS(fullname, value) {
  return function() {
    this.setAttributeNS(fullname.space, fullname.local, value);
  };
}

function attrFunction(name, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) this.removeAttribute(name);
    else this.setAttribute(name, v);
  };
}

function attrFunctionNS(fullname, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) this.removeAttributeNS(fullname.space, fullname.local);
    else this.setAttributeNS(fullname.space, fullname.local, v);
  };
}

var selection_attr = function(name, value) {
  var fullname = namespace(name);

  if (arguments.length < 2) {
    var node = this.node();
    return fullname.local
        ? node.getAttributeNS(fullname.space, fullname.local)
        : node.getAttribute(fullname);
  }

  return this.each((value == null
      ? (fullname.local ? attrRemoveNS : attrRemove) : (typeof value === "function"
      ? (fullname.local ? attrFunctionNS : attrFunction)
      : (fullname.local ? attrConstantNS : attrConstant)))(fullname, value));
};

var defaultView = function(node) {
  return (node.ownerDocument && node.ownerDocument.defaultView) // node is a Node
      || (node.document && node) // node is a Window
      || node.defaultView; // node is a Document
};

function styleRemove(name) {
  return function() {
    this.style.removeProperty(name);
  };
}

function styleConstant(name, value, priority) {
  return function() {
    this.style.setProperty(name, value, priority);
  };
}

function styleFunction(name, value, priority) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) this.style.removeProperty(name);
    else this.style.setProperty(name, v, priority);
  };
}

var selection_style = function(name, value, priority) {
  return arguments.length > 1
      ? this.each((value == null
            ? styleRemove : typeof value === "function"
            ? styleFunction
            : styleConstant)(name, value, priority == null ? "" : priority))
      : styleValue(this.node(), name);
};

function styleValue(node, name) {
  return node.style.getPropertyValue(name)
      || defaultView(node).getComputedStyle(node, null).getPropertyValue(name);
}

function propertyRemove(name) {
  return function() {
    delete this[name];
  };
}

function propertyConstant(name, value) {
  return function() {
    this[name] = value;
  };
}

function propertyFunction(name, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null) delete this[name];
    else this[name] = v;
  };
}

var selection_property = function(name, value) {
  return arguments.length > 1
      ? this.each((value == null
          ? propertyRemove : typeof value === "function"
          ? propertyFunction
          : propertyConstant)(name, value))
      : this.node()[name];
};

function classArray(string) {
  return string.trim().split(/^|\s+/);
}

function classList(node) {
  return node.classList || new ClassList(node);
}

function ClassList(node) {
  this._node = node;
  this._names = classArray(node.getAttribute("class") || "");
}

ClassList.prototype = {
  add: function(name) {
    var i = this._names.indexOf(name);
    if (i < 0) {
      this._names.push(name);
      this._node.setAttribute("class", this._names.join(" "));
    }
  },
  remove: function(name) {
    var i = this._names.indexOf(name);
    if (i >= 0) {
      this._names.splice(i, 1);
      this._node.setAttribute("class", this._names.join(" "));
    }
  },
  contains: function(name) {
    return this._names.indexOf(name) >= 0;
  }
};

function classedAdd(node, names) {
  var list = classList(node), i = -1, n = names.length;
  while (++i < n) list.add(names[i]);
}

function classedRemove(node, names) {
  var list = classList(node), i = -1, n = names.length;
  while (++i < n) list.remove(names[i]);
}

function classedTrue(names) {
  return function() {
    classedAdd(this, names);
  };
}

function classedFalse(names) {
  return function() {
    classedRemove(this, names);
  };
}

function classedFunction(names, value) {
  return function() {
    (value.apply(this, arguments) ? classedAdd : classedRemove)(this, names);
  };
}

var selection_classed = function(name, value) {
  var names = classArray(name + "");

  if (arguments.length < 2) {
    var list = classList(this.node()), i = -1, n = names.length;
    while (++i < n) if (!list.contains(names[i])) return false;
    return true;
  }

  return this.each((typeof value === "function"
      ? classedFunction : value
      ? classedTrue
      : classedFalse)(names, value));
};

function textRemove() {
  this.textContent = "";
}

function textConstant(value) {
  return function() {
    this.textContent = value;
  };
}

function textFunction(value) {
  return function() {
    var v = value.apply(this, arguments);
    this.textContent = v == null ? "" : v;
  };
}

var selection_text = function(value) {
  return arguments.length
      ? this.each(value == null
          ? textRemove : (typeof value === "function"
          ? textFunction
          : textConstant)(value))
      : this.node().textContent;
};

function htmlRemove() {
  this.innerHTML = "";
}

function htmlConstant(value) {
  return function() {
    this.innerHTML = value;
  };
}

function htmlFunction(value) {
  return function() {
    var v = value.apply(this, arguments);
    this.innerHTML = v == null ? "" : v;
  };
}

var selection_html = function(value) {
  return arguments.length
      ? this.each(value == null
          ? htmlRemove : (typeof value === "function"
          ? htmlFunction
          : htmlConstant)(value))
      : this.node().innerHTML;
};

function raise() {
  if (this.nextSibling) this.parentNode.appendChild(this);
}

var selection_raise = function() {
  return this.each(raise);
};

function lower() {
  if (this.previousSibling) this.parentNode.insertBefore(this, this.parentNode.firstChild);
}

var selection_lower = function() {
  return this.each(lower);
};

var selection_append = function(name) {
  var create = typeof name === "function" ? name : creator(name);
  return this.select(function() {
    return this.appendChild(create.apply(this, arguments));
  });
};

function constantNull() {
  return null;
}

var selection_insert = function(name, before) {
  var create = typeof name === "function" ? name : creator(name),
      select = before == null ? constantNull : typeof before === "function" ? before : selector(before);
  return this.select(function() {
    return this.insertBefore(create.apply(this, arguments), select.apply(this, arguments) || null);
  });
};

function remove() {
  var parent = this.parentNode;
  if (parent) parent.removeChild(this);
}

var selection_remove = function() {
  return this.each(remove);
};

var selection_datum = function(value) {
  return arguments.length
      ? this.property("__data__", value)
      : this.node().__data__;
};

function dispatchEvent(node, type, params) {
  var window = defaultView(node),
      event = window.CustomEvent;

  if (typeof event === "function") {
    event = new event(type, params);
  } else {
    event = window.document.createEvent("Event");
    if (params) event.initEvent(type, params.bubbles, params.cancelable), event.detail = params.detail;
    else event.initEvent(type, false, false);
  }

  node.dispatchEvent(event);
}

function dispatchConstant(type, params) {
  return function() {
    return dispatchEvent(this, type, params);
  };
}

function dispatchFunction(type, params) {
  return function() {
    return dispatchEvent(this, type, params.apply(this, arguments));
  };
}

var selection_dispatch = function(type, params) {
  return this.each((typeof params === "function"
      ? dispatchFunction
      : dispatchConstant)(type, params));
};

var root = [null];

function Selection(groups, parents) {
  this._groups = groups;
  this._parents = parents;
}

function selection() {
  return new Selection([[document.documentElement]], root);
}

Selection.prototype = selection.prototype = {
  constructor: Selection,
  select: selection_select,
  selectAll: selection_selectAll,
  filter: selection_filter,
  data: selection_data,
  enter: selection_enter,
  exit: selection_exit,
  merge: selection_merge,
  order: selection_order,
  sort: selection_sort,
  call: selection_call,
  nodes: selection_nodes,
  node: selection_node,
  size: selection_size,
  empty: selection_empty,
  each: selection_each,
  attr: selection_attr,
  style: selection_style,
  property: selection_property,
  classed: selection_classed,
  text: selection_text,
  html: selection_html,
  raise: selection_raise,
  lower: selection_lower,
  append: selection_append,
  insert: selection_insert,
  remove: selection_remove,
  datum: selection_datum,
  on: selection_on,
  dispatch: selection_dispatch
};

var select = function(selector) {
  return typeof selector === "string"
      ? new Selection([[document.querySelector(selector)]], [document.documentElement])
      : new Selection([[selector]], root);
};

var define = function(constructor, factory, prototype) {
  constructor.prototype = factory.prototype = prototype;
  prototype.constructor = constructor;
};

function extend(parent, definition) {
  var prototype = Object.create(parent.prototype);
  for (var key in definition) prototype[key] = definition[key];
  return prototype;
}

function Color() {}

var darker = 0.7;
var brighter = 1 / darker;

var reI = "\\s*([+-]?\\d+)\\s*";
var reN = "\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*";
var reP = "\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*";
var reHex3 = /^#([0-9a-f]{3})$/;
var reHex6 = /^#([0-9a-f]{6})$/;
var reRgbInteger = new RegExp("^rgb\\(" + [reI, reI, reI] + "\\)$");
var reRgbPercent = new RegExp("^rgb\\(" + [reP, reP, reP] + "\\)$");
var reRgbaInteger = new RegExp("^rgba\\(" + [reI, reI, reI, reN] + "\\)$");
var reRgbaPercent = new RegExp("^rgba\\(" + [reP, reP, reP, reN] + "\\)$");
var reHslPercent = new RegExp("^hsl\\(" + [reN, reP, reP] + "\\)$");
var reHslaPercent = new RegExp("^hsla\\(" + [reN, reP, reP, reN] + "\\)$");

var named = {
  aliceblue: 0xf0f8ff,
  antiquewhite: 0xfaebd7,
  aqua: 0x00ffff,
  aquamarine: 0x7fffd4,
  azure: 0xf0ffff,
  beige: 0xf5f5dc,
  bisque: 0xffe4c4,
  black: 0x000000,
  blanchedalmond: 0xffebcd,
  blue: 0x0000ff,
  blueviolet: 0x8a2be2,
  brown: 0xa52a2a,
  burlywood: 0xdeb887,
  cadetblue: 0x5f9ea0,
  chartreuse: 0x7fff00,
  chocolate: 0xd2691e,
  coral: 0xff7f50,
  cornflowerblue: 0x6495ed,
  cornsilk: 0xfff8dc,
  crimson: 0xdc143c,
  cyan: 0x00ffff,
  darkblue: 0x00008b,
  darkcyan: 0x008b8b,
  darkgoldenrod: 0xb8860b,
  darkgray: 0xa9a9a9,
  darkgreen: 0x006400,
  darkgrey: 0xa9a9a9,
  darkkhaki: 0xbdb76b,
  darkmagenta: 0x8b008b,
  darkolivegreen: 0x556b2f,
  darkorange: 0xff8c00,
  darkorchid: 0x9932cc,
  darkred: 0x8b0000,
  darksalmon: 0xe9967a,
  darkseagreen: 0x8fbc8f,
  darkslateblue: 0x483d8b,
  darkslategray: 0x2f4f4f,
  darkslategrey: 0x2f4f4f,
  darkturquoise: 0x00ced1,
  darkviolet: 0x9400d3,
  deeppink: 0xff1493,
  deepskyblue: 0x00bfff,
  dimgray: 0x696969,
  dimgrey: 0x696969,
  dodgerblue: 0x1e90ff,
  firebrick: 0xb22222,
  floralwhite: 0xfffaf0,
  forestgreen: 0x228b22,
  fuchsia: 0xff00ff,
  gainsboro: 0xdcdcdc,
  ghostwhite: 0xf8f8ff,
  gold: 0xffd700,
  goldenrod: 0xdaa520,
  gray: 0x808080,
  green: 0x008000,
  greenyellow: 0xadff2f,
  grey: 0x808080,
  honeydew: 0xf0fff0,
  hotpink: 0xff69b4,
  indianred: 0xcd5c5c,
  indigo: 0x4b0082,
  ivory: 0xfffff0,
  khaki: 0xf0e68c,
  lavender: 0xe6e6fa,
  lavenderblush: 0xfff0f5,
  lawngreen: 0x7cfc00,
  lemonchiffon: 0xfffacd,
  lightblue: 0xadd8e6,
  lightcoral: 0xf08080,
  lightcyan: 0xe0ffff,
  lightgoldenrodyellow: 0xfafad2,
  lightgray: 0xd3d3d3,
  lightgreen: 0x90ee90,
  lightgrey: 0xd3d3d3,
  lightpink: 0xffb6c1,
  lightsalmon: 0xffa07a,
  lightseagreen: 0x20b2aa,
  lightskyblue: 0x87cefa,
  lightslategray: 0x778899,
  lightslategrey: 0x778899,
  lightsteelblue: 0xb0c4de,
  lightyellow: 0xffffe0,
  lime: 0x00ff00,
  limegreen: 0x32cd32,
  linen: 0xfaf0e6,
  magenta: 0xff00ff,
  maroon: 0x800000,
  mediumaquamarine: 0x66cdaa,
  mediumblue: 0x0000cd,
  mediumorchid: 0xba55d3,
  mediumpurple: 0x9370db,
  mediumseagreen: 0x3cb371,
  mediumslateblue: 0x7b68ee,
  mediumspringgreen: 0x00fa9a,
  mediumturquoise: 0x48d1cc,
  mediumvioletred: 0xc71585,
  midnightblue: 0x191970,
  mintcream: 0xf5fffa,
  mistyrose: 0xffe4e1,
  moccasin: 0xffe4b5,
  navajowhite: 0xffdead,
  navy: 0x000080,
  oldlace: 0xfdf5e6,
  olive: 0x808000,
  olivedrab: 0x6b8e23,
  orange: 0xffa500,
  orangered: 0xff4500,
  orchid: 0xda70d6,
  palegoldenrod: 0xeee8aa,
  palegreen: 0x98fb98,
  paleturquoise: 0xafeeee,
  palevioletred: 0xdb7093,
  papayawhip: 0xffefd5,
  peachpuff: 0xffdab9,
  peru: 0xcd853f,
  pink: 0xffc0cb,
  plum: 0xdda0dd,
  powderblue: 0xb0e0e6,
  purple: 0x800080,
  rebeccapurple: 0x663399,
  red: 0xff0000,
  rosybrown: 0xbc8f8f,
  royalblue: 0x4169e1,
  saddlebrown: 0x8b4513,
  salmon: 0xfa8072,
  sandybrown: 0xf4a460,
  seagreen: 0x2e8b57,
  seashell: 0xfff5ee,
  sienna: 0xa0522d,
  silver: 0xc0c0c0,
  skyblue: 0x87ceeb,
  slateblue: 0x6a5acd,
  slategray: 0x708090,
  slategrey: 0x708090,
  snow: 0xfffafa,
  springgreen: 0x00ff7f,
  steelblue: 0x4682b4,
  tan: 0xd2b48c,
  teal: 0x008080,
  thistle: 0xd8bfd8,
  tomato: 0xff6347,
  turquoise: 0x40e0d0,
  violet: 0xee82ee,
  wheat: 0xf5deb3,
  white: 0xffffff,
  whitesmoke: 0xf5f5f5,
  yellow: 0xffff00,
  yellowgreen: 0x9acd32
};

define(Color, color, {
  displayable: function() {
    return this.rgb().displayable();
  },
  toString: function() {
    return this.rgb() + "";
  }
});

function color(format) {
  var m;
  format = (format + "").trim().toLowerCase();
  return (m = reHex3.exec(format)) ? (m = parseInt(m[1], 16), new Rgb((m >> 8 & 0xf) | (m >> 4 & 0x0f0), (m >> 4 & 0xf) | (m & 0xf0), ((m & 0xf) << 4) | (m & 0xf), 1)) // #f00
      : (m = reHex6.exec(format)) ? rgbn(parseInt(m[1], 16)) // #ff0000
      : (m = reRgbInteger.exec(format)) ? new Rgb(m[1], m[2], m[3], 1) // rgb(255, 0, 0)
      : (m = reRgbPercent.exec(format)) ? new Rgb(m[1] * 255 / 100, m[2] * 255 / 100, m[3] * 255 / 100, 1) // rgb(100%, 0%, 0%)
      : (m = reRgbaInteger.exec(format)) ? rgba(m[1], m[2], m[3], m[4]) // rgba(255, 0, 0, 1)
      : (m = reRgbaPercent.exec(format)) ? rgba(m[1] * 255 / 100, m[2] * 255 / 100, m[3] * 255 / 100, m[4]) // rgb(100%, 0%, 0%, 1)
      : (m = reHslPercent.exec(format)) ? hsla(m[1], m[2] / 100, m[3] / 100, 1) // hsl(120, 50%, 50%)
      : (m = reHslaPercent.exec(format)) ? hsla(m[1], m[2] / 100, m[3] / 100, m[4]) // hsla(120, 50%, 50%, 1)
      : named.hasOwnProperty(format) ? rgbn(named[format])
      : format === "transparent" ? new Rgb(NaN, NaN, NaN, 0)
      : null;
}

function rgbn(n) {
  return new Rgb(n >> 16 & 0xff, n >> 8 & 0xff, n & 0xff, 1);
}

function rgba(r, g, b, a) {
  if (a <= 0) r = g = b = NaN;
  return new Rgb(r, g, b, a);
}

function rgbConvert(o) {
  if (!(o instanceof Color)) o = color(o);
  if (!o) return new Rgb;
  o = o.rgb();
  return new Rgb(o.r, o.g, o.b, o.opacity);
}

function rgb(r, g, b, opacity) {
  return arguments.length === 1 ? rgbConvert(r) : new Rgb(r, g, b, opacity == null ? 1 : opacity);
}

function Rgb(r, g, b, opacity) {
  this.r = +r;
  this.g = +g;
  this.b = +b;
  this.opacity = +opacity;
}

define(Rgb, rgb, extend(Color, {
  brighter: function(k) {
    k = k == null ? brighter : Math.pow(brighter, k);
    return new Rgb(this.r * k, this.g * k, this.b * k, this.opacity);
  },
  darker: function(k) {
    k = k == null ? darker : Math.pow(darker, k);
    return new Rgb(this.r * k, this.g * k, this.b * k, this.opacity);
  },
  rgb: function() {
    return this;
  },
  displayable: function() {
    return (0 <= this.r && this.r <= 255)
        && (0 <= this.g && this.g <= 255)
        && (0 <= this.b && this.b <= 255)
        && (0 <= this.opacity && this.opacity <= 1);
  },
  toString: function() {
    var a = this.opacity; a = isNaN(a) ? 1 : Math.max(0, Math.min(1, a));
    return (a === 1 ? "rgb(" : "rgba(")
        + Math.max(0, Math.min(255, Math.round(this.r) || 0)) + ", "
        + Math.max(0, Math.min(255, Math.round(this.g) || 0)) + ", "
        + Math.max(0, Math.min(255, Math.round(this.b) || 0))
        + (a === 1 ? ")" : ", " + a + ")");
  }
}));

function hsla(h, s, l, a) {
  if (a <= 0) h = s = l = NaN;
  else if (l <= 0 || l >= 1) h = s = NaN;
  else if (s <= 0) h = NaN;
  return new Hsl(h, s, l, a);
}

function hslConvert(o) {
  if (o instanceof Hsl) return new Hsl(o.h, o.s, o.l, o.opacity);
  if (!(o instanceof Color)) o = color(o);
  if (!o) return new Hsl;
  if (o instanceof Hsl) return o;
  o = o.rgb();
  var r = o.r / 255,
      g = o.g / 255,
      b = o.b / 255,
      min = Math.min(r, g, b),
      max = Math.max(r, g, b),
      h = NaN,
      s = max - min,
      l = (max + min) / 2;
  if (s) {
    if (r === max) h = (g - b) / s + (g < b) * 6;
    else if (g === max) h = (b - r) / s + 2;
    else h = (r - g) / s + 4;
    s /= l < 0.5 ? max + min : 2 - max - min;
    h *= 60;
  } else {
    s = l > 0 && l < 1 ? 0 : h;
  }
  return new Hsl(h, s, l, o.opacity);
}

function hsl(h, s, l, opacity) {
  return arguments.length === 1 ? hslConvert(h) : new Hsl(h, s, l, opacity == null ? 1 : opacity);
}

function Hsl(h, s, l, opacity) {
  this.h = +h;
  this.s = +s;
  this.l = +l;
  this.opacity = +opacity;
}

define(Hsl, hsl, extend(Color, {
  brighter: function(k) {
    k = k == null ? brighter : Math.pow(brighter, k);
    return new Hsl(this.h, this.s, this.l * k, this.opacity);
  },
  darker: function(k) {
    k = k == null ? darker : Math.pow(darker, k);
    return new Hsl(this.h, this.s, this.l * k, this.opacity);
  },
  rgb: function() {
    var h = this.h % 360 + (this.h < 0) * 360,
        s = isNaN(h) || isNaN(this.s) ? 0 : this.s,
        l = this.l,
        m2 = l + (l < 0.5 ? l : 1 - l) * s,
        m1 = 2 * l - m2;
    return new Rgb(
      hsl2rgb(h >= 240 ? h - 240 : h + 120, m1, m2),
      hsl2rgb(h, m1, m2),
      hsl2rgb(h < 120 ? h + 240 : h - 120, m1, m2),
      this.opacity
    );
  },
  displayable: function() {
    return (0 <= this.s && this.s <= 1 || isNaN(this.s))
        && (0 <= this.l && this.l <= 1)
        && (0 <= this.opacity && this.opacity <= 1);
  }
}));

/* From FvD 13.37, CSS Color Module Level 3 */
function hsl2rgb(h, m1, m2) {
  return (h < 60 ? m1 + (m2 - m1) * h / 60
      : h < 180 ? m2
      : h < 240 ? m1 + (m2 - m1) * (240 - h) / 60
      : m1) * 255;
}

var deg2rad = Math.PI / 180;
var rad2deg = 180 / Math.PI;

var Kn = 18;
var Xn = 0.950470;
var Yn = 1;
var Zn = 1.088830;
var t0 = 4 / 29;
var t1 = 6 / 29;
var t2 = 3 * t1 * t1;
var t3 = t1 * t1 * t1;

function labConvert(o) {
  if (o instanceof Lab) return new Lab(o.l, o.a, o.b, o.opacity);
  if (o instanceof Hcl) {
    var h = o.h * deg2rad;
    return new Lab(o.l, Math.cos(h) * o.c, Math.sin(h) * o.c, o.opacity);
  }
  if (!(o instanceof Rgb)) o = rgbConvert(o);
  var b = rgb2xyz(o.r),
      a = rgb2xyz(o.g),
      l = rgb2xyz(o.b),
      x = xyz2lab((0.4124564 * b + 0.3575761 * a + 0.1804375 * l) / Xn),
      y = xyz2lab((0.2126729 * b + 0.7151522 * a + 0.0721750 * l) / Yn),
      z = xyz2lab((0.0193339 * b + 0.1191920 * a + 0.9503041 * l) / Zn);
  return new Lab(116 * y - 16, 500 * (x - y), 200 * (y - z), o.opacity);
}

function lab(l, a, b, opacity) {
  return arguments.length === 1 ? labConvert(l) : new Lab(l, a, b, opacity == null ? 1 : opacity);
}

function Lab(l, a, b, opacity) {
  this.l = +l;
  this.a = +a;
  this.b = +b;
  this.opacity = +opacity;
}

define(Lab, lab, extend(Color, {
  brighter: function(k) {
    return new Lab(this.l + Kn * (k == null ? 1 : k), this.a, this.b, this.opacity);
  },
  darker: function(k) {
    return new Lab(this.l - Kn * (k == null ? 1 : k), this.a, this.b, this.opacity);
  },
  rgb: function() {
    var y = (this.l + 16) / 116,
        x = isNaN(this.a) ? y : y + this.a / 500,
        z = isNaN(this.b) ? y : y - this.b / 200;
    y = Yn * lab2xyz(y);
    x = Xn * lab2xyz(x);
    z = Zn * lab2xyz(z);
    return new Rgb(
      xyz2rgb( 3.2404542 * x - 1.5371385 * y - 0.4985314 * z), // D65 -> sRGB
      xyz2rgb(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
      xyz2rgb( 0.0556434 * x - 0.2040259 * y + 1.0572252 * z),
      this.opacity
    );
  }
}));

function xyz2lab(t) {
  return t > t3 ? Math.pow(t, 1 / 3) : t / t2 + t0;
}

function lab2xyz(t) {
  return t > t1 ? t * t * t : t2 * (t - t0);
}

function xyz2rgb(x) {
  return 255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
}

function rgb2xyz(x) {
  return (x /= 255) <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

function hclConvert(o) {
  if (o instanceof Hcl) return new Hcl(o.h, o.c, o.l, o.opacity);
  if (!(o instanceof Lab)) o = labConvert(o);
  var h = Math.atan2(o.b, o.a) * rad2deg;
  return new Hcl(h < 0 ? h + 360 : h, Math.sqrt(o.a * o.a + o.b * o.b), o.l, o.opacity);
}

function hcl(h, c, l, opacity) {
  return arguments.length === 1 ? hclConvert(h) : new Hcl(h, c, l, opacity == null ? 1 : opacity);
}

function Hcl(h, c, l, opacity) {
  this.h = +h;
  this.c = +c;
  this.l = +l;
  this.opacity = +opacity;
}

define(Hcl, hcl, extend(Color, {
  brighter: function(k) {
    return new Hcl(this.h, this.c, this.l + Kn * (k == null ? 1 : k), this.opacity);
  },
  darker: function(k) {
    return new Hcl(this.h, this.c, this.l - Kn * (k == null ? 1 : k), this.opacity);
  },
  rgb: function() {
    return labConvert(this).rgb();
  }
}));

var A = -0.14861;
var B = +1.78277;
var C = -0.29227;
var D = -0.90649;
var E = +1.97294;
var ED = E * D;
var EB = E * B;
var BC_DA = B * C - D * A;

function cubehelixConvert(o) {
  if (o instanceof Cubehelix) return new Cubehelix(o.h, o.s, o.l, o.opacity);
  if (!(o instanceof Rgb)) o = rgbConvert(o);
  var r = o.r / 255,
      g = o.g / 255,
      b = o.b / 255,
      l = (BC_DA * b + ED * r - EB * g) / (BC_DA + ED - EB),
      bl = b - l,
      k = (E * (g - l) - C * bl) / D,
      s = Math.sqrt(k * k + bl * bl) / (E * l * (1 - l)), // NaN if l=0 or l=1
      h = s ? Math.atan2(k, bl) * rad2deg - 120 : NaN;
  return new Cubehelix(h < 0 ? h + 360 : h, s, l, o.opacity);
}

function cubehelix(h, s, l, opacity) {
  return arguments.length === 1 ? cubehelixConvert(h) : new Cubehelix(h, s, l, opacity == null ? 1 : opacity);
}

function Cubehelix(h, s, l, opacity) {
  this.h = +h;
  this.s = +s;
  this.l = +l;
  this.opacity = +opacity;
}

define(Cubehelix, cubehelix, extend(Color, {
  brighter: function(k) {
    k = k == null ? brighter : Math.pow(brighter, k);
    return new Cubehelix(this.h, this.s, this.l * k, this.opacity);
  },
  darker: function(k) {
    k = k == null ? darker : Math.pow(darker, k);
    return new Cubehelix(this.h, this.s, this.l * k, this.opacity);
  },
  rgb: function() {
    var h = isNaN(this.h) ? 0 : (this.h + 120) * deg2rad,
        l = +this.l,
        a = isNaN(this.s) ? 0 : this.s * l * (1 - l),
        cosh = Math.cos(h),
        sinh = Math.sin(h);
    return new Rgb(
      255 * (l + a * (A * cosh + B * sinh)),
      255 * (l + a * (C * cosh + D * sinh)),
      255 * (l + a * (E * cosh)),
      this.opacity
    );
  }
}));

function basis(t1, v0, v1, v2, v3) {
  var t2 = t1 * t1, t3 = t2 * t1;
  return ((1 - 3 * t1 + 3 * t2 - t3) * v0
      + (4 - 6 * t2 + 3 * t3) * v1
      + (1 + 3 * t1 + 3 * t2 - 3 * t3) * v2
      + t3 * v3) / 6;
}

var constant$3 = function(x) {
  return function() {
    return x;
  };
};

function linear(a, d) {
  return function(t) {
    return a + t * d;
  };
}

function exponential(a, b, y) {
  return a = Math.pow(a, y), b = Math.pow(b, y) - a, y = 1 / y, function(t) {
    return Math.pow(a + t * b, y);
  };
}

function hue(a, b) {
  var d = b - a;
  return d ? linear(a, d > 180 || d < -180 ? d - 360 * Math.round(d / 360) : d) : constant$3(isNaN(a) ? b : a);
}

function gamma(y) {
  return (y = +y) === 1 ? nogamma : function(a, b) {
    return b - a ? exponential(a, b, y) : constant$3(isNaN(a) ? b : a);
  };
}

function nogamma(a, b) {
  var d = b - a;
  return d ? linear(a, d) : constant$3(isNaN(a) ? b : a);
}

var interpolateRgb = ((function rgbGamma(y) {
  var color$$1 = gamma(y);

  function rgb$$1(start, end) {
    var r = color$$1((start = rgb(start)).r, (end = rgb(end)).r),
        g = color$$1(start.g, end.g),
        b = color$$1(start.b, end.b),
        opacity = nogamma(start.opacity, end.opacity);
    return function(t) {
      start.r = r(t);
      start.g = g(t);
      start.b = b(t);
      start.opacity = opacity(t);
      return start + "";
    };
  }

  rgb$$1.gamma = rgbGamma;

  return rgb$$1;
}))(1);

var array$1 = function(a, b) {
  var nb = b ? b.length : 0,
      na = a ? Math.min(nb, a.length) : 0,
      x = new Array(nb),
      c = new Array(nb),
      i;

  for (i = 0; i < na; ++i) x[i] = interpolateValue(a[i], b[i]);
  for (; i < nb; ++i) c[i] = b[i];

  return function(t) {
    for (i = 0; i < na; ++i) c[i] = x[i](t);
    return c;
  };
};

var date = function(a, b) {
  var d = new Date;
  return a = +a, b -= a, function(t) {
    return d.setTime(a + b * t), d;
  };
};

var reinterpolate = function(a, b) {
  return a = +a, b -= a, function(t) {
    return a + b * t;
  };
};

var object = function(a, b) {
  var i = {},
      c = {},
      k;

  if (a === null || typeof a !== "object") a = {};
  if (b === null || typeof b !== "object") b = {};

  for (k in b) {
    if (k in a) {
      i[k] = interpolateValue(a[k], b[k]);
    } else {
      c[k] = b[k];
    }
  }

  return function(t) {
    for (k in i) c[k] = i[k](t);
    return c;
  };
};

var reA = /[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g;
var reB = new RegExp(reA.source, "g");

function zero(b) {
  return function() {
    return b;
  };
}

function one(b) {
  return function(t) {
    return b(t) + "";
  };
}

var interpolateString = function(a, b) {
  var bi = reA.lastIndex = reB.lastIndex = 0, // scan index for next number in b
      am, // current match in a
      bm, // current match in b
      bs, // string preceding current number in b, if any
      i = -1, // index in s
      s = [], // string constants and placeholders
      q = []; // number interpolators

  // Coerce inputs to strings.
  a = a + "", b = b + "";

  // Interpolate pairs of numbers in a & b.
  while ((am = reA.exec(a))
      && (bm = reB.exec(b))) {
    if ((bs = bm.index) > bi) { // a string precedes the next number in b
      bs = b.slice(bi, bs);
      if (s[i]) s[i] += bs; // coalesce with previous string
      else s[++i] = bs;
    }
    if ((am = am[0]) === (bm = bm[0])) { // numbers in a & b match
      if (s[i]) s[i] += bm; // coalesce with previous string
      else s[++i] = bm;
    } else { // interpolate non-matching numbers
      s[++i] = null;
      q.push({i: i, x: reinterpolate(am, bm)});
    }
    bi = reB.lastIndex;
  }

  // Add remains of b.
  if (bi < b.length) {
    bs = b.slice(bi);
    if (s[i]) s[i] += bs; // coalesce with previous string
    else s[++i] = bs;
  }

  // Special optimization for only a single match.
  // Otherwise, interpolate each of the numbers and rejoin the string.
  return s.length < 2 ? (q[0]
      ? one(q[0].x)
      : zero(b))
      : (b = q.length, function(t) {
          for (var i = 0, o; i < b; ++i) s[(o = q[i]).i] = o.x(t);
          return s.join("");
        });
};

var interpolateValue = function(a, b) {
  var t = typeof b, c;
  return b == null || t === "boolean" ? constant$3(b)
      : (t === "number" ? reinterpolate
      : t === "string" ? ((c = color(b)) ? (b = c, interpolateRgb) : interpolateString)
      : b instanceof color ? interpolateRgb
      : b instanceof Date ? date
      : Array.isArray(b) ? array$1
      : typeof b.valueOf !== "function" && typeof b.toString !== "function" || isNaN(b) ? object
      : reinterpolate)(a, b);
};

var interpolateRound = function(a, b) {
  return a = +a, b -= a, function(t) {
    return Math.round(a + b * t);
  };
};

var degrees = 180 / Math.PI;

var identity$2 = {
  translateX: 0,
  translateY: 0,
  rotate: 0,
  skewX: 0,
  scaleX: 1,
  scaleY: 1
};

var decompose = function(a, b, c, d, e, f) {
  var scaleX, scaleY, skewX;
  if (scaleX = Math.sqrt(a * a + b * b)) a /= scaleX, b /= scaleX;
  if (skewX = a * c + b * d) c -= a * skewX, d -= b * skewX;
  if (scaleY = Math.sqrt(c * c + d * d)) c /= scaleY, d /= scaleY, skewX /= scaleY;
  if (a * d < b * c) a = -a, b = -b, skewX = -skewX, scaleX = -scaleX;
  return {
    translateX: e,
    translateY: f,
    rotate: Math.atan2(b, a) * degrees,
    skewX: Math.atan(skewX) * degrees,
    scaleX: scaleX,
    scaleY: scaleY
  };
};

var cssNode;
var cssRoot;
var cssView;
var svgNode;

function parseCss(value) {
  if (value === "none") return identity$2;
  if (!cssNode) cssNode = document.createElement("DIV"), cssRoot = document.documentElement, cssView = document.defaultView;
  cssNode.style.transform = value;
  value = cssView.getComputedStyle(cssRoot.appendChild(cssNode), null).getPropertyValue("transform");
  cssRoot.removeChild(cssNode);
  value = value.slice(7, -1).split(",");
  return decompose(+value[0], +value[1], +value[2], +value[3], +value[4], +value[5]);
}

function parseSvg(value) {
  if (value == null) return identity$2;
  if (!svgNode) svgNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svgNode.setAttribute("transform", value);
  if (!(value = svgNode.transform.baseVal.consolidate())) return identity$2;
  value = value.matrix;
  return decompose(value.a, value.b, value.c, value.d, value.e, value.f);
}

function interpolateTransform(parse, pxComma, pxParen, degParen) {

  function pop(s) {
    return s.length ? s.pop() + " " : "";
  }

  function translate(xa, ya, xb, yb, s, q) {
    if (xa !== xb || ya !== yb) {
      var i = s.push("translate(", null, pxComma, null, pxParen);
      q.push({i: i - 4, x: reinterpolate(xa, xb)}, {i: i - 2, x: reinterpolate(ya, yb)});
    } else if (xb || yb) {
      s.push("translate(" + xb + pxComma + yb + pxParen);
    }
  }

  function rotate(a, b, s, q) {
    if (a !== b) {
      if (a - b > 180) b += 360; else if (b - a > 180) a += 360; // shortest path
      q.push({i: s.push(pop(s) + "rotate(", null, degParen) - 2, x: reinterpolate(a, b)});
    } else if (b) {
      s.push(pop(s) + "rotate(" + b + degParen);
    }
  }

  function skewX(a, b, s, q) {
    if (a !== b) {
      q.push({i: s.push(pop(s) + "skewX(", null, degParen) - 2, x: reinterpolate(a, b)});
    } else if (b) {
      s.push(pop(s) + "skewX(" + b + degParen);
    }
  }

  function scale(xa, ya, xb, yb, s, q) {
    if (xa !== xb || ya !== yb) {
      var i = s.push(pop(s) + "scale(", null, ",", null, ")");
      q.push({i: i - 4, x: reinterpolate(xa, xb)}, {i: i - 2, x: reinterpolate(ya, yb)});
    } else if (xb !== 1 || yb !== 1) {
      s.push(pop(s) + "scale(" + xb + "," + yb + ")");
    }
  }

  return function(a, b) {
    var s = [], // string constants and placeholders
        q = []; // number interpolators
    a = parse(a), b = parse(b);
    translate(a.translateX, a.translateY, b.translateX, b.translateY, s, q);
    rotate(a.rotate, b.rotate, s, q);
    skewX(a.skewX, b.skewX, s, q);
    scale(a.scaleX, a.scaleY, b.scaleX, b.scaleY, s, q);
    a = b = null; // gc
    return function(t) {
      var i = -1, n = q.length, o;
      while (++i < n) s[(o = q[i]).i] = o.x(t);
      return s.join("");
    };
  };
}

var interpolateTransformCss = interpolateTransform(parseCss, "px, ", "px)", "deg)");
var interpolateTransformSvg = interpolateTransform(parseSvg, ", ", ")", ")");

// p0 = [ux0, uy0, w0]
// p1 = [ux1, uy1, w1]

function cubehelix$1(hue$$1) {
  return (function cubehelixGamma(y) {
    y = +y;

    function cubehelix$$1(start, end) {
      var h = hue$$1((start = cubehelix(start)).h, (end = cubehelix(end)).h),
          s = nogamma(start.s, end.s),
          l = nogamma(start.l, end.l),
          opacity = nogamma(start.opacity, end.opacity);
      return function(t) {
        start.h = h(t);
        start.s = s(t);
        start.l = l(Math.pow(t, y));
        start.opacity = opacity(t);
        return start + "";
      };
    }

    cubehelix$$1.gamma = cubehelixGamma;

    return cubehelix$$1;
  })(1);
}

cubehelix$1(hue);
var cubehelixLong = cubehelix$1(nogamma);

var frame = 0;
var timeout = 0;
var interval = 0;
var pokeDelay = 1000;
var taskHead;
var taskTail;
var clockLast = 0;
var clockNow = 0;
var clockSkew = 0;
var clock = typeof performance === "object" && performance.now ? performance : Date;
var setFrame = typeof requestAnimationFrame === "function" ? requestAnimationFrame : function(f) { setTimeout(f, 17); };

function now() {
  return clockNow || (setFrame(clearNow), clockNow = clock.now() + clockSkew);
}

function clearNow() {
  clockNow = 0;
}

function Timer() {
  this._call =
  this._time =
  this._next = null;
}

Timer.prototype = timer.prototype = {
  constructor: Timer,
  restart: function(callback, delay, time) {
    if (typeof callback !== "function") throw new TypeError("callback is not a function");
    time = (time == null ? now() : +time) + (delay == null ? 0 : +delay);
    if (!this._next && taskTail !== this) {
      if (taskTail) taskTail._next = this;
      else taskHead = this;
      taskTail = this;
    }
    this._call = callback;
    this._time = time;
    sleep();
  },
  stop: function() {
    if (this._call) {
      this._call = null;
      this._time = Infinity;
      sleep();
    }
  }
};

function timer(callback, delay, time) {
  var t = new Timer;
  t.restart(callback, delay, time);
  return t;
}

function timerFlush() {
  now(); // Get the current time, if not already set.
  ++frame; // Pretend we’ve set an alarm, if we haven’t already.
  var t = taskHead, e;
  while (t) {
    if ((e = clockNow - t._time) >= 0) t._call.call(null, e);
    t = t._next;
  }
  --frame;
}

function wake() {
  clockNow = (clockLast = clock.now()) + clockSkew;
  frame = timeout = 0;
  try {
    timerFlush();
  } finally {
    frame = 0;
    nap();
    clockNow = 0;
  }
}

function poke() {
  var now = clock.now(), delay = now - clockLast;
  if (delay > pokeDelay) clockSkew -= delay, clockLast = now;
}

function nap() {
  var t0, t1 = taskHead, t2, time = Infinity;
  while (t1) {
    if (t1._call) {
      if (time > t1._time) time = t1._time;
      t0 = t1, t1 = t1._next;
    } else {
      t2 = t1._next, t1._next = null;
      t1 = t0 ? t0._next = t2 : taskHead = t2;
    }
  }
  taskTail = t0;
  sleep(time);
}

function sleep(time) {
  if (frame) return; // Soonest alarm already set, or will be.
  if (timeout) timeout = clearTimeout(timeout);
  var delay = time - clockNow;
  if (delay > 24) {
    if (time < Infinity) timeout = setTimeout(wake, delay);
    if (interval) interval = clearInterval(interval);
  } else {
    if (!interval) clockLast = clockNow, interval = setInterval(poke, pokeDelay);
    frame = 1, setFrame(wake);
  }
}

var timeout$1 = function(callback, delay, time) {
  var t = new Timer;
  delay = delay == null ? 0 : +delay;
  t.restart(function(elapsed) {
    t.stop();
    callback(elapsed + delay);
  }, delay, time);
  return t;
};

var emptyOn = dispatch("start", "end", "interrupt");
var emptyTween = [];

var CREATED = 0;
var SCHEDULED = 1;
var STARTING = 2;
var STARTED = 3;
var RUNNING = 4;
var ENDING = 5;
var ENDED = 6;

var schedule = function(node, name, id, index, group, timing) {
  var schedules = node.__transition;
  if (!schedules) node.__transition = {};
  else if (id in schedules) return;
  create(node, id, {
    name: name,
    index: index, // For context during callback.
    group: group, // For context during callback.
    on: emptyOn,
    tween: emptyTween,
    time: timing.time,
    delay: timing.delay,
    duration: timing.duration,
    ease: timing.ease,
    timer: null,
    state: CREATED
  });
};

function init(node, id) {
  var schedule = node.__transition;
  if (!schedule || !(schedule = schedule[id]) || schedule.state > CREATED) throw new Error("too late");
  return schedule;
}

function set$1(node, id) {
  var schedule = node.__transition;
  if (!schedule || !(schedule = schedule[id]) || schedule.state > STARTING) throw new Error("too late");
  return schedule;
}

function get$1(node, id) {
  var schedule = node.__transition;
  if (!schedule || !(schedule = schedule[id])) throw new Error("too late");
  return schedule;
}

function create(node, id, self) {
  var schedules = node.__transition,
      tween;

  // Initialize the self timer when the transition is created.
  // Note the actual delay is not known until the first callback!
  schedules[id] = self;
  self.timer = timer(schedule, 0, self.time);

  function schedule(elapsed) {
    self.state = SCHEDULED;
    self.timer.restart(start, self.delay, self.time);

    // If the elapsed delay is less than our first sleep, start immediately.
    if (self.delay <= elapsed) start(elapsed - self.delay);
  }

  function start(elapsed) {
    var i, j, n, o;

    // If the state is not SCHEDULED, then we previously errored on start.
    if (self.state !== SCHEDULED) return stop();

    for (i in schedules) {
      o = schedules[i];
      if (o.name !== self.name) continue;

      // While this element already has a starting transition during this frame,
      // defer starting an interrupting transition until that transition has a
      // chance to tick (and possibly end); see d3/d3-transition#54!
      if (o.state === STARTED) return timeout$1(start);

      // Interrupt the active transition, if any.
      // Dispatch the interrupt event.
      if (o.state === RUNNING) {
        o.state = ENDED;
        o.timer.stop();
        o.on.call("interrupt", node, node.__data__, o.index, o.group);
        delete schedules[i];
      }

      // Cancel any pre-empted transitions. No interrupt event is dispatched
      // because the cancelled transitions never started. Note that this also
      // removes this transition from the pending list!
      else if (+i < id) {
        o.state = ENDED;
        o.timer.stop();
        delete schedules[i];
      }
    }

    // Defer the first tick to end of the current frame; see d3/d3#1576.
    // Note the transition may be canceled after start and before the first tick!
    // Note this must be scheduled before the start event; see d3/d3-transition#16!
    // Assuming this is successful, subsequent callbacks go straight to tick.
    timeout$1(function() {
      if (self.state === STARTED) {
        self.state = RUNNING;
        self.timer.restart(tick, self.delay, self.time);
        tick(elapsed);
      }
    });

    // Dispatch the start event.
    // Note this must be done before the tween are initialized.
    self.state = STARTING;
    self.on.call("start", node, node.__data__, self.index, self.group);
    if (self.state !== STARTING) return; // interrupted
    self.state = STARTED;

    // Initialize the tween, deleting null tween.
    tween = new Array(n = self.tween.length);
    for (i = 0, j = -1; i < n; ++i) {
      if (o = self.tween[i].value.call(node, node.__data__, self.index, self.group)) {
        tween[++j] = o;
      }
    }
    tween.length = j + 1;
  }

  function tick(elapsed) {
    var t = elapsed < self.duration ? self.ease.call(null, elapsed / self.duration) : (self.timer.restart(stop), self.state = ENDING, 1),
        i = -1,
        n = tween.length;

    while (++i < n) {
      tween[i].call(null, t);
    }

    // Dispatch the end event.
    if (self.state === ENDING) {
      self.on.call("end", node, node.__data__, self.index, self.group);
      stop();
    }
  }

  function stop() {
    self.state = ENDED;
    self.timer.stop();
    delete schedules[id];
    for (var i in schedules) return; // eslint-disable-line no-unused-vars
    delete node.__transition;
  }
}

var interrupt = function(node, name) {
  var schedules = node.__transition,
      schedule,
      active,
      empty = true,
      i;

  if (!schedules) return;

  name = name == null ? null : name + "";

  for (i in schedules) {
    if ((schedule = schedules[i]).name !== name) { empty = false; continue; }
    active = schedule.state > STARTING && schedule.state < ENDING;
    schedule.state = ENDED;
    schedule.timer.stop();
    if (active) schedule.on.call("interrupt", node, node.__data__, schedule.index, schedule.group);
    delete schedules[i];
  }

  if (empty) delete node.__transition;
};

var selection_interrupt = function(name) {
  return this.each(function() {
    interrupt(this, name);
  });
};

function tweenRemove(id, name) {
  var tween0, tween1;
  return function() {
    var schedule = set$1(this, id),
        tween = schedule.tween;

    // If this node shared tween with the previous node,
    // just assign the updated shared tween and we’re done!
    // Otherwise, copy-on-write.
    if (tween !== tween0) {
      tween1 = tween0 = tween;
      for (var i = 0, n = tween1.length; i < n; ++i) {
        if (tween1[i].name === name) {
          tween1 = tween1.slice();
          tween1.splice(i, 1);
          break;
        }
      }
    }

    schedule.tween = tween1;
  };
}

function tweenFunction(id, name, value) {
  var tween0, tween1;
  if (typeof value !== "function") throw new Error;
  return function() {
    var schedule = set$1(this, id),
        tween = schedule.tween;

    // If this node shared tween with the previous node,
    // just assign the updated shared tween and we’re done!
    // Otherwise, copy-on-write.
    if (tween !== tween0) {
      tween1 = (tween0 = tween).slice();
      for (var t = {name: name, value: value}, i = 0, n = tween1.length; i < n; ++i) {
        if (tween1[i].name === name) {
          tween1[i] = t;
          break;
        }
      }
      if (i === n) tween1.push(t);
    }

    schedule.tween = tween1;
  };
}

var transition_tween = function(name, value) {
  var id = this._id;

  name += "";

  if (arguments.length < 2) {
    var tween = get$1(this.node(), id).tween;
    for (var i = 0, n = tween.length, t; i < n; ++i) {
      if ((t = tween[i]).name === name) {
        return t.value;
      }
    }
    return null;
  }

  return this.each((value == null ? tweenRemove : tweenFunction)(id, name, value));
};

function tweenValue(transition, name, value) {
  var id = transition._id;

  transition.each(function() {
    var schedule = set$1(this, id);
    (schedule.value || (schedule.value = {}))[name] = value.apply(this, arguments);
  });

  return function(node) {
    return get$1(node, id).value[name];
  };
}

var interpolate$$1 = function(a, b) {
  var c;
  return (typeof b === "number" ? reinterpolate
      : b instanceof color ? interpolateRgb
      : (c = color(b)) ? (b = c, interpolateRgb)
      : interpolateString)(a, b);
};

function attrRemove$1(name) {
  return function() {
    this.removeAttribute(name);
  };
}

function attrRemoveNS$1(fullname) {
  return function() {
    this.removeAttributeNS(fullname.space, fullname.local);
  };
}

function attrConstant$1(name, interpolate$$1, value1) {
  var value00,
      interpolate0;
  return function() {
    var value0 = this.getAttribute(name);
    return value0 === value1 ? null
        : value0 === value00 ? interpolate0
        : interpolate0 = interpolate$$1(value00 = value0, value1);
  };
}

function attrConstantNS$1(fullname, interpolate$$1, value1) {
  var value00,
      interpolate0;
  return function() {
    var value0 = this.getAttributeNS(fullname.space, fullname.local);
    return value0 === value1 ? null
        : value0 === value00 ? interpolate0
        : interpolate0 = interpolate$$1(value00 = value0, value1);
  };
}

function attrFunction$1(name, interpolate$$1, value) {
  var value00,
      value10,
      interpolate0;
  return function() {
    var value0, value1 = value(this);
    if (value1 == null) return void this.removeAttribute(name);
    value0 = this.getAttribute(name);
    return value0 === value1 ? null
        : value0 === value00 && value1 === value10 ? interpolate0
        : interpolate0 = interpolate$$1(value00 = value0, value10 = value1);
  };
}

function attrFunctionNS$1(fullname, interpolate$$1, value) {
  var value00,
      value10,
      interpolate0;
  return function() {
    var value0, value1 = value(this);
    if (value1 == null) return void this.removeAttributeNS(fullname.space, fullname.local);
    value0 = this.getAttributeNS(fullname.space, fullname.local);
    return value0 === value1 ? null
        : value0 === value00 && value1 === value10 ? interpolate0
        : interpolate0 = interpolate$$1(value00 = value0, value10 = value1);
  };
}

var transition_attr = function(name, value) {
  var fullname = namespace(name), i = fullname === "transform" ? interpolateTransformSvg : interpolate$$1;
  return this.attrTween(name, typeof value === "function"
      ? (fullname.local ? attrFunctionNS$1 : attrFunction$1)(fullname, i, tweenValue(this, "attr." + name, value))
      : value == null ? (fullname.local ? attrRemoveNS$1 : attrRemove$1)(fullname)
      : (fullname.local ? attrConstantNS$1 : attrConstant$1)(fullname, i, value + ""));
};

function attrTweenNS(fullname, value) {
  function tween() {
    var node = this, i = value.apply(node, arguments);
    return i && function(t) {
      node.setAttributeNS(fullname.space, fullname.local, i(t));
    };
  }
  tween._value = value;
  return tween;
}

function attrTween(name, value) {
  function tween() {
    var node = this, i = value.apply(node, arguments);
    return i && function(t) {
      node.setAttribute(name, i(t));
    };
  }
  tween._value = value;
  return tween;
}

var transition_attrTween = function(name, value) {
  var key = "attr." + name;
  if (arguments.length < 2) return (key = this.tween(key)) && key._value;
  if (value == null) return this.tween(key, null);
  if (typeof value !== "function") throw new Error;
  var fullname = namespace(name);
  return this.tween(key, (fullname.local ? attrTweenNS : attrTween)(fullname, value));
};

function delayFunction(id, value) {
  return function() {
    init(this, id).delay = +value.apply(this, arguments);
  };
}

function delayConstant(id, value) {
  return value = +value, function() {
    init(this, id).delay = value;
  };
}

var transition_delay = function(value) {
  var id = this._id;

  return arguments.length
      ? this.each((typeof value === "function"
          ? delayFunction
          : delayConstant)(id, value))
      : get$1(this.node(), id).delay;
};

function durationFunction(id, value) {
  return function() {
    set$1(this, id).duration = +value.apply(this, arguments);
  };
}

function durationConstant(id, value) {
  return value = +value, function() {
    set$1(this, id).duration = value;
  };
}

var transition_duration = function(value) {
  var id = this._id;

  return arguments.length
      ? this.each((typeof value === "function"
          ? durationFunction
          : durationConstant)(id, value))
      : get$1(this.node(), id).duration;
};

function easeConstant(id, value) {
  if (typeof value !== "function") throw new Error;
  return function() {
    set$1(this, id).ease = value;
  };
}

var transition_ease = function(value) {
  var id = this._id;

  return arguments.length
      ? this.each(easeConstant(id, value))
      : get$1(this.node(), id).ease;
};

var transition_filter = function(match) {
  if (typeof match !== "function") match = matcher$1(match);

  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, subgroup = subgroups[j] = [], node, i = 0; i < n; ++i) {
      if ((node = group[i]) && match.call(node, node.__data__, i, group)) {
        subgroup.push(node);
      }
    }
  }

  return new Transition(subgroups, this._parents, this._name, this._id);
};

var transition_merge = function(transition) {
  if (transition._id !== this._id) throw new Error;

  for (var groups0 = this._groups, groups1 = transition._groups, m0 = groups0.length, m1 = groups1.length, m = Math.min(m0, m1), merges = new Array(m0), j = 0; j < m; ++j) {
    for (var group0 = groups0[j], group1 = groups1[j], n = group0.length, merge = merges[j] = new Array(n), node, i = 0; i < n; ++i) {
      if (node = group0[i] || group1[i]) {
        merge[i] = node;
      }
    }
  }

  for (; j < m0; ++j) {
    merges[j] = groups0[j];
  }

  return new Transition(merges, this._parents, this._name, this._id);
};

function start(name) {
  return (name + "").trim().split(/^|\s+/).every(function(t) {
    var i = t.indexOf(".");
    if (i >= 0) t = t.slice(0, i);
    return !t || t === "start";
  });
}

function onFunction(id, name, listener) {
  var on0, on1, sit = start(name) ? init : set$1;
  return function() {
    var schedule = sit(this, id),
        on = schedule.on;

    // If this node shared a dispatch with the previous node,
    // just assign the updated shared dispatch and we’re done!
    // Otherwise, copy-on-write.
    if (on !== on0) (on1 = (on0 = on).copy()).on(name, listener);

    schedule.on = on1;
  };
}

var transition_on = function(name, listener) {
  var id = this._id;

  return arguments.length < 2
      ? get$1(this.node(), id).on.on(name)
      : this.each(onFunction(id, name, listener));
};

function removeFunction(id) {
  return function() {
    var parent = this.parentNode;
    for (var i in this.__transition) if (+i !== id) return;
    if (parent) parent.removeChild(this);
  };
}

var transition_remove = function() {
  return this.on("end.remove", removeFunction(this._id));
};

var transition_select = function(select$$1) {
  var name = this._name,
      id = this._id;

  if (typeof select$$1 !== "function") select$$1 = selector(select$$1);

  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, subgroup = subgroups[j] = new Array(n), node, subnode, i = 0; i < n; ++i) {
      if ((node = group[i]) && (subnode = select$$1.call(node, node.__data__, i, group))) {
        if ("__data__" in node) subnode.__data__ = node.__data__;
        subgroup[i] = subnode;
        schedule(subgroup[i], name, id, i, subgroup, get$1(node, id));
      }
    }
  }

  return new Transition(subgroups, this._parents, name, id);
};

var transition_selectAll = function(select$$1) {
  var name = this._name,
      id = this._id;

  if (typeof select$$1 !== "function") select$$1 = selectorAll(select$$1);

  for (var groups = this._groups, m = groups.length, subgroups = [], parents = [], j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        for (var children = select$$1.call(node, node.__data__, i, group), child, inherit = get$1(node, id), k = 0, l = children.length; k < l; ++k) {
          if (child = children[k]) {
            schedule(child, name, id, k, children, inherit);
          }
        }
        subgroups.push(children);
        parents.push(node);
      }
    }
  }

  return new Transition(subgroups, parents, name, id);
};

var Selection$1 = selection.prototype.constructor;

var transition_selection = function() {
  return new Selection$1(this._groups, this._parents);
};

function styleRemove$1(name, interpolate$$2) {
  var value00,
      value10,
      interpolate0;
  return function() {
    var value0 = styleValue(this, name),
        value1 = (this.style.removeProperty(name), styleValue(this, name));
    return value0 === value1 ? null
        : value0 === value00 && value1 === value10 ? interpolate0
        : interpolate0 = interpolate$$2(value00 = value0, value10 = value1);
  };
}

function styleRemoveEnd(name) {
  return function() {
    this.style.removeProperty(name);
  };
}

function styleConstant$1(name, interpolate$$2, value1) {
  var value00,
      interpolate0;
  return function() {
    var value0 = styleValue(this, name);
    return value0 === value1 ? null
        : value0 === value00 ? interpolate0
        : interpolate0 = interpolate$$2(value00 = value0, value1);
  };
}

function styleFunction$1(name, interpolate$$2, value) {
  var value00,
      value10,
      interpolate0;
  return function() {
    var value0 = styleValue(this, name),
        value1 = value(this);
    if (value1 == null) value1 = (this.style.removeProperty(name), styleValue(this, name));
    return value0 === value1 ? null
        : value0 === value00 && value1 === value10 ? interpolate0
        : interpolate0 = interpolate$$2(value00 = value0, value10 = value1);
  };
}

var transition_style = function(name, value, priority) {
  var i = (name += "") === "transform" ? interpolateTransformCss : interpolate$$1;
  return value == null ? this
          .styleTween(name, styleRemove$1(name, i))
          .on("end.style." + name, styleRemoveEnd(name))
      : this.styleTween(name, typeof value === "function"
          ? styleFunction$1(name, i, tweenValue(this, "style." + name, value))
          : styleConstant$1(name, i, value + ""), priority);
};

function styleTween(name, value, priority) {
  function tween() {
    var node = this, i = value.apply(node, arguments);
    return i && function(t) {
      node.style.setProperty(name, i(t), priority);
    };
  }
  tween._value = value;
  return tween;
}

var transition_styleTween = function(name, value, priority) {
  var key = "style." + (name += "");
  if (arguments.length < 2) return (key = this.tween(key)) && key._value;
  if (value == null) return this.tween(key, null);
  if (typeof value !== "function") throw new Error;
  return this.tween(key, styleTween(name, value, priority == null ? "" : priority));
};

function textConstant$1(value) {
  return function() {
    this.textContent = value;
  };
}

function textFunction$1(value) {
  return function() {
    var value1 = value(this);
    this.textContent = value1 == null ? "" : value1;
  };
}

var transition_text = function(value) {
  return this.tween("text", typeof value === "function"
      ? textFunction$1(tweenValue(this, "text", value))
      : textConstant$1(value == null ? "" : value + ""));
};

var transition_transition = function() {
  var name = this._name,
      id0 = this._id,
      id1 = newId();

  for (var groups = this._groups, m = groups.length, j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        var inherit = get$1(node, id0);
        schedule(node, name, id1, i, group, {
          time: inherit.time + inherit.delay + inherit.duration,
          delay: 0,
          duration: inherit.duration,
          ease: inherit.ease
        });
      }
    }
  }

  return new Transition(groups, this._parents, name, id1);
};

var id = 0;

function Transition(groups, parents, name, id) {
  this._groups = groups;
  this._parents = parents;
  this._name = name;
  this._id = id;
}

function transition(name) {
  return selection().transition(name);
}

function newId() {
  return ++id;
}

var selection_prototype = selection.prototype;

Transition.prototype = transition.prototype = {
  constructor: Transition,
  select: transition_select,
  selectAll: transition_selectAll,
  filter: transition_filter,
  merge: transition_merge,
  selection: transition_selection,
  transition: transition_transition,
  call: selection_prototype.call,
  nodes: selection_prototype.nodes,
  node: selection_prototype.node,
  size: selection_prototype.size,
  empty: selection_prototype.empty,
  each: selection_prototype.each,
  on: transition_on,
  attr: transition_attr,
  attrTween: transition_attrTween,
  style: transition_style,
  styleTween: transition_styleTween,
  text: transition_text,
  remove: transition_remove,
  tween: transition_tween,
  delay: transition_delay,
  duration: transition_duration,
  ease: transition_ease
};

function cubicInOut(t) {
  return ((t *= 2) <= 1 ? t * t * t : (t -= 2) * t * t + 2) / 2;
}

var exponent = 3;

var polyIn = (function custom(e) {
  e = +e;

  function polyIn(t) {
    return Math.pow(t, e);
  }

  polyIn.exponent = custom;

  return polyIn;
})(exponent);

var polyOut = (function custom(e) {
  e = +e;

  function polyOut(t) {
    return 1 - Math.pow(1 - t, e);
  }

  polyOut.exponent = custom;

  return polyOut;
})(exponent);

var polyInOut = (function custom(e) {
  e = +e;

  function polyInOut(t) {
    return ((t *= 2) <= 1 ? Math.pow(t, e) : 2 - Math.pow(2 - t, e)) / 2;
  }

  polyInOut.exponent = custom;

  return polyInOut;
})(exponent);

var overshoot = 1.70158;

var backIn = (function custom(s) {
  s = +s;

  function backIn(t) {
    return t * t * ((s + 1) * t - s);
  }

  backIn.overshoot = custom;

  return backIn;
})(overshoot);

var backOut = (function custom(s) {
  s = +s;

  function backOut(t) {
    return --t * t * ((s + 1) * t + s) + 1;
  }

  backOut.overshoot = custom;

  return backOut;
})(overshoot);

var backInOut = (function custom(s) {
  s = +s;

  function backInOut(t) {
    return ((t *= 2) < 1 ? t * t * ((s + 1) * t - s) : (t -= 2) * t * ((s + 1) * t + s) + 2) / 2;
  }

  backInOut.overshoot = custom;

  return backInOut;
})(overshoot);

var tau = 2 * Math.PI;
var amplitude = 1;
var period = 0.3;

var elasticIn = (function custom(a, p) {
  var s = Math.asin(1 / (a = Math.max(1, a))) * (p /= tau);

  function elasticIn(t) {
    return a * Math.pow(2, 10 * --t) * Math.sin((s - t) / p);
  }

  elasticIn.amplitude = function(a) { return custom(a, p * tau); };
  elasticIn.period = function(p) { return custom(a, p); };

  return elasticIn;
})(amplitude, period);

var elasticOut = (function custom(a, p) {
  var s = Math.asin(1 / (a = Math.max(1, a))) * (p /= tau);

  function elasticOut(t) {
    return 1 - a * Math.pow(2, -10 * (t = +t)) * Math.sin((t + s) / p);
  }

  elasticOut.amplitude = function(a) { return custom(a, p * tau); };
  elasticOut.period = function(p) { return custom(a, p); };

  return elasticOut;
})(amplitude, period);

var elasticInOut = (function custom(a, p) {
  var s = Math.asin(1 / (a = Math.max(1, a))) * (p /= tau);

  function elasticInOut(t) {
    return ((t = t * 2 - 1) < 0
        ? a * Math.pow(2, 10 * t) * Math.sin((s - t) / p)
        : 2 - a * Math.pow(2, -10 * t) * Math.sin((s + t) / p)) / 2;
  }

  elasticInOut.amplitude = function(a) { return custom(a, p * tau); };
  elasticInOut.period = function(p) { return custom(a, p); };

  return elasticInOut;
})(amplitude, period);

var defaultTiming = {
  time: null, // Set on use.
  delay: 0,
  duration: 250,
  ease: cubicInOut
};

function inherit(node, id) {
  var timing;
  while (!(timing = node.__transition) || !(timing = timing[id])) {
    if (!(node = node.parentNode)) {
      return defaultTiming.time = now(), defaultTiming;
    }
  }
  return timing;
}

var selection_transition = function(name) {
  var id,
      timing;

  if (name instanceof Transition) {
    id = name._id, name = name._name;
  } else {
    id = newId(), (timing = defaultTiming).time = now(), name = name == null ? null : name + "";
  }

  for (var groups = this._groups, m = groups.length, j = 0; j < m; ++j) {
    for (var group = groups[j], n = group.length, node, i = 0; i < n; ++i) {
      if (node = group[i]) {
        schedule(node, name, id, i, group, timing || inherit(node, id));
      }
    }
  }

  return new Transition(groups, this._parents, name, id);
};

selection.prototype.interrupt = selection_interrupt;
selection.prototype.transition = selection_transition;

var X = {
  name: "x",
  handles: ["e", "w"].map(type),
  input: function(x, e) { return x && [[x[0], e[0][1]], [x[1], e[1][1]]]; },
  output: function(xy) { return xy && [xy[0][0], xy[1][0]]; }
};

var Y = {
  name: "y",
  handles: ["n", "s"].map(type),
  input: function(y, e) { return y && [[e[0][0], y[0]], [e[1][0], y[1]]]; },
  output: function(xy) { return xy && [xy[0][1], xy[1][1]]; }
};

var XY = {
  name: "xy",
  handles: ["n", "e", "s", "w", "nw", "ne", "se", "sw"].map(type),
  input: function(xy) { return xy; },
  output: function(xy) { return xy; }
};

function type(t) {
  return {type: t};
}

var pi$1 = Math.PI;

var tau$1 = pi$1 * 2;
var max$1 = Math.max;

var pi$2 = Math.PI;
var tau$2 = 2 * pi$2;
var epsilon$1 = 1e-6;
var tauEpsilon = tau$2 - epsilon$1;

function Path() {
  this._x0 = this._y0 = // start of current subpath
  this._x1 = this._y1 = null; // end of current subpath
  this._ = "";
}

function path() {
  return new Path;
}

Path.prototype = path.prototype = {
  constructor: Path,
  moveTo: function(x, y) {
    this._ += "M" + (this._x0 = this._x1 = +x) + "," + (this._y0 = this._y1 = +y);
  },
  closePath: function() {
    if (this._x1 !== null) {
      this._x1 = this._x0, this._y1 = this._y0;
      this._ += "Z";
    }
  },
  lineTo: function(x, y) {
    this._ += "L" + (this._x1 = +x) + "," + (this._y1 = +y);
  },
  quadraticCurveTo: function(x1, y1, x, y) {
    this._ += "Q" + (+x1) + "," + (+y1) + "," + (this._x1 = +x) + "," + (this._y1 = +y);
  },
  bezierCurveTo: function(x1, y1, x2, y2, x, y) {
    this._ += "C" + (+x1) + "," + (+y1) + "," + (+x2) + "," + (+y2) + "," + (this._x1 = +x) + "," + (this._y1 = +y);
  },
  arcTo: function(x1, y1, x2, y2, r) {
    x1 = +x1, y1 = +y1, x2 = +x2, y2 = +y2, r = +r;
    var x0 = this._x1,
        y0 = this._y1,
        x21 = x2 - x1,
        y21 = y2 - y1,
        x01 = x0 - x1,
        y01 = y0 - y1,
        l01_2 = x01 * x01 + y01 * y01;

    // Is the radius negative? Error.
    if (r < 0) throw new Error("negative radius: " + r);

    // Is this path empty? Move to (x1,y1).
    if (this._x1 === null) {
      this._ += "M" + (this._x1 = x1) + "," + (this._y1 = y1);
    }

    // Or, is (x1,y1) coincident with (x0,y0)? Do nothing.
    else if (!(l01_2 > epsilon$1)) {}

    // Or, are (x0,y0), (x1,y1) and (x2,y2) collinear?
    // Equivalently, is (x1,y1) coincident with (x2,y2)?
    // Or, is the radius zero? Line to (x1,y1).
    else if (!(Math.abs(y01 * x21 - y21 * x01) > epsilon$1) || !r) {
      this._ += "L" + (this._x1 = x1) + "," + (this._y1 = y1);
    }

    // Otherwise, draw an arc!
    else {
      var x20 = x2 - x0,
          y20 = y2 - y0,
          l21_2 = x21 * x21 + y21 * y21,
          l20_2 = x20 * x20 + y20 * y20,
          l21 = Math.sqrt(l21_2),
          l01 = Math.sqrt(l01_2),
          l = r * Math.tan((pi$2 - Math.acos((l21_2 + l01_2 - l20_2) / (2 * l21 * l01))) / 2),
          t01 = l / l01,
          t21 = l / l21;

      // If the start tangent is not coincident with (x0,y0), line to.
      if (Math.abs(t01 - 1) > epsilon$1) {
        this._ += "L" + (x1 + t01 * x01) + "," + (y1 + t01 * y01);
      }

      this._ += "A" + r + "," + r + ",0,0," + (+(y01 * x20 > x01 * y20)) + "," + (this._x1 = x1 + t21 * x21) + "," + (this._y1 = y1 + t21 * y21);
    }
  },
  arc: function(x, y, r, a0, a1, ccw) {
    x = +x, y = +y, r = +r;
    var dx = r * Math.cos(a0),
        dy = r * Math.sin(a0),
        x0 = x + dx,
        y0 = y + dy,
        cw = 1 ^ ccw,
        da = ccw ? a0 - a1 : a1 - a0;

    // Is the radius negative? Error.
    if (r < 0) throw new Error("negative radius: " + r);

    // Is this path empty? Move to (x0,y0).
    if (this._x1 === null) {
      this._ += "M" + x0 + "," + y0;
    }

    // Or, is (x0,y0) not coincident with the previous point? Line to (x0,y0).
    else if (Math.abs(this._x1 - x0) > epsilon$1 || Math.abs(this._y1 - y0) > epsilon$1) {
      this._ += "L" + x0 + "," + y0;
    }

    // Is this arc empty? We’re done.
    if (!r) return;

    // Does the angle go the wrong way? Flip the direction.
    if (da < 0) da = da % tau$2 + tau$2;

    // Is this a complete circle? Draw two arcs to complete the circle.
    if (da > tauEpsilon) {
      this._ += "A" + r + "," + r + ",0,1," + cw + "," + (x - dx) + "," + (y - dy) + "A" + r + "," + r + ",0,1," + cw + "," + (this._x1 = x0) + "," + (this._y1 = y0);
    }

    // Is this arc non-empty? Draw an arc!
    else if (da > epsilon$1) {
      this._ += "A" + r + "," + r + ",0," + (+(da >= pi$2)) + "," + cw + "," + (this._x1 = x + r * Math.cos(a1)) + "," + (this._y1 = y + r * Math.sin(a1));
    }
  },
  rect: function(x, y, w, h) {
    this._ += "M" + (this._x0 = this._x1 = +x) + "," + (this._y0 = this._y1 = +y) + "h" + (+w) + "v" + (+h) + "h" + (-w) + "Z";
  },
  toString: function() {
    return this._;
  }
};

var prefix = "$";

function Map() {}

Map.prototype = map$1.prototype = {
  constructor: Map,
  has: function(key) {
    return (prefix + key) in this;
  },
  get: function(key) {
    return this[prefix + key];
  },
  set: function(key, value) {
    this[prefix + key] = value;
    return this;
  },
  remove: function(key) {
    var property = prefix + key;
    return property in this && delete this[property];
  },
  clear: function() {
    for (var property in this) if (property[0] === prefix) delete this[property];
  },
  keys: function() {
    var keys = [];
    for (var property in this) if (property[0] === prefix) keys.push(property.slice(1));
    return keys;
  },
  values: function() {
    var values = [];
    for (var property in this) if (property[0] === prefix) values.push(this[property]);
    return values;
  },
  entries: function() {
    var entries = [];
    for (var property in this) if (property[0] === prefix) entries.push({key: property.slice(1), value: this[property]});
    return entries;
  },
  size: function() {
    var size = 0;
    for (var property in this) if (property[0] === prefix) ++size;
    return size;
  },
  empty: function() {
    for (var property in this) if (property[0] === prefix) return false;
    return true;
  },
  each: function(f) {
    for (var property in this) if (property[0] === prefix) f(this[property], property.slice(1), this);
  }
};

function map$1(object, f) {
  var map = new Map;

  // Copy constructor.
  if (object instanceof Map) object.each(function(value, key) { map.set(key, value); });

  // Index array by numeric index or specified key function.
  else if (Array.isArray(object)) {
    var i = -1,
        n = object.length,
        o;

    if (f == null) while (++i < n) map.set(i, object[i]);
    else while (++i < n) map.set(f(o = object[i], i, object), o);
  }

  // Convert object to map.
  else if (object) for (var key in object) map.set(key, object[key]);

  return map;
}

function Set() {}

var proto = map$1.prototype;

Set.prototype = set$2.prototype = {
  constructor: Set,
  has: proto.has,
  add: function(value) {
    value += "";
    this[prefix + value] = value;
    return this;
  },
  remove: proto.remove,
  clear: proto.clear,
  values: proto.keys,
  size: proto.size,
  empty: proto.empty,
  each: proto.each
};

function set$2(object, f) {
  var set = new Set;

  // Copy constructor.
  if (object instanceof Set) object.each(function(value) { set.add(value); });

  // Otherwise, assume it’s an array.
  else if (object) {
    var i = -1, n = object.length;
    if (f == null) while (++i < n) set.add(object[i]);
    else while (++i < n) set.add(f(object[i], i, object));
  }

  return set;
}

function objectConverter(columns) {
  return new Function("d", "return {" + columns.map(function(name, i) {
    return JSON.stringify(name) + ": d[" + i + "]";
  }).join(",") + "}");
}

function customConverter(columns, f) {
  var object = objectConverter(columns);
  return function(row, i) {
    return f(object(row), i, columns);
  };
}

// Compute unique columns in order of discovery.
function inferColumns(rows) {
  var columnSet = Object.create(null),
      columns = [];

  rows.forEach(function(row) {
    for (var column in row) {
      if (!(column in columnSet)) {
        columns.push(columnSet[column] = column);
      }
    }
  });

  return columns;
}

var dsv = function(delimiter) {
  var reFormat = new RegExp("[\"" + delimiter + "\n\r]"),
      delimiterCode = delimiter.charCodeAt(0);

  function parse(text, f) {
    var convert, columns, rows = parseRows(text, function(row, i) {
      if (convert) return convert(row, i - 1);
      columns = row, convert = f ? customConverter(row, f) : objectConverter(row);
    });
    rows.columns = columns;
    return rows;
  }

  function parseRows(text, f) {
    var EOL = {}, // sentinel value for end-of-line
        EOF = {}, // sentinel value for end-of-file
        rows = [], // output rows
        N = text.length,
        I = 0, // current character index
        n = 0, // the current line number
        t, // the current token
        eol; // is the current token followed by EOL?

    function token() {
      if (I >= N) return EOF; // special case: end of file
      if (eol) return eol = false, EOL; // special case: end of line

      // special case: quotes
      var j = I, c;
      if (text.charCodeAt(j) === 34) {
        var i = j;
        while (i++ < N) {
          if (text.charCodeAt(i) === 34) {
            if (text.charCodeAt(i + 1) !== 34) break;
            ++i;
          }
        }
        I = i + 2;
        c = text.charCodeAt(i + 1);
        if (c === 13) {
          eol = true;
          if (text.charCodeAt(i + 2) === 10) ++I;
        } else if (c === 10) {
          eol = true;
        }
        return text.slice(j + 1, i).replace(/""/g, "\"");
      }

      // common case: find next delimiter or newline
      while (I < N) {
        var k = 1;
        c = text.charCodeAt(I++);
        if (c === 10) eol = true; // \n
        else if (c === 13) { eol = true; if (text.charCodeAt(I) === 10) ++I, ++k; } // \r|\r\n
        else if (c !== delimiterCode) continue;
        return text.slice(j, I - k);
      }

      // special case: last token before EOF
      return text.slice(j);
    }

    while ((t = token()) !== EOF) {
      var a = [];
      while (t !== EOL && t !== EOF) {
        a.push(t);
        t = token();
      }
      if (f && (a = f(a, n++)) == null) continue;
      rows.push(a);
    }

    return rows;
  }

  function format(rows, columns) {
    if (columns == null) columns = inferColumns(rows);
    return [columns.map(formatValue).join(delimiter)].concat(rows.map(function(row) {
      return columns.map(function(column) {
        return formatValue(row[column]);
      }).join(delimiter);
    })).join("\n");
  }

  function formatRows(rows) {
    return rows.map(formatRow).join("\n");
  }

  function formatRow(row) {
    return row.map(formatValue).join(delimiter);
  }

  function formatValue(text) {
    return text == null ? ""
        : reFormat.test(text += "") ? "\"" + text.replace(/\"/g, "\"\"") + "\""
        : text;
  }

  return {
    parse: parse,
    parseRows: parseRows,
    format: format,
    formatRows: formatRows
  };
};

var csv = dsv(",");

var csvParse = csv.parse;
var csvParseRows = csv.parseRows;

var tsv = dsv("\t");

var tsvParse = tsv.parse;

var tree_add = function(d) {
  var x = +this._x.call(null, d),
      y = +this._y.call(null, d);
  return add(this.cover(x, y), x, y, d);
};

function add(tree, x, y, d) {
  if (isNaN(x) || isNaN(y)) return tree; // ignore invalid points

  var parent,
      node = tree._root,
      leaf = {data: d},
      x0 = tree._x0,
      y0 = tree._y0,
      x1 = tree._x1,
      y1 = tree._y1,
      xm,
      ym,
      xp,
      yp,
      right,
      bottom,
      i,
      j;

  // If the tree is empty, initialize the root as a leaf.
  if (!node) return tree._root = leaf, tree;

  // Find the existing leaf for the new point, or add it.
  while (node.length) {
    if (right = x >= (xm = (x0 + x1) / 2)) x0 = xm; else x1 = xm;
    if (bottom = y >= (ym = (y0 + y1) / 2)) y0 = ym; else y1 = ym;
    if (parent = node, !(node = node[i = bottom << 1 | right])) return parent[i] = leaf, tree;
  }

  // Is the new point is exactly coincident with the existing point?
  xp = +tree._x.call(null, node.data);
  yp = +tree._y.call(null, node.data);
  if (x === xp && y === yp) return leaf.next = node, parent ? parent[i] = leaf : tree._root = leaf, tree;

  // Otherwise, split the leaf node until the old and new point are separated.
  do {
    parent = parent ? parent[i] = new Array(4) : tree._root = new Array(4);
    if (right = x >= (xm = (x0 + x1) / 2)) x0 = xm; else x1 = xm;
    if (bottom = y >= (ym = (y0 + y1) / 2)) y0 = ym; else y1 = ym;
  } while ((i = bottom << 1 | right) === (j = (yp >= ym) << 1 | (xp >= xm)));
  return parent[j] = node, parent[i] = leaf, tree;
}

function addAll(data) {
  var d, i, n = data.length,
      x,
      y,
      xz = new Array(n),
      yz = new Array(n),
      x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;

  // Compute the points and their extent.
  for (i = 0; i < n; ++i) {
    if (isNaN(x = +this._x.call(null, d = data[i])) || isNaN(y = +this._y.call(null, d))) continue;
    xz[i] = x;
    yz[i] = y;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }

  // If there were no (valid) points, inherit the existing extent.
  if (x1 < x0) x0 = this._x0, x1 = this._x1;
  if (y1 < y0) y0 = this._y0, y1 = this._y1;

  // Expand the tree to cover the new points.
  this.cover(x0, y0).cover(x1, y1);

  // Add the new points.
  for (i = 0; i < n; ++i) {
    add(this, xz[i], yz[i], data[i]);
  }

  return this;
}

var tree_cover = function(x, y) {
  if (isNaN(x = +x) || isNaN(y = +y)) return this; // ignore invalid points

  var x0 = this._x0,
      y0 = this._y0,
      x1 = this._x1,
      y1 = this._y1;

  // If the quadtree has no extent, initialize them.
  // Integer extent are necessary so that if we later double the extent,
  // the existing quadrant boundaries don’t change due to floating point error!
  if (isNaN(x0)) {
    x1 = (x0 = Math.floor(x)) + 1;
    y1 = (y0 = Math.floor(y)) + 1;
  }

  // Otherwise, double repeatedly to cover.
  else if (x0 > x || x > x1 || y0 > y || y > y1) {
    var z = x1 - x0,
        node = this._root,
        parent,
        i;

    switch (i = (y < (y0 + y1) / 2) << 1 | (x < (x0 + x1) / 2)) {
      case 0: {
        do parent = new Array(4), parent[i] = node, node = parent;
        while (z *= 2, x1 = x0 + z, y1 = y0 + z, x > x1 || y > y1);
        break;
      }
      case 1: {
        do parent = new Array(4), parent[i] = node, node = parent;
        while (z *= 2, x0 = x1 - z, y1 = y0 + z, x0 > x || y > y1);
        break;
      }
      case 2: {
        do parent = new Array(4), parent[i] = node, node = parent;
        while (z *= 2, x1 = x0 + z, y0 = y1 - z, x > x1 || y0 > y);
        break;
      }
      case 3: {
        do parent = new Array(4), parent[i] = node, node = parent;
        while (z *= 2, x0 = x1 - z, y0 = y1 - z, x0 > x || y0 > y);
        break;
      }
    }

    if (this._root && this._root.length) this._root = node;
  }

  // If the quadtree covers the point already, just return.
  else return this;

  this._x0 = x0;
  this._y0 = y0;
  this._x1 = x1;
  this._y1 = y1;
  return this;
};

var tree_data = function() {
  var data = [];
  this.visit(function(node) {
    if (!node.length) do data.push(node.data); while (node = node.next)
  });
  return data;
};

var tree_extent = function(_) {
  return arguments.length
      ? this.cover(+_[0][0], +_[0][1]).cover(+_[1][0], +_[1][1])
      : isNaN(this._x0) ? undefined : [[this._x0, this._y0], [this._x1, this._y1]];
};

var Quad = function(node, x0, y0, x1, y1) {
  this.node = node;
  this.x0 = x0;
  this.y0 = y0;
  this.x1 = x1;
  this.y1 = y1;
};

var tree_find = function(x, y, radius) {
  var data,
      x0 = this._x0,
      y0 = this._y0,
      x1,
      y1,
      x2,
      y2,
      x3 = this._x1,
      y3 = this._y1,
      quads = [],
      node = this._root,
      q,
      i;

  if (node) quads.push(new Quad(node, x0, y0, x3, y3));
  if (radius == null) radius = Infinity;
  else {
    x0 = x - radius, y0 = y - radius;
    x3 = x + radius, y3 = y + radius;
    radius *= radius;
  }

  while (q = quads.pop()) {

    // Stop searching if this quadrant can’t contain a closer node.
    if (!(node = q.node)
        || (x1 = q.x0) > x3
        || (y1 = q.y0) > y3
        || (x2 = q.x1) < x0
        || (y2 = q.y1) < y0) continue;

    // Bisect the current quadrant.
    if (node.length) {
      var xm = (x1 + x2) / 2,
          ym = (y1 + y2) / 2;

      quads.push(
        new Quad(node[3], xm, ym, x2, y2),
        new Quad(node[2], x1, ym, xm, y2),
        new Quad(node[1], xm, y1, x2, ym),
        new Quad(node[0], x1, y1, xm, ym)
      );

      // Visit the closest quadrant first.
      if (i = (y >= ym) << 1 | (x >= xm)) {
        q = quads[quads.length - 1];
        quads[quads.length - 1] = quads[quads.length - 1 - i];
        quads[quads.length - 1 - i] = q;
      }
    }

    // Visit this point. (Visiting coincident points isn’t necessary!)
    else {
      var dx = x - +this._x.call(null, node.data),
          dy = y - +this._y.call(null, node.data),
          d2 = dx * dx + dy * dy;
      if (d2 < radius) {
        var d = Math.sqrt(radius = d2);
        x0 = x - d, y0 = y - d;
        x3 = x + d, y3 = y + d;
        data = node.data;
      }
    }
  }

  return data;
};

var tree_remove = function(d) {
  if (isNaN(x = +this._x.call(null, d)) || isNaN(y = +this._y.call(null, d))) return this; // ignore invalid points

  var parent,
      node = this._root,
      retainer,
      previous,
      next,
      x0 = this._x0,
      y0 = this._y0,
      x1 = this._x1,
      y1 = this._y1,
      x,
      y,
      xm,
      ym,
      right,
      bottom,
      i,
      j;

  // If the tree is empty, initialize the root as a leaf.
  if (!node) return this;

  // Find the leaf node for the point.
  // While descending, also retain the deepest parent with a non-removed sibling.
  if (node.length) while (true) {
    if (right = x >= (xm = (x0 + x1) / 2)) x0 = xm; else x1 = xm;
    if (bottom = y >= (ym = (y0 + y1) / 2)) y0 = ym; else y1 = ym;
    if (!(parent = node, node = node[i = bottom << 1 | right])) return this;
    if (!node.length) break;
    if (parent[(i + 1) & 3] || parent[(i + 2) & 3] || parent[(i + 3) & 3]) retainer = parent, j = i;
  }

  // Find the point to remove.
  while (node.data !== d) if (!(previous = node, node = node.next)) return this;
  if (next = node.next) delete node.next;

  // If there are multiple coincident points, remove just the point.
  if (previous) return (next ? previous.next = next : delete previous.next), this;

  // If this is the root point, remove it.
  if (!parent) return this._root = next, this;

  // Remove this leaf.
  next ? parent[i] = next : delete parent[i];

  // If the parent now contains exactly one leaf, collapse superfluous parents.
  if ((node = parent[0] || parent[1] || parent[2] || parent[3])
      && node === (parent[3] || parent[2] || parent[1] || parent[0])
      && !node.length) {
    if (retainer) retainer[j] = node;
    else this._root = node;
  }

  return this;
};

function removeAll(data) {
  for (var i = 0, n = data.length; i < n; ++i) this.remove(data[i]);
  return this;
}

var tree_root = function() {
  return this._root;
};

var tree_size = function() {
  var size = 0;
  this.visit(function(node) {
    if (!node.length) do ++size; while (node = node.next)
  });
  return size;
};

var tree_visit = function(callback) {
  var quads = [], q, node = this._root, child, x0, y0, x1, y1;
  if (node) quads.push(new Quad(node, this._x0, this._y0, this._x1, this._y1));
  while (q = quads.pop()) {
    if (!callback(node = q.node, x0 = q.x0, y0 = q.y0, x1 = q.x1, y1 = q.y1) && node.length) {
      var xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
      if (child = node[3]) quads.push(new Quad(child, xm, ym, x1, y1));
      if (child = node[2]) quads.push(new Quad(child, x0, ym, xm, y1));
      if (child = node[1]) quads.push(new Quad(child, xm, y0, x1, ym));
      if (child = node[0]) quads.push(new Quad(child, x0, y0, xm, ym));
    }
  }
  return this;
};

var tree_visitAfter = function(callback) {
  var quads = [], next = [], q;
  if (this._root) quads.push(new Quad(this._root, this._x0, this._y0, this._x1, this._y1));
  while (q = quads.pop()) {
    var node = q.node;
    if (node.length) {
      var child, x0 = q.x0, y0 = q.y0, x1 = q.x1, y1 = q.y1, xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
      if (child = node[0]) quads.push(new Quad(child, x0, y0, xm, ym));
      if (child = node[1]) quads.push(new Quad(child, xm, y0, x1, ym));
      if (child = node[2]) quads.push(new Quad(child, x0, ym, xm, y1));
      if (child = node[3]) quads.push(new Quad(child, xm, ym, x1, y1));
    }
    next.push(q);
  }
  while (q = next.pop()) {
    callback(q.node, q.x0, q.y0, q.x1, q.y1);
  }
  return this;
};

function defaultX(d) {
  return d[0];
}

var tree_x = function(_) {
  return arguments.length ? (this._x = _, this) : this._x;
};

function defaultY(d) {
  return d[1];
}

var tree_y = function(_) {
  return arguments.length ? (this._y = _, this) : this._y;
};

function quadtree(nodes, x, y) {
  var tree = new Quadtree(x == null ? defaultX : x, y == null ? defaultY : y, NaN, NaN, NaN, NaN);
  return nodes == null ? tree : tree.addAll(nodes);
}

function Quadtree(x, y, x0, y0, x1, y1) {
  this._x = x;
  this._y = y;
  this._x0 = x0;
  this._y0 = y0;
  this._x1 = x1;
  this._y1 = y1;
  this._root = undefined;
}

function leaf_copy(leaf) {
  var copy = {data: leaf.data}, next = copy;
  while (leaf = leaf.next) next = next.next = {data: leaf.data};
  return copy;
}

var treeProto = quadtree.prototype = Quadtree.prototype;

treeProto.copy = function() {
  var copy = new Quadtree(this._x, this._y, this._x0, this._y0, this._x1, this._y1),
      node = this._root,
      nodes,
      child;

  if (!node) return copy;

  if (!node.length) return copy._root = leaf_copy(node), copy;

  nodes = [{source: node, target: copy._root = new Array(4)}];
  while (node = nodes.pop()) {
    for (var i = 0; i < 4; ++i) {
      if (child = node.source[i]) {
        if (child.length) nodes.push({source: child, target: node.target[i] = new Array(4)});
        else node.target[i] = leaf_copy(child);
      }
    }
  }

  return copy;
};

treeProto.add = tree_add;
treeProto.addAll = addAll;
treeProto.cover = tree_cover;
treeProto.data = tree_data;
treeProto.extent = tree_extent;
treeProto.find = tree_find;
treeProto.remove = tree_remove;
treeProto.removeAll = removeAll;
treeProto.root = tree_root;
treeProto.size = tree_size;
treeProto.visit = tree_visit;
treeProto.visitAfter = tree_visitAfter;
treeProto.x = tree_x;
treeProto.y = tree_y;

// Computes the decimal coefficient and exponent of the specified number x with
// significant digits p, where x is positive and p is in [1, 21] or undefined.
// For example, formatDecimal(1.23) returns ["123", 0].
var formatDecimal = function(x, p) {
  if ((i = (x = p ? x.toExponential(p - 1) : x.toExponential()).indexOf("e")) < 0) return null; // NaN, ±Infinity
  var i, coefficient = x.slice(0, i);

  // The string returned by toExponential either has the form \d\.\d+e[-+]\d+
  // (e.g., 1.2e+3) or the form \de[-+]\d+ (e.g., 1e+3).
  return [
    coefficient.length > 1 ? coefficient[0] + coefficient.slice(2) : coefficient,
    +x.slice(i + 1)
  ];
};

var exponent$1 = function(x) {
  return x = formatDecimal(Math.abs(x)), x ? x[1] : NaN;
};

var formatGroup = function(grouping, thousands) {
  return function(value, width) {
    var i = value.length,
        t = [],
        j = 0,
        g = grouping[0],
        length = 0;

    while (i > 0 && g > 0) {
      if (length + g + 1 > width) g = Math.max(1, width - length);
      t.push(value.substring(i -= g, i + g));
      if ((length += g + 1) > width) break;
      g = grouping[j = (j + 1) % grouping.length];
    }

    return t.reverse().join(thousands);
  };
};

var formatNumerals = function(numerals) {
  return function(value) {
    return value.replace(/[0-9]/g, function(i) {
      return numerals[+i];
    });
  };
};

var formatDefault = function(x, p) {
  x = x.toPrecision(p);

  out: for (var n = x.length, i = 1, i0 = -1, i1; i < n; ++i) {
    switch (x[i]) {
      case ".": i0 = i1 = i; break;
      case "0": if (i0 === 0) i0 = i; i1 = i; break;
      case "e": break out;
      default: if (i0 > 0) i0 = 0; break;
    }
  }

  return i0 > 0 ? x.slice(0, i0) + x.slice(i1 + 1) : x;
};

var prefixExponent;

var formatPrefixAuto = function(x, p) {
  var d = formatDecimal(x, p);
  if (!d) return x + "";
  var coefficient = d[0],
      exponent = d[1],
      i = exponent - (prefixExponent = Math.max(-8, Math.min(8, Math.floor(exponent / 3))) * 3) + 1,
      n = coefficient.length;
  return i === n ? coefficient
      : i > n ? coefficient + new Array(i - n + 1).join("0")
      : i > 0 ? coefficient.slice(0, i) + "." + coefficient.slice(i)
      : "0." + new Array(1 - i).join("0") + formatDecimal(x, Math.max(0, p + i - 1))[0]; // less than 1y!
};

var formatRounded = function(x, p) {
  var d = formatDecimal(x, p);
  if (!d) return x + "";
  var coefficient = d[0],
      exponent = d[1];
  return exponent < 0 ? "0." + new Array(-exponent).join("0") + coefficient
      : coefficient.length > exponent + 1 ? coefficient.slice(0, exponent + 1) + "." + coefficient.slice(exponent + 1)
      : coefficient + new Array(exponent - coefficient.length + 2).join("0");
};

var formatTypes = {
  "": formatDefault,
  "%": function(x, p) { return (x * 100).toFixed(p); },
  "b": function(x) { return Math.round(x).toString(2); },
  "c": function(x) { return x + ""; },
  "d": function(x) { return Math.round(x).toString(10); },
  "e": function(x, p) { return x.toExponential(p); },
  "f": function(x, p) { return x.toFixed(p); },
  "g": function(x, p) { return x.toPrecision(p); },
  "o": function(x) { return Math.round(x).toString(8); },
  "p": function(x, p) { return formatRounded(x * 100, p); },
  "r": formatRounded,
  "s": formatPrefixAuto,
  "X": function(x) { return Math.round(x).toString(16).toUpperCase(); },
  "x": function(x) { return Math.round(x).toString(16); }
};

// [[fill]align][sign][symbol][0][width][,][.precision][type]
var re = /^(?:(.)?([<>=^]))?([+\-\( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?([a-z%])?$/i;

function formatSpecifier(specifier) {
  return new FormatSpecifier(specifier);
}

formatSpecifier.prototype = FormatSpecifier.prototype; // instanceof

function FormatSpecifier(specifier) {
  if (!(match = re.exec(specifier))) throw new Error("invalid format: " + specifier);

  var match,
      fill = match[1] || " ",
      align = match[2] || ">",
      sign = match[3] || "-",
      symbol = match[4] || "",
      zero = !!match[5],
      width = match[6] && +match[6],
      comma = !!match[7],
      precision = match[8] && +match[8].slice(1),
      type = match[9] || "";

  // The "n" type is an alias for ",g".
  if (type === "n") comma = true, type = "g";

  // Map invalid types to the default format.
  else if (!formatTypes[type]) type = "";

  // If zero fill is specified, padding goes after sign and before digits.
  if (zero || (fill === "0" && align === "=")) zero = true, fill = "0", align = "=";

  this.fill = fill;
  this.align = align;
  this.sign = sign;
  this.symbol = symbol;
  this.zero = zero;
  this.width = width;
  this.comma = comma;
  this.precision = precision;
  this.type = type;
}

FormatSpecifier.prototype.toString = function() {
  return this.fill
      + this.align
      + this.sign
      + this.symbol
      + (this.zero ? "0" : "")
      + (this.width == null ? "" : Math.max(1, this.width | 0))
      + (this.comma ? "," : "")
      + (this.precision == null ? "" : "." + Math.max(0, this.precision | 0))
      + this.type;
};

var identity$3 = function(x) {
  return x;
};

var prefixes = ["y","z","a","f","p","n","µ","m","","k","M","G","T","P","E","Z","Y"];

var formatLocale = function(locale) {
  var group = locale.grouping && locale.thousands ? formatGroup(locale.grouping, locale.thousands) : identity$3,
      currency = locale.currency,
      decimal = locale.decimal,
      numerals = locale.numerals ? formatNumerals(locale.numerals) : identity$3,
      percent = locale.percent || "%";

  function newFormat(specifier) {
    specifier = formatSpecifier(specifier);

    var fill = specifier.fill,
        align = specifier.align,
        sign = specifier.sign,
        symbol = specifier.symbol,
        zero = specifier.zero,
        width = specifier.width,
        comma = specifier.comma,
        precision = specifier.precision,
        type = specifier.type;

    // Compute the prefix and suffix.
    // For SI-prefix, the suffix is lazily computed.
    var prefix = symbol === "$" ? currency[0] : symbol === "#" && /[boxX]/.test(type) ? "0" + type.toLowerCase() : "",
        suffix = symbol === "$" ? currency[1] : /[%p]/.test(type) ? percent : "";

    // What format function should we use?
    // Is this an integer type?
    // Can this type generate exponential notation?
    var formatType = formatTypes[type],
        maybeSuffix = !type || /[defgprs%]/.test(type);

    // Set the default precision if not specified,
    // or clamp the specified precision to the supported range.
    // For significant precision, it must be in [1, 21].
    // For fixed precision, it must be in [0, 20].
    precision = precision == null ? (type ? 6 : 12)
        : /[gprs]/.test(type) ? Math.max(1, Math.min(21, precision))
        : Math.max(0, Math.min(20, precision));

    function format(value) {
      var valuePrefix = prefix,
          valueSuffix = suffix,
          i, n, c;

      if (type === "c") {
        valueSuffix = formatType(value) + valueSuffix;
        value = "";
      } else {
        value = +value;

        // Perform the initial formatting.
        var valueNegative = value < 0;
        value = formatType(Math.abs(value), precision);

        // If a negative value rounds to zero during formatting, treat as positive.
        if (valueNegative && +value === 0) valueNegative = false;

        // Compute the prefix and suffix.
        valuePrefix = (valueNegative ? (sign === "(" ? sign : "-") : sign === "-" || sign === "(" ? "" : sign) + valuePrefix;
        valueSuffix = valueSuffix + (type === "s" ? prefixes[8 + prefixExponent / 3] : "") + (valueNegative && sign === "(" ? ")" : "");

        // Break the formatted value into the integer “value” part that can be
        // grouped, and fractional or exponential “suffix” part that is not.
        if (maybeSuffix) {
          i = -1, n = value.length;
          while (++i < n) {
            if (c = value.charCodeAt(i), 48 > c || c > 57) {
              valueSuffix = (c === 46 ? decimal + value.slice(i + 1) : value.slice(i)) + valueSuffix;
              value = value.slice(0, i);
              break;
            }
          }
        }
      }

      // If the fill character is not "0", grouping is applied before padding.
      if (comma && !zero) value = group(value, Infinity);

      // Compute the padding.
      var length = valuePrefix.length + value.length + valueSuffix.length,
          padding = length < width ? new Array(width - length + 1).join(fill) : "";

      // If the fill character is "0", grouping is applied after padding.
      if (comma && zero) value = group(padding + value, padding.length ? width - valueSuffix.length : Infinity), padding = "";

      // Reconstruct the final output based on the desired alignment.
      switch (align) {
        case "<": value = valuePrefix + value + valueSuffix + padding; break;
        case "=": value = valuePrefix + padding + value + valueSuffix; break;
        case "^": value = padding.slice(0, length = padding.length >> 1) + valuePrefix + value + valueSuffix + padding.slice(length); break;
        default: value = padding + valuePrefix + value + valueSuffix; break;
      }

      return numerals(value);
    }

    format.toString = function() {
      return specifier + "";
    };

    return format;
  }

  function formatPrefix(specifier, value) {
    var f = newFormat((specifier = formatSpecifier(specifier), specifier.type = "f", specifier)),
        e = Math.max(-8, Math.min(8, Math.floor(exponent$1(value) / 3))) * 3,
        k = Math.pow(10, -e),
        prefix = prefixes[8 + e / 3];
    return function(value) {
      return f(k * value) + prefix;
    };
  }

  return {
    format: newFormat,
    formatPrefix: formatPrefix
  };
};

var locale$1;
var format;
var formatPrefix;

defaultLocale({
  decimal: ".",
  thousands: ",",
  grouping: [3],
  currency: ["$", ""]
});

function defaultLocale(definition) {
  locale$1 = formatLocale(definition);
  format = locale$1.format;
  formatPrefix = locale$1.formatPrefix;
  return locale$1;
}

var precisionFixed = function(step) {
  return Math.max(0, -exponent$1(Math.abs(step)));
};

var precisionPrefix = function(step, value) {
  return Math.max(0, Math.max(-8, Math.min(8, Math.floor(exponent$1(value) / 3))) * 3 - exponent$1(Math.abs(step)));
};

var precisionRound = function(step, max) {
  step = Math.abs(step), max = Math.abs(max) - step;
  return Math.max(0, exponent$1(max) - exponent$1(step)) + 1;
};

// Adds floating point numbers with twice the normal precision.
// Reference: J. R. Shewchuk, Adaptive Precision Floating-Point Arithmetic and
// Fast Robust Geometric Predicates, Discrete & Computational Geometry 18(3)
// 305–363 (1997).
// Code adapted from GeographicLib by Charles F. F. Karney,
// http://geographiclib.sourceforge.net/

var adder = function() {
  return new Adder;
};

function Adder() {
  this.reset();
}

Adder.prototype = {
  constructor: Adder,
  reset: function() {
    this.s = // rounded value
    this.t = 0; // exact error
  },
  add: function(y) {
    add$1(temp, y, this.t);
    add$1(this, temp.s, this.s);
    if (this.s) this.t += temp.t;
    else this.s = temp.t;
  },
  valueOf: function() {
    return this.s;
  }
};

var temp = new Adder;

function add$1(adder, a, b) {
  var x = adder.s = a + b,
      bv = x - a,
      av = x - bv;
  adder.t = (a - av) + (b - bv);
}

var epsilon$2 = 1e-6;

var pi$3 = Math.PI;
var halfPi$2 = pi$3 / 2;
var quarterPi = pi$3 / 4;
var tau$3 = pi$3 * 2;


var radians = pi$3 / 180;

var abs = Math.abs;
var atan = Math.atan;
var atan2 = Math.atan2;
var cos$1 = Math.cos;





var sin$1 = Math.sin;

var sqrt = Math.sqrt;


function acos(x) {
  return x > 1 ? 0 : x < -1 ? pi$3 : Math.acos(x);
}

function asin(x) {
  return x > 1 ? halfPi$2 : x < -1 ? -halfPi$2 : Math.asin(x);
}

function noop$1() {}

function streamGeometry(geometry, stream) {
  if (geometry && streamGeometryType.hasOwnProperty(geometry.type)) {
    streamGeometryType[geometry.type](geometry, stream);
  }
}

var streamObjectType = {
  Feature: function(object, stream) {
    streamGeometry(object.geometry, stream);
  },
  FeatureCollection: function(object, stream) {
    var features = object.features, i = -1, n = features.length;
    while (++i < n) streamGeometry(features[i].geometry, stream);
  }
};

var streamGeometryType = {
  Sphere: function(object, stream) {
    stream.sphere();
  },
  Point: function(object, stream) {
    object = object.coordinates;
    stream.point(object[0], object[1], object[2]);
  },
  MultiPoint: function(object, stream) {
    var coordinates = object.coordinates, i = -1, n = coordinates.length;
    while (++i < n) object = coordinates[i], stream.point(object[0], object[1], object[2]);
  },
  LineString: function(object, stream) {
    streamLine(object.coordinates, stream, 0);
  },
  MultiLineString: function(object, stream) {
    var coordinates = object.coordinates, i = -1, n = coordinates.length;
    while (++i < n) streamLine(coordinates[i], stream, 0);
  },
  Polygon: function(object, stream) {
    streamPolygon(object.coordinates, stream);
  },
  MultiPolygon: function(object, stream) {
    var coordinates = object.coordinates, i = -1, n = coordinates.length;
    while (++i < n) streamPolygon(coordinates[i], stream);
  },
  GeometryCollection: function(object, stream) {
    var geometries = object.geometries, i = -1, n = geometries.length;
    while (++i < n) streamGeometry(geometries[i], stream);
  }
};

function streamLine(coordinates, stream, closed) {
  var i = -1, n = coordinates.length - closed, coordinate;
  stream.lineStart();
  while (++i < n) coordinate = coordinates[i], stream.point(coordinate[0], coordinate[1], coordinate[2]);
  stream.lineEnd();
}

function streamPolygon(coordinates, stream) {
  var i = -1, n = coordinates.length;
  stream.polygonStart();
  while (++i < n) streamLine(coordinates[i], stream, 1);
  stream.polygonEnd();
}

var geoStream = function(object, stream) {
  if (object && streamObjectType.hasOwnProperty(object.type)) {
    streamObjectType[object.type](object, stream);
  } else {
    streamGeometry(object, stream);
  }
};

var areaRingSum = adder();

var areaSum = adder();
var lambda00;
var phi00;
var lambda0;
var cosPhi0;
var sinPhi0;

function cartesian(spherical) {
  var lambda = spherical[0], phi = spherical[1], cosPhi = cos$1(phi);
  return [cosPhi * cos$1(lambda), cosPhi * sin$1(lambda), sin$1(phi)];
}



function cartesianCross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

// TODO return a




// TODO return d
function cartesianNormalizeInPlace(d) {
  var l = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  d[0] /= l, d[1] /= l, d[2] /= l;
}

var lambda0$1;
var phi0;
var lambda1;
var phi1;
var lambda2;
var lambda00$1;
var phi00$1;
var p0;
var deltaSum = adder();
var ranges;
var range;

var W0;
var X0;
var Y0;
var Z0; // previous point

// Generates a circle centered at [0°, 0°], with a given radius and precision.

var clipBuffer = function() {
  var lines = [],
      line;
  return {
    point: function(x, y) {
      line.push([x, y]);
    },
    lineStart: function() {
      lines.push(line = []);
    },
    lineEnd: noop$1,
    rejoin: function() {
      if (lines.length > 1) lines.push(lines.pop().concat(lines.shift()));
    },
    result: function() {
      var result = lines;
      lines = [];
      line = null;
      return result;
    }
  };
};

var pointEqual = function(a, b) {
  return abs(a[0] - b[0]) < epsilon$2 && abs(a[1] - b[1]) < epsilon$2;
};

function Intersection(point, points, other, entry) {
  this.x = point;
  this.z = points;
  this.o = other; // another intersection
  this.e = entry; // is an entry?
  this.v = false; // visited
  this.n = this.p = null; // next & previous
}

// A generalized polygon clipping algorithm: given a polygon that has been cut
// into its visible line segments, and rejoins the segments by interpolating
// along the clip edge.
var clipPolygon = function(segments, compareIntersection, startInside, interpolate, stream) {
  var subject = [],
      clip = [],
      i,
      n;

  segments.forEach(function(segment) {
    if ((n = segment.length - 1) <= 0) return;
    var n, p0 = segment[0], p1 = segment[n], x;

    // If the first and last points of a segment are coincident, then treat as a
    // closed ring. TODO if all rings are closed, then the winding order of the
    // exterior ring should be checked.
    if (pointEqual(p0, p1)) {
      stream.lineStart();
      for (i = 0; i < n; ++i) stream.point((p0 = segment[i])[0], p0[1]);
      stream.lineEnd();
      return;
    }

    subject.push(x = new Intersection(p0, segment, null, true));
    clip.push(x.o = new Intersection(p0, null, x, false));
    subject.push(x = new Intersection(p1, segment, null, false));
    clip.push(x.o = new Intersection(p1, null, x, true));
  });

  if (!subject.length) return;

  clip.sort(compareIntersection);
  link$1(subject);
  link$1(clip);

  for (i = 0, n = clip.length; i < n; ++i) {
    clip[i].e = startInside = !startInside;
  }

  var start = subject[0],
      points,
      point;

  while (1) {
    // Find first unvisited intersection.
    var current = start,
        isSubject = true;
    while (current.v) if ((current = current.n) === start) return;
    points = current.z;
    stream.lineStart();
    do {
      current.v = current.o.v = true;
      if (current.e) {
        if (isSubject) {
          for (i = 0, n = points.length; i < n; ++i) stream.point((point = points[i])[0], point[1]);
        } else {
          interpolate(current.x, current.n.x, 1, stream);
        }
        current = current.n;
      } else {
        if (isSubject) {
          points = current.p.z;
          for (i = points.length - 1; i >= 0; --i) stream.point((point = points[i])[0], point[1]);
        } else {
          interpolate(current.x, current.p.x, -1, stream);
        }
        current = current.p;
      }
      current = current.o;
      points = current.z;
      isSubject = !isSubject;
    } while (!current.v);
    stream.lineEnd();
  }
};

function link$1(array) {
  if (!(n = array.length)) return;
  var n,
      i = 0,
      a = array[0],
      b;
  while (++i < n) {
    a.n = b = array[i];
    b.p = a;
    a = b;
  }
  a.n = b = array[0];
  b.p = a;
}

// TODO Use d3-polygon’s polygonContains here for the ring check?
// TODO Eliminate duplicate buffering in clipBuffer and polygon.push?

var sum$1 = adder();

var polygonContains = function(polygon, point) {
  var lambda = point[0],
      phi = point[1],
      normal = [sin$1(lambda), -cos$1(lambda), 0],
      angle = 0,
      winding = 0;

  sum$1.reset();

  for (var i = 0, n = polygon.length; i < n; ++i) {
    if (!(m = (ring = polygon[i]).length)) continue;
    var ring,
        m,
        point0 = ring[m - 1],
        lambda0 = point0[0],
        phi0 = point0[1] / 2 + quarterPi,
        sinPhi0 = sin$1(phi0),
        cosPhi0 = cos$1(phi0);

    for (var j = 0; j < m; ++j, lambda0 = lambda1, sinPhi0 = sinPhi1, cosPhi0 = cosPhi1, point0 = point1) {
      var point1 = ring[j],
          lambda1 = point1[0],
          phi1 = point1[1] / 2 + quarterPi,
          sinPhi1 = sin$1(phi1),
          cosPhi1 = cos$1(phi1),
          delta = lambda1 - lambda0,
          sign$$1 = delta >= 0 ? 1 : -1,
          absDelta = sign$$1 * delta,
          antimeridian = absDelta > pi$3,
          k = sinPhi0 * sinPhi1;

      sum$1.add(atan2(k * sign$$1 * sin$1(absDelta), cosPhi0 * cosPhi1 + k * cos$1(absDelta)));
      angle += antimeridian ? delta + sign$$1 * tau$3 : delta;

      // Are the longitudes either side of the point’s meridian (lambda),
      // and are the latitudes smaller than the parallel (phi)?
      if (antimeridian ^ lambda0 >= lambda ^ lambda1 >= lambda) {
        var arc = cartesianCross(cartesian(point0), cartesian(point1));
        cartesianNormalizeInPlace(arc);
        var intersection = cartesianCross(normal, arc);
        cartesianNormalizeInPlace(intersection);
        var phiArc = (antimeridian ^ delta >= 0 ? -1 : 1) * asin(intersection[2]);
        if (phi > phiArc || phi === phiArc && (arc[0] || arc[1])) {
          winding += antimeridian ^ delta >= 0 ? 1 : -1;
        }
      }
    }
  }

  // First, determine whether the South pole is inside or outside:
  //
  // It is inside if:
  // * the polygon winds around it in a clockwise direction.
  // * the polygon does not (cumulatively) wind around it, but has a negative
  //   (counter-clockwise) area.
  //
  // Second, count the (signed) number of times a segment crosses a lambda
  // from the point to the South pole.  If it is zero, then the point is the
  // same side as the South pole.

  return (angle < -epsilon$2 || angle < epsilon$2 && sum$1 < -epsilon$2) ^ (winding & 1);
};

var lengthSum = adder();
var lambda0$2;
var sinPhi0$1;
var cosPhi0$1;

var lengthStream = {
  sphere: noop$1,
  point: noop$1,
  lineStart: lengthLineStart,
  lineEnd: noop$1,
  polygonStart: noop$1,
  polygonEnd: noop$1
};

function lengthLineStart() {
  lengthStream.point = lengthPointFirst;
  lengthStream.lineEnd = lengthLineEnd;
}

function lengthLineEnd() {
  lengthStream.point = lengthStream.lineEnd = noop$1;
}

function lengthPointFirst(lambda, phi) {
  lambda *= radians, phi *= radians;
  lambda0$2 = lambda, sinPhi0$1 = sin$1(phi), cosPhi0$1 = cos$1(phi);
  lengthStream.point = lengthPoint;
}

function lengthPoint(lambda, phi) {
  lambda *= radians, phi *= radians;
  var sinPhi = sin$1(phi),
      cosPhi = cos$1(phi),
      delta = abs(lambda - lambda0$2),
      cosDelta = cos$1(delta),
      sinDelta = sin$1(delta),
      x = cosPhi * sinDelta,
      y = cosPhi0$1 * sinPhi - sinPhi0$1 * cosPhi * cosDelta,
      z = sinPhi0$1 * sinPhi + cosPhi0$1 * cosPhi * cosDelta;
  lengthSum.add(atan2(sqrt(x * x + y * y), z));
  lambda0$2 = lambda, sinPhi0$1 = sinPhi, cosPhi0$1 = cosPhi;
}

var length$1 = function(object) {
  lengthSum.reset();
  geoStream(object, lengthStream);
  return +lengthSum;
};

var coordinates = [null, null];
var object$1 = {type: "LineString", coordinates: coordinates};

var distance = function(a, b) {
  coordinates[0] = a;
  coordinates[1] = b;
  return length$1(object$1);
};

var containsGeometryType = {
  Sphere: function() {
    return true;
  },
  Point: function(object, point) {
    return containsPoint(object.coordinates, point);
  },
  MultiPoint: function(object, point) {
    var coordinates = object.coordinates, i = -1, n = coordinates.length;
    while (++i < n) if (containsPoint(coordinates[i], point)) return true;
    return false;
  },
  LineString: function(object, point) {
    return containsLine(object.coordinates, point);
  },
  MultiLineString: function(object, point) {
    var coordinates = object.coordinates, i = -1, n = coordinates.length;
    while (++i < n) if (containsLine(coordinates[i], point)) return true;
    return false;
  },
  Polygon: function(object, point) {
    return containsPolygon(object.coordinates, point);
  },
  MultiPolygon: function(object, point) {
    var coordinates = object.coordinates, i = -1, n = coordinates.length;
    while (++i < n) if (containsPolygon(coordinates[i], point)) return true;
    return false;
  },
  GeometryCollection: function(object, point) {
    var geometries = object.geometries, i = -1, n = geometries.length;
    while (++i < n) if (containsGeometry(geometries[i], point)) return true;
    return false;
  }
};

function containsGeometry(geometry, point) {
  return geometry && containsGeometryType.hasOwnProperty(geometry.type)
      ? containsGeometryType[geometry.type](geometry, point)
      : false;
}

function containsPoint(coordinates, point) {
  return distance(coordinates, point) === 0;
}

function containsLine(coordinates, point) {
  var ab = distance(coordinates[0], coordinates[1]),
      ao = distance(coordinates[0], point),
      ob = distance(point, coordinates[1]);
  return ao + ob <= ab + epsilon$2;
}

function containsPolygon(coordinates, point) {
  return !!polygonContains(coordinates.map(ringRadians), pointRadians(point));
}

function ringRadians(ring) {
  return ring = ring.map(pointRadians), ring.pop(), ring;
}

function pointRadians(point) {
  return [point[0] * radians, point[1] * radians];
}

var areaSum$1 = adder();
var areaRingSum$1 = adder();
var x00;
var y00;
var x0$1;
var y0$1;

// TODO Enforce positive area for exterior, negative area for interior?

var X0$1 = 0;
var Y0$1 = 0;
var Z0$1 = 0;

var lengthSum$1 = adder();
var lengthRing;
var x00$2;
var y00$2;
var x0$4;
var y0$4;

var clip = function(pointVisible, clipLine, interpolate, start) {
  return function(rotate, sink) {
    var line = clipLine(sink),
        rotatedStart = rotate.invert(start[0], start[1]),
        ringBuffer = clipBuffer(),
        ringSink = clipLine(ringBuffer),
        polygonStarted = false,
        polygon,
        segments,
        ring;

    var clip = {
      point: point,
      lineStart: lineStart,
      lineEnd: lineEnd,
      polygonStart: function() {
        clip.point = pointRing;
        clip.lineStart = ringStart;
        clip.lineEnd = ringEnd;
        segments = [];
        polygon = [];
      },
      polygonEnd: function() {
        clip.point = point;
        clip.lineStart = lineStart;
        clip.lineEnd = lineEnd;
        segments = merge(segments);
        var startInside = polygonContains(polygon, rotatedStart);
        if (segments.length) {
          if (!polygonStarted) sink.polygonStart(), polygonStarted = true;
          clipPolygon(segments, compareIntersection, startInside, interpolate, sink);
        } else if (startInside) {
          if (!polygonStarted) sink.polygonStart(), polygonStarted = true;
          sink.lineStart();
          interpolate(null, null, 1, sink);
          sink.lineEnd();
        }
        if (polygonStarted) sink.polygonEnd(), polygonStarted = false;
        segments = polygon = null;
      },
      sphere: function() {
        sink.polygonStart();
        sink.lineStart();
        interpolate(null, null, 1, sink);
        sink.lineEnd();
        sink.polygonEnd();
      }
    };

    function point(lambda, phi) {
      var point = rotate(lambda, phi);
      if (pointVisible(lambda = point[0], phi = point[1])) sink.point(lambda, phi);
    }

    function pointLine(lambda, phi) {
      var point = rotate(lambda, phi);
      line.point(point[0], point[1]);
    }

    function lineStart() {
      clip.point = pointLine;
      line.lineStart();
    }

    function lineEnd() {
      clip.point = point;
      line.lineEnd();
    }

    function pointRing(lambda, phi) {
      ring.push([lambda, phi]);
      var point = rotate(lambda, phi);
      ringSink.point(point[0], point[1]);
    }

    function ringStart() {
      ringSink.lineStart();
      ring = [];
    }

    function ringEnd() {
      pointRing(ring[0][0], ring[0][1]);
      ringSink.lineEnd();

      var clean = ringSink.clean(),
          ringSegments = ringBuffer.result(),
          i, n = ringSegments.length, m,
          segment,
          point;

      ring.pop();
      polygon.push(ring);
      ring = null;

      if (!n) return;

      // No intersections.
      if (clean & 1) {
        segment = ringSegments[0];
        if ((m = segment.length - 1) > 0) {
          if (!polygonStarted) sink.polygonStart(), polygonStarted = true;
          sink.lineStart();
          for (i = 0; i < m; ++i) sink.point((point = segment[i])[0], point[1]);
          sink.lineEnd();
        }
        return;
      }

      // Rejoin connected segments.
      // TODO reuse ringBuffer.rejoin()?
      if (n > 1 && clean & 2) ringSegments.push(ringSegments.pop().concat(ringSegments.shift()));

      segments.push(ringSegments.filter(validSegment));
    }

    return clip;
  };
};

function validSegment(segment) {
  return segment.length > 1;
}

// Intersections are sorted along the clip edge. For both antimeridian cutting
// and circle clipping, the same comparison is used.
function compareIntersection(a, b) {
  return ((a = a.x)[0] < 0 ? a[1] - halfPi$2 - epsilon$2 : halfPi$2 - a[1])
       - ((b = b.x)[0] < 0 ? b[1] - halfPi$2 - epsilon$2 : halfPi$2 - b[1]);
}

clip(
  function() { return true; },
  clipAntimeridianLine,
  clipAntimeridianInterpolate,
  [-pi$3, -halfPi$2]
);

// Takes a line and cuts into visible segments. Return values: 0 - there were
// intersections or the line was empty; 1 - no intersections; 2 - there were
// intersections, and the first and last segments should be rejoined.
function clipAntimeridianLine(stream) {
  var lambda0 = NaN,
      phi0 = NaN,
      sign0 = NaN,
      clean; // no intersections

  return {
    lineStart: function() {
      stream.lineStart();
      clean = 1;
    },
    point: function(lambda1, phi1) {
      var sign1 = lambda1 > 0 ? pi$3 : -pi$3,
          delta = abs(lambda1 - lambda0);
      if (abs(delta - pi$3) < epsilon$2) { // line crosses a pole
        stream.point(lambda0, phi0 = (phi0 + phi1) / 2 > 0 ? halfPi$2 : -halfPi$2);
        stream.point(sign0, phi0);
        stream.lineEnd();
        stream.lineStart();
        stream.point(sign1, phi0);
        stream.point(lambda1, phi0);
        clean = 0;
      } else if (sign0 !== sign1 && delta >= pi$3) { // line crosses antimeridian
        if (abs(lambda0 - sign0) < epsilon$2) lambda0 -= sign0 * epsilon$2; // handle degeneracies
        if (abs(lambda1 - sign1) < epsilon$2) lambda1 -= sign1 * epsilon$2;
        phi0 = clipAntimeridianIntersect(lambda0, phi0, lambda1, phi1);
        stream.point(sign0, phi0);
        stream.lineEnd();
        stream.lineStart();
        stream.point(sign1, phi0);
        clean = 0;
      }
      stream.point(lambda0 = lambda1, phi0 = phi1);
      sign0 = sign1;
    },
    lineEnd: function() {
      stream.lineEnd();
      lambda0 = phi0 = NaN;
    },
    clean: function() {
      return 2 - clean; // if intersections, rejoin first and last segments
    }
  };
}

function clipAntimeridianIntersect(lambda0, phi0, lambda1, phi1) {
  var cosPhi0,
      cosPhi1,
      sinLambda0Lambda1 = sin$1(lambda0 - lambda1);
  return abs(sinLambda0Lambda1) > epsilon$2
      ? atan((sin$1(phi0) * (cosPhi1 = cos$1(phi1)) * sin$1(lambda1)
          - sin$1(phi1) * (cosPhi0 = cos$1(phi0)) * sin$1(lambda0))
          / (cosPhi0 * cosPhi1 * sinLambda0Lambda1))
      : (phi0 + phi1) / 2;
}

function clipAntimeridianInterpolate(from, to, direction, stream) {
  var phi;
  if (from == null) {
    phi = direction * halfPi$2;
    stream.point(-pi$3, phi);
    stream.point(0, phi);
    stream.point(pi$3, phi);
    stream.point(pi$3, 0);
    stream.point(pi$3, -phi);
    stream.point(0, -phi);
    stream.point(-pi$3, -phi);
    stream.point(-pi$3, 0);
    stream.point(-pi$3, phi);
  } else if (abs(from[0] - to[0]) > epsilon$2) {
    var lambda = from[0] < to[0] ? pi$3 : -pi$3;
    phi = direction * lambda / 2;
    stream.point(-lambda, phi);
    stream.point(0, phi);
    stream.point(lambda, phi);
  } else {
    stream.point(to[0], to[1]);
  }
}

var cosMinDistance = cos$1(30 * radians); // cos(minimum angular distance)

// A composite projection for the United States, configured by default for
// 960×500. The projection also works quite well at 960×600 if you change the
// scale to 1285 and adjust the translate accordingly. The set of standard
// parallels for each region comes from USGS, which is published here:
// http://egsc.usgs.gov/isb/pubs/MapProjections/projections.html#albers

function azimuthalRaw(scale) {
  return function(x, y) {
    var cx = cos$1(x),
        cy = cos$1(y),
        k = scale(cx * cy);
    return [
      k * cy * sin$1(x),
      k * sin$1(y)
    ];
  }
}

function azimuthalInvert(angle) {
  return function(x, y) {
    var z = sqrt(x * x + y * y),
        c = angle(z),
        sc = sin$1(c),
        cc = cos$1(c);
    return [
      atan2(x * sc, z * cc),
      asin(z && y * sc / z)
    ];
  }
}

var azimuthalEqualAreaRaw = azimuthalRaw(function(cxcy) {
  return sqrt(2 / (1 + cxcy));
});

azimuthalEqualAreaRaw.invert = azimuthalInvert(function(z) {
  return 2 * asin(z / 2);
});

var azimuthalEquidistantRaw = azimuthalRaw(function(c) {
  return (c = acos(c)) && c / sin$1(c);
});

azimuthalEquidistantRaw.invert = azimuthalInvert(function(z) {
  return z;
});

function count(node) {
  var sum = 0,
      children = node.children,
      i = children && children.length;
  if (!i) sum = 1;
  else while (--i >= 0) sum += children[i].value;
  node.value = sum;
}

var node_count = function() {
  return this.eachAfter(count);
};

var node_each = function(callback) {
  var node = this, current, next = [node], children, i, n;
  do {
    current = next.reverse(), next = [];
    while (node = current.pop()) {
      callback(node), children = node.children;
      if (children) for (i = 0, n = children.length; i < n; ++i) {
        next.push(children[i]);
      }
    }
  } while (next.length);
  return this;
};

var node_eachBefore = function(callback) {
  var node = this, nodes = [node], children, i;
  while (node = nodes.pop()) {
    callback(node), children = node.children;
    if (children) for (i = children.length - 1; i >= 0; --i) {
      nodes.push(children[i]);
    }
  }
  return this;
};

var node_eachAfter = function(callback) {
  var node = this, nodes = [node], next = [], children, i, n;
  while (node = nodes.pop()) {
    next.push(node), children = node.children;
    if (children) for (i = 0, n = children.length; i < n; ++i) {
      nodes.push(children[i]);
    }
  }
  while (node = next.pop()) {
    callback(node);
  }
  return this;
};

var node_sum = function(value) {
  return this.eachAfter(function(node) {
    var sum = +value(node.data) || 0,
        children = node.children,
        i = children && children.length;
    while (--i >= 0) sum += children[i].value;
    node.value = sum;
  });
};

var node_sort = function(compare) {
  return this.eachBefore(function(node) {
    if (node.children) {
      node.children.sort(compare);
    }
  });
};

var node_path = function(end) {
  var start = this,
      ancestor = leastCommonAncestor(start, end),
      nodes = [start];
  while (start !== ancestor) {
    start = start.parent;
    nodes.push(start);
  }
  var k = nodes.length;
  while (end !== ancestor) {
    nodes.splice(k, 0, end);
    end = end.parent;
  }
  return nodes;
};

function leastCommonAncestor(a, b) {
  if (a === b) return a;
  var aNodes = a.ancestors(),
      bNodes = b.ancestors(),
      c = null;
  a = aNodes.pop();
  b = bNodes.pop();
  while (a === b) {
    c = a;
    a = aNodes.pop();
    b = bNodes.pop();
  }
  return c;
}

var node_ancestors = function() {
  var node = this, nodes = [node];
  while (node = node.parent) {
    nodes.push(node);
  }
  return nodes;
};

var node_descendants = function() {
  var nodes = [];
  this.each(function(node) {
    nodes.push(node);
  });
  return nodes;
};

var node_leaves = function() {
  var leaves = [];
  this.eachBefore(function(node) {
    if (!node.children) {
      leaves.push(node);
    }
  });
  return leaves;
};

var node_links = function() {
  var root = this, links = [];
  root.each(function(node) {
    if (node !== root) { // Don’t include the root’s parent, if any.
      links.push({source: node.parent, target: node});
    }
  });
  return links;
};

function hierarchy(data, children) {
  var root = new Node(data),
      valued = +data.value && (root.value = data.value),
      node,
      nodes = [root],
      child,
      childs,
      i,
      n;

  if (children == null) children = defaultChildren;

  while (node = nodes.pop()) {
    if (valued) node.value = +node.data.value;
    if ((childs = children(node.data)) && (n = childs.length)) {
      node.children = new Array(n);
      for (i = n - 1; i >= 0; --i) {
        nodes.push(child = node.children[i] = new Node(childs[i]));
        child.parent = node;
        child.depth = node.depth + 1;
      }
    }
  }

  return root.eachBefore(computeHeight);
}

function node_copy() {
  return hierarchy(this).eachBefore(copyData);
}

function defaultChildren(d) {
  return d.children;
}

function copyData(node) {
  node.data = node.data.data;
}

function computeHeight(node) {
  var height = 0;
  do node.height = height;
  while ((node = node.parent) && (node.height < ++height));
}

function Node(data) {
  this.data = data;
  this.depth =
  this.height = 0;
  this.parent = null;
}

Node.prototype = hierarchy.prototype = {
  constructor: Node,
  count: node_count,
  each: node_each,
  eachAfter: node_eachAfter,
  eachBefore: node_eachBefore,
  sum: node_sum,
  sort: node_sort,
  path: node_path,
  ancestors: node_ancestors,
  descendants: node_descendants,
  leaves: node_leaves,
  links: node_links,
  copy: node_copy
};

function Node$2(value) {
  this._ = value;
  this.next = null;
}

var treemapDice = function(parent, x0, y0, x1, y1) {
  var nodes = parent.children,
      node,
      i = -1,
      n = nodes.length,
      k = parent.value && (x1 - x0) / parent.value;

  while (++i < n) {
    node = nodes[i], node.y0 = y0, node.y1 = y1;
    node.x0 = x0, node.x1 = x0 += node.value * k;
  }
};

function TreeNode(node, i) {
  this._ = node;
  this.parent = null;
  this.children = null;
  this.A = null; // default ancestor
  this.a = this; // ancestor
  this.z = 0; // prelim
  this.m = 0; // mod
  this.c = 0; // change
  this.s = 0; // shift
  this.t = null; // thread
  this.i = i; // number
}

TreeNode.prototype = Object.create(Node.prototype);

// Node-link tree diagram using the Reingold-Tilford "tidy" algorithm

var treemapSlice = function(parent, x0, y0, x1, y1) {
  var nodes = parent.children,
      node,
      i = -1,
      n = nodes.length,
      k = parent.value && (y1 - y0) / parent.value;

  while (++i < n) {
    node = nodes[i], node.x0 = x0, node.x1 = x1;
    node.y0 = y0, node.y1 = y0 += node.value * k;
  }
};

function squarifyRatio(ratio, parent, x0, y0, x1, y1) {
  var rows = [],
      nodes = parent.children,
      row,
      nodeValue,
      i0 = 0,
      i1 = 0,
      n = nodes.length,
      dx, dy,
      value = parent.value,
      sumValue,
      minValue,
      maxValue,
      newRatio,
      minRatio,
      alpha,
      beta;

  while (i0 < n) {
    dx = x1 - x0, dy = y1 - y0;

    // Find the next non-empty node.
    do sumValue = nodes[i1++].value; while (!sumValue && i1 < n);
    minValue = maxValue = sumValue;
    alpha = Math.max(dy / dx, dx / dy) / (value * ratio);
    beta = sumValue * sumValue * alpha;
    minRatio = Math.max(maxValue / beta, beta / minValue);

    // Keep adding nodes while the aspect ratio maintains or improves.
    for (; i1 < n; ++i1) {
      sumValue += nodeValue = nodes[i1].value;
      if (nodeValue < minValue) minValue = nodeValue;
      if (nodeValue > maxValue) maxValue = nodeValue;
      beta = sumValue * sumValue * alpha;
      newRatio = Math.max(maxValue / beta, beta / minValue);
      if (newRatio > minRatio) { sumValue -= nodeValue; break; }
      minRatio = newRatio;
    }

    // Position and record the row orientation.
    rows.push(row = {value: sumValue, dice: dx < dy, children: nodes.slice(i0, i1)});
    if (row.dice) treemapDice(row, x0, y0, x1, value ? y0 += dy * sumValue / value : y1);
    else treemapSlice(row, x0, y0, value ? x0 += dx * sumValue / value : x1, y1);
    value -= sumValue, i0 = i1;
  }

  return rows;
}

// Returns the 2D cross product of AB and AC vectors, i.e., the z-component of
// the 3D cross product in a quadrant I Cartesian coordinate system (+x is
// right, +y is up). Returns a positive value if ABC is counter-clockwise,
// negative if clockwise, and zero if the points are collinear.
var cross$1 = function(a, b, c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
};

function lexicographicOrder(a, b) {
  return a[0] - b[0] || a[1] - b[1];
}

// Computes the upper convex hull per the monotone chain algorithm.
// Assumes points.length >= 3, is sorted by x, unique in y.
// Returns an array of indices into points in left-to-right order.
function computeUpperHullIndexes(points) {
  var n = points.length,
      indexes = [0, 1],
      size = 2;

  for (var i = 2; i < n; ++i) {
    while (size > 1 && cross$1(points[indexes[size - 2]], points[indexes[size - 1]], points[i]) <= 0) --size;
    indexes[size++] = i;
  }

  return indexes.slice(0, size); // remove popped points
}

var slice$3 = [].slice;

var noabort = {};

function poke$1(q) {
  if (!q._start) {
    try { start$1(q); } // let the current task complete
    catch (e) {
      if (q._tasks[q._ended + q._active - 1]) abort(q, e); // task errored synchronously
      else if (!q._data) throw e; // await callback errored synchronously
    }
  }
}

function start$1(q) {
  while (q._start = q._waiting && q._active < q._size) {
    var i = q._ended + q._active,
        t = q._tasks[i],
        j = t.length - 1,
        c = t[j];
    t[j] = end(q, i);
    --q._waiting, ++q._active;
    t = c.apply(null, t);
    if (!q._tasks[i]) continue; // task finished synchronously
    q._tasks[i] = t || noabort;
  }
}

function end(q, i) {
  return function(e, r) {
    if (!q._tasks[i]) return; // ignore multiple callbacks
    --q._active, ++q._ended;
    q._tasks[i] = null;
    if (q._error != null) return; // ignore secondary errors
    if (e != null) {
      abort(q, e);
    } else {
      q._data[i] = r;
      if (q._waiting) poke$1(q);
      else maybeNotify(q);
    }
  };
}

function abort(q, e) {
  var i = q._tasks.length, t;
  q._error = e; // ignore active callbacks
  q._data = undefined; // allow gc
  q._waiting = NaN; // prevent starting

  while (--i >= 0) {
    if (t = q._tasks[i]) {
      q._tasks[i] = null;
      if (t.abort) {
        try { t.abort(); }
        catch (e) { /* ignore */ }
      }
    }
  }

  q._active = NaN; // allow notification
  maybeNotify(q);
}

function maybeNotify(q) {
  if (!q._active && q._call) {
    var d = q._data;
    q._data = undefined; // allow gc
    q._call(q._error, d);
  }
}

var request = function(url, callback) {
  var request,
      event = dispatch("beforesend", "progress", "load", "error"),
      mimeType,
      headers = map$1(),
      xhr = new XMLHttpRequest,
      user = null,
      password = null,
      response,
      responseType,
      timeout = 0;

  // If IE does not support CORS, use XDomainRequest.
  if (typeof XDomainRequest !== "undefined"
      && !("withCredentials" in xhr)
      && /^(http(s)?:)?\/\//.test(url)) xhr = new XDomainRequest;

  "onload" in xhr
      ? xhr.onload = xhr.onerror = xhr.ontimeout = respond
      : xhr.onreadystatechange = function(o) { xhr.readyState > 3 && respond(o); };

  function respond(o) {
    var status = xhr.status, result;
    if (!status && hasResponse(xhr)
        || status >= 200 && status < 300
        || status === 304) {
      if (response) {
        try {
          result = response.call(request, xhr);
        } catch (e) {
          event.call("error", request, e);
          return;
        }
      } else {
        result = xhr;
      }
      event.call("load", request, result);
    } else {
      event.call("error", request, o);
    }
  }

  xhr.onprogress = function(e) {
    event.call("progress", request, e);
  };

  request = {
    header: function(name, value) {
      name = (name + "").toLowerCase();
      if (arguments.length < 2) return headers.get(name);
      if (value == null) headers.remove(name);
      else headers.set(name, value + "");
      return request;
    },

    // If mimeType is non-null and no Accept header is set, a default is used.
    mimeType: function(value) {
      if (!arguments.length) return mimeType;
      mimeType = value == null ? null : value + "";
      return request;
    },

    // Specifies what type the response value should take;
    // for instance, arraybuffer, blob, document, or text.
    responseType: function(value) {
      if (!arguments.length) return responseType;
      responseType = value;
      return request;
    },

    timeout: function(value) {
      if (!arguments.length) return timeout;
      timeout = +value;
      return request;
    },

    user: function(value) {
      return arguments.length < 1 ? user : (user = value == null ? null : value + "", request);
    },

    password: function(value) {
      return arguments.length < 1 ? password : (password = value == null ? null : value + "", request);
    },

    // Specify how to convert the response content to a specific type;
    // changes the callback value on "load" events.
    response: function(value) {
      response = value;
      return request;
    },

    // Alias for send("GET", …).
    get: function(data, callback) {
      return request.send("GET", data, callback);
    },

    // Alias for send("POST", …).
    post: function(data, callback) {
      return request.send("POST", data, callback);
    },

    // If callback is non-null, it will be used for error and load events.
    send: function(method, data, callback) {
      xhr.open(method, url, true, user, password);
      if (mimeType != null && !headers.has("accept")) headers.set("accept", mimeType + ",*/*");
      if (xhr.setRequestHeader) headers.each(function(value, name) { xhr.setRequestHeader(name, value); });
      if (mimeType != null && xhr.overrideMimeType) xhr.overrideMimeType(mimeType);
      if (responseType != null) xhr.responseType = responseType;
      if (timeout > 0) xhr.timeout = timeout;
      if (callback == null && typeof data === "function") callback = data, data = null;
      if (callback != null && callback.length === 1) callback = fixCallback(callback);
      if (callback != null) request.on("error", callback).on("load", function(xhr) { callback(null, xhr); });
      event.call("beforesend", request, xhr);
      xhr.send(data == null ? null : data);
      return request;
    },

    abort: function() {
      xhr.abort();
      return request;
    },

    on: function() {
      var value = event.on.apply(event, arguments);
      return value === event ? request : value;
    }
  };

  if (callback != null) {
    if (typeof callback !== "function") throw new Error("invalid callback: " + callback);
    return request.get(callback);
  }

  return request;
};

function fixCallback(callback) {
  return function(error, xhr) {
    callback(error == null ? xhr : null);
  };
}

function hasResponse(xhr) {
  var type = xhr.responseType;
  return type && type !== "text"
      ? xhr.response // null on error
      : xhr.responseText; // "" on error
}

var type$1 = function(defaultMimeType, response) {
  return function(url, callback) {
    var r = request(url).mimeType(defaultMimeType).response(response);
    if (callback != null) {
      if (typeof callback !== "function") throw new Error("invalid callback: " + callback);
      return r.get(callback);
    }
    return r;
  };
};

type$1("text/html", function(xhr) {
  return document.createRange().createContextualFragment(xhr.responseText);
});

type$1("application/json", function(xhr) {
  return JSON.parse(xhr.responseText);
});

var text = type$1("text/plain", function(xhr) {
  return xhr.responseText;
});

type$1("application/xml", function(xhr) {
  var xml = xhr.responseXML;
  if (!xml) throw new Error("parse error");
  return xml;
});

var dsv$1 = function(defaultMimeType, parse) {
  return function(url, row, callback) {
    if (arguments.length < 3) callback = row, row = null;
    var r = request(url).mimeType(defaultMimeType);
    r.row = function(_) { return arguments.length ? r.response(responseOf(parse, row = _)) : row; };
    r.row(row);
    return callback ? r.get(callback) : r;
  };
};

function responseOf(parse, row) {
  return function(request$$1) {
    return parse(request$$1.responseText, row);
  };
}

dsv$1("text/csv", csvParse);

dsv$1("text/tab-separated-values", tsvParse);

var array$2 = Array.prototype;

var map$3 = array$2.map;
var slice$4 = array$2.slice;

var implicit = {name: "implicit"};

function ordinal(range) {
  var index = map$1(),
      domain = [],
      unknown = implicit;

  range = range == null ? [] : slice$4.call(range);

  function scale(d) {
    var key = d + "", i = index.get(key);
    if (!i) {
      if (unknown !== implicit) return unknown;
      index.set(key, i = domain.push(d));
    }
    return range[(i - 1) % range.length];
  }

  scale.domain = function(_) {
    if (!arguments.length) return domain.slice();
    domain = [], index = map$1();
    var i = -1, n = _.length, d, key;
    while (++i < n) if (!index.has(key = (d = _[i]) + "")) index.set(key, domain.push(d));
    return scale;
  };

  scale.range = function(_) {
    return arguments.length ? (range = slice$4.call(_), scale) : range.slice();
  };

  scale.unknown = function(_) {
    return arguments.length ? (unknown = _, scale) : unknown;
  };

  scale.copy = function() {
    return ordinal()
        .domain(domain)
        .range(range)
        .unknown(unknown);
  };

  return scale;
}

var constant$9 = function(x) {
  return function() {
    return x;
  };
};

var number$1 = function(x) {
  return +x;
};

var unit = [0, 1];

function deinterpolateLinear(a, b) {
  return (b -= (a = +a))
      ? function(x) { return (x - a) / b; }
      : constant$9(b);
}

function deinterpolateClamp(deinterpolate) {
  return function(a, b) {
    var d = deinterpolate(a = +a, b = +b);
    return function(x) { return x <= a ? 0 : x >= b ? 1 : d(x); };
  };
}

function reinterpolateClamp(reinterpolate) {
  return function(a, b) {
    var r = reinterpolate(a = +a, b = +b);
    return function(t) { return t <= 0 ? a : t >= 1 ? b : r(t); };
  };
}

function bimap(domain, range$$1, deinterpolate, reinterpolate) {
  var d0 = domain[0], d1 = domain[1], r0 = range$$1[0], r1 = range$$1[1];
  if (d1 < d0) d0 = deinterpolate(d1, d0), r0 = reinterpolate(r1, r0);
  else d0 = deinterpolate(d0, d1), r0 = reinterpolate(r0, r1);
  return function(x) { return r0(d0(x)); };
}

function polymap(domain, range$$1, deinterpolate, reinterpolate) {
  var j = Math.min(domain.length, range$$1.length) - 1,
      d = new Array(j),
      r = new Array(j),
      i = -1;

  // Reverse descending domains.
  if (domain[j] < domain[0]) {
    domain = domain.slice().reverse();
    range$$1 = range$$1.slice().reverse();
  }

  while (++i < j) {
    d[i] = deinterpolate(domain[i], domain[i + 1]);
    r[i] = reinterpolate(range$$1[i], range$$1[i + 1]);
  }

  return function(x) {
    var i = bisectRight(domain, x, 1, j) - 1;
    return r[i](d[i](x));
  };
}

function copy(source, target) {
  return target
      .domain(source.domain())
      .range(source.range())
      .interpolate(source.interpolate())
      .clamp(source.clamp());
}

// deinterpolate(a, b)(x) takes a domain value x in [a,b] and returns the corresponding parameter t in [0,1].
// reinterpolate(a, b)(t) takes a parameter t in [0,1] and returns the corresponding domain value x in [a,b].
function continuous(deinterpolate, reinterpolate) {
  var domain = unit,
      range$$1 = unit,
      interpolate$$1 = interpolateValue,
      clamp = false,
      piecewise,
      output,
      input;

  function rescale() {
    piecewise = Math.min(domain.length, range$$1.length) > 2 ? polymap : bimap;
    output = input = null;
    return scale;
  }

  function scale(x) {
    return (output || (output = piecewise(domain, range$$1, clamp ? deinterpolateClamp(deinterpolate) : deinterpolate, interpolate$$1)))(+x);
  }

  scale.invert = function(y) {
    return (input || (input = piecewise(range$$1, domain, deinterpolateLinear, clamp ? reinterpolateClamp(reinterpolate) : reinterpolate)))(+y);
  };

  scale.domain = function(_) {
    return arguments.length ? (domain = map$3.call(_, number$1), rescale()) : domain.slice();
  };

  scale.range = function(_) {
    return arguments.length ? (range$$1 = slice$4.call(_), rescale()) : range$$1.slice();
  };

  scale.rangeRound = function(_) {
    return range$$1 = slice$4.call(_), interpolate$$1 = interpolateRound, rescale();
  };

  scale.clamp = function(_) {
    return arguments.length ? (clamp = !!_, rescale()) : clamp;
  };

  scale.interpolate = function(_) {
    return arguments.length ? (interpolate$$1 = _, rescale()) : interpolate$$1;
  };

  return rescale();
}

var tickFormat = function(domain, count, specifier) {
  var start = domain[0],
      stop = domain[domain.length - 1],
      step = tickStep(start, stop, count == null ? 10 : count),
      precision;
  specifier = formatSpecifier(specifier == null ? ",f" : specifier);
  switch (specifier.type) {
    case "s": {
      var value = Math.max(Math.abs(start), Math.abs(stop));
      if (specifier.precision == null && !isNaN(precision = precisionPrefix(step, value))) specifier.precision = precision;
      return formatPrefix(specifier, value);
    }
    case "":
    case "e":
    case "g":
    case "p":
    case "r": {
      if (specifier.precision == null && !isNaN(precision = precisionRound(step, Math.max(Math.abs(start), Math.abs(stop))))) specifier.precision = precision - (specifier.type === "e");
      break;
    }
    case "f":
    case "%": {
      if (specifier.precision == null && !isNaN(precision = precisionFixed(step))) specifier.precision = precision - (specifier.type === "%") * 2;
      break;
    }
  }
  return format(specifier);
};

function linearish(scale) {
  var domain = scale.domain;

  scale.ticks = function(count) {
    var d = domain();
    return ticks(d[0], d[d.length - 1], count == null ? 10 : count);
  };

  scale.tickFormat = function(count, specifier) {
    return tickFormat(domain(), count, specifier);
  };

  scale.nice = function(count) {
    if (count == null) count = 10;

    var d = domain(),
        i0 = 0,
        i1 = d.length - 1,
        start = d[i0],
        stop = d[i1],
        step;

    if (stop < start) {
      step = start, start = stop, stop = step;
      step = i0, i0 = i1, i1 = step;
    }

    step = tickIncrement(start, stop, count);

    if (step > 0) {
      start = Math.floor(start / step) * step;
      stop = Math.ceil(stop / step) * step;
      step = tickIncrement(start, stop, count);
    } else if (step < 0) {
      start = Math.ceil(start * step) / step;
      stop = Math.floor(stop * step) / step;
      step = tickIncrement(start, stop, count);
    }

    if (step > 0) {
      d[i0] = Math.floor(start / step) * step;
      d[i1] = Math.ceil(stop / step) * step;
      domain(d);
    } else if (step < 0) {
      d[i0] = Math.ceil(start * step) / step;
      d[i1] = Math.floor(stop * step) / step;
      domain(d);
    }

    return scale;
  };

  return scale;
}

function linear$2() {
  var scale = continuous(deinterpolateLinear, reinterpolate);

  scale.copy = function() {
    return copy(scale, linear$2());
  };

  return linearish(scale);
}

function deinterpolate(a, b) {
  return (b = Math.log(b / a))
      ? function(x) { return Math.log(x / a) / b; }
      : constant$9(b);
}

function reinterpolate$1(a, b) {
  return a < 0
      ? function(t) { return -Math.pow(-b, t) * Math.pow(-a, 1 - t); }
      : function(t) { return Math.pow(b, t) * Math.pow(a, 1 - t); };
}

function pow10(x) {
  return isFinite(x) ? +("1e" + x) : x < 0 ? 0 : x;
}

function powp(base) {
  return base === 10 ? pow10
      : base === Math.E ? Math.exp
      : function(x) { return Math.pow(base, x); };
}

function logp(base) {
  return base === Math.E ? Math.log
      : base === 10 && Math.log10
      || base === 2 && Math.log2
      || (base = Math.log(base), function(x) { return Math.log(x) / base; });
}

var t0$1 = new Date;
var t1$1 = new Date;

function newInterval(floori, offseti, count, field) {

  function interval(date) {
    return floori(date = new Date(+date)), date;
  }

  interval.floor = interval;

  interval.ceil = function(date) {
    return floori(date = new Date(date - 1)), offseti(date, 1), floori(date), date;
  };

  interval.round = function(date) {
    var d0 = interval(date),
        d1 = interval.ceil(date);
    return date - d0 < d1 - date ? d0 : d1;
  };

  interval.offset = function(date, step) {
    return offseti(date = new Date(+date), step == null ? 1 : Math.floor(step)), date;
  };

  interval.range = function(start, stop, step) {
    var range = [];
    start = interval.ceil(start);
    step = step == null ? 1 : Math.floor(step);
    if (!(start < stop) || !(step > 0)) return range; // also handles Invalid Date
    do range.push(new Date(+start)); while (offseti(start, step), floori(start), start < stop)
    return range;
  };

  interval.filter = function(test) {
    return newInterval(function(date) {
      if (date >= date) while (floori(date), !test(date)) date.setTime(date - 1);
    }, function(date, step) {
      if (date >= date) while (--step >= 0) while (offseti(date, 1), !test(date)) {} // eslint-disable-line no-empty
    });
  };

  if (count) {
    interval.count = function(start, end) {
      t0$1.setTime(+start), t1$1.setTime(+end);
      floori(t0$1), floori(t1$1);
      return Math.floor(count(t0$1, t1$1));
    };

    interval.every = function(step) {
      step = Math.floor(step);
      return !isFinite(step) || !(step > 0) ? null
          : !(step > 1) ? interval
          : interval.filter(field
              ? function(d) { return field(d) % step === 0; }
              : function(d) { return interval.count(0, d) % step === 0; });
    };
  }

  return interval;
}

var millisecond = newInterval(function() {
  // noop
}, function(date, step) {
  date.setTime(+date + step);
}, function(start, end) {
  return end - start;
});

// An optimized implementation for this simple case.
millisecond.every = function(k) {
  k = Math.floor(k);
  if (!isFinite(k) || !(k > 0)) return null;
  if (!(k > 1)) return millisecond;
  return newInterval(function(date) {
    date.setTime(Math.floor(date / k) * k);
  }, function(date, step) {
    date.setTime(+date + step * k);
  }, function(start, end) {
    return (end - start) / k;
  });
};

var durationSecond$1 = 1e3;
var durationMinute$1 = 6e4;
var durationHour$1 = 36e5;
var durationDay$1 = 864e5;
var durationWeek$1 = 6048e5;

var second = newInterval(function(date) {
  date.setTime(Math.floor(date / durationSecond$1) * durationSecond$1);
}, function(date, step) {
  date.setTime(+date + step * durationSecond$1);
}, function(start, end) {
  return (end - start) / durationSecond$1;
}, function(date) {
  return date.getUTCSeconds();
});

var minute = newInterval(function(date) {
  date.setTime(Math.floor(date / durationMinute$1) * durationMinute$1);
}, function(date, step) {
  date.setTime(+date + step * durationMinute$1);
}, function(start, end) {
  return (end - start) / durationMinute$1;
}, function(date) {
  return date.getMinutes();
});

var hour = newInterval(function(date) {
  var offset = date.getTimezoneOffset() * durationMinute$1 % durationHour$1;
  if (offset < 0) offset += durationHour$1;
  date.setTime(Math.floor((+date - offset) / durationHour$1) * durationHour$1 + offset);
}, function(date, step) {
  date.setTime(+date + step * durationHour$1);
}, function(start, end) {
  return (end - start) / durationHour$1;
}, function(date) {
  return date.getHours();
});

var day = newInterval(function(date) {
  date.setHours(0, 0, 0, 0);
}, function(date, step) {
  date.setDate(date.getDate() + step);
}, function(start, end) {
  return (end - start - (end.getTimezoneOffset() - start.getTimezoneOffset()) * durationMinute$1) / durationDay$1;
}, function(date) {
  return date.getDate() - 1;
});

function weekday(i) {
  return newInterval(function(date) {
    date.setDate(date.getDate() - (date.getDay() + 7 - i) % 7);
    date.setHours(0, 0, 0, 0);
  }, function(date, step) {
    date.setDate(date.getDate() + step * 7);
  }, function(start, end) {
    return (end - start - (end.getTimezoneOffset() - start.getTimezoneOffset()) * durationMinute$1) / durationWeek$1;
  });
}

var sunday = weekday(0);
var monday = weekday(1);
var tuesday = weekday(2);
var wednesday = weekday(3);
var thursday = weekday(4);
var friday = weekday(5);
var saturday = weekday(6);

var month = newInterval(function(date) {
  date.setDate(1);
  date.setHours(0, 0, 0, 0);
}, function(date, step) {
  date.setMonth(date.getMonth() + step);
}, function(start, end) {
  return end.getMonth() - start.getMonth() + (end.getFullYear() - start.getFullYear()) * 12;
}, function(date) {
  return date.getMonth();
});

var year = newInterval(function(date) {
  date.setMonth(0, 1);
  date.setHours(0, 0, 0, 0);
}, function(date, step) {
  date.setFullYear(date.getFullYear() + step);
}, function(start, end) {
  return end.getFullYear() - start.getFullYear();
}, function(date) {
  return date.getFullYear();
});

// An optimized implementation for this simple case.
year.every = function(k) {
  return !isFinite(k = Math.floor(k)) || !(k > 0) ? null : newInterval(function(date) {
    date.setFullYear(Math.floor(date.getFullYear() / k) * k);
    date.setMonth(0, 1);
    date.setHours(0, 0, 0, 0);
  }, function(date, step) {
    date.setFullYear(date.getFullYear() + step * k);
  });
};

var utcMinute = newInterval(function(date) {
  date.setUTCSeconds(0, 0);
}, function(date, step) {
  date.setTime(+date + step * durationMinute$1);
}, function(start, end) {
  return (end - start) / durationMinute$1;
}, function(date) {
  return date.getUTCMinutes();
});

var utcHour = newInterval(function(date) {
  date.setUTCMinutes(0, 0, 0);
}, function(date, step) {
  date.setTime(+date + step * durationHour$1);
}, function(start, end) {
  return (end - start) / durationHour$1;
}, function(date) {
  return date.getUTCHours();
});

var utcDay = newInterval(function(date) {
  date.setUTCHours(0, 0, 0, 0);
}, function(date, step) {
  date.setUTCDate(date.getUTCDate() + step);
}, function(start, end) {
  return (end - start) / durationDay$1;
}, function(date) {
  return date.getUTCDate() - 1;
});

function utcWeekday(i) {
  return newInterval(function(date) {
    date.setUTCDate(date.getUTCDate() - (date.getUTCDay() + 7 - i) % 7);
    date.setUTCHours(0, 0, 0, 0);
  }, function(date, step) {
    date.setUTCDate(date.getUTCDate() + step * 7);
  }, function(start, end) {
    return (end - start) / durationWeek$1;
  });
}

var utcSunday = utcWeekday(0);
var utcMonday = utcWeekday(1);
var utcTuesday = utcWeekday(2);
var utcWednesday = utcWeekday(3);
var utcThursday = utcWeekday(4);
var utcFriday = utcWeekday(5);
var utcSaturday = utcWeekday(6);

var utcMonth = newInterval(function(date) {
  date.setUTCDate(1);
  date.setUTCHours(0, 0, 0, 0);
}, function(date, step) {
  date.setUTCMonth(date.getUTCMonth() + step);
}, function(start, end) {
  return end.getUTCMonth() - start.getUTCMonth() + (end.getUTCFullYear() - start.getUTCFullYear()) * 12;
}, function(date) {
  return date.getUTCMonth();
});

var utcYear = newInterval(function(date) {
  date.setUTCMonth(0, 1);
  date.setUTCHours(0, 0, 0, 0);
}, function(date, step) {
  date.setUTCFullYear(date.getUTCFullYear() + step);
}, function(start, end) {
  return end.getUTCFullYear() - start.getUTCFullYear();
}, function(date) {
  return date.getUTCFullYear();
});

// An optimized implementation for this simple case.
utcYear.every = function(k) {
  return !isFinite(k = Math.floor(k)) || !(k > 0) ? null : newInterval(function(date) {
    date.setUTCFullYear(Math.floor(date.getUTCFullYear() / k) * k);
    date.setUTCMonth(0, 1);
    date.setUTCHours(0, 0, 0, 0);
  }, function(date, step) {
    date.setUTCFullYear(date.getUTCFullYear() + step * k);
  });
};

function localDate(d) {
  if (0 <= d.y && d.y < 100) {
    var date = new Date(-1, d.m, d.d, d.H, d.M, d.S, d.L);
    date.setFullYear(d.y);
    return date;
  }
  return new Date(d.y, d.m, d.d, d.H, d.M, d.S, d.L);
}

function utcDate(d) {
  if (0 <= d.y && d.y < 100) {
    var date = new Date(Date.UTC(-1, d.m, d.d, d.H, d.M, d.S, d.L));
    date.setUTCFullYear(d.y);
    return date;
  }
  return new Date(Date.UTC(d.y, d.m, d.d, d.H, d.M, d.S, d.L));
}

function newYear(y) {
  return {y: y, m: 0, d: 1, H: 0, M: 0, S: 0, L: 0};
}

function formatLocale$1(locale) {
  var locale_dateTime = locale.dateTime,
      locale_date = locale.date,
      locale_time = locale.time,
      locale_periods = locale.periods,
      locale_weekdays = locale.days,
      locale_shortWeekdays = locale.shortDays,
      locale_months = locale.months,
      locale_shortMonths = locale.shortMonths;

  var periodRe = formatRe(locale_periods),
      periodLookup = formatLookup(locale_periods),
      weekdayRe = formatRe(locale_weekdays),
      weekdayLookup = formatLookup(locale_weekdays),
      shortWeekdayRe = formatRe(locale_shortWeekdays),
      shortWeekdayLookup = formatLookup(locale_shortWeekdays),
      monthRe = formatRe(locale_months),
      monthLookup = formatLookup(locale_months),
      shortMonthRe = formatRe(locale_shortMonths),
      shortMonthLookup = formatLookup(locale_shortMonths);

  var formats = {
    "a": formatShortWeekday,
    "A": formatWeekday,
    "b": formatShortMonth,
    "B": formatMonth,
    "c": null,
    "d": formatDayOfMonth,
    "e": formatDayOfMonth,
    "H": formatHour24,
    "I": formatHour12,
    "j": formatDayOfYear,
    "L": formatMilliseconds,
    "m": formatMonthNumber,
    "M": formatMinutes,
    "p": formatPeriod,
    "S": formatSeconds,
    "U": formatWeekNumberSunday,
    "w": formatWeekdayNumber,
    "W": formatWeekNumberMonday,
    "x": null,
    "X": null,
    "y": formatYear,
    "Y": formatFullYear,
    "Z": formatZone,
    "%": formatLiteralPercent
  };

  var utcFormats = {
    "a": formatUTCShortWeekday,
    "A": formatUTCWeekday,
    "b": formatUTCShortMonth,
    "B": formatUTCMonth,
    "c": null,
    "d": formatUTCDayOfMonth,
    "e": formatUTCDayOfMonth,
    "H": formatUTCHour24,
    "I": formatUTCHour12,
    "j": formatUTCDayOfYear,
    "L": formatUTCMilliseconds,
    "m": formatUTCMonthNumber,
    "M": formatUTCMinutes,
    "p": formatUTCPeriod,
    "S": formatUTCSeconds,
    "U": formatUTCWeekNumberSunday,
    "w": formatUTCWeekdayNumber,
    "W": formatUTCWeekNumberMonday,
    "x": null,
    "X": null,
    "y": formatUTCYear,
    "Y": formatUTCFullYear,
    "Z": formatUTCZone,
    "%": formatLiteralPercent
  };

  var parses = {
    "a": parseShortWeekday,
    "A": parseWeekday,
    "b": parseShortMonth,
    "B": parseMonth,
    "c": parseLocaleDateTime,
    "d": parseDayOfMonth,
    "e": parseDayOfMonth,
    "H": parseHour24,
    "I": parseHour24,
    "j": parseDayOfYear,
    "L": parseMilliseconds,
    "m": parseMonthNumber,
    "M": parseMinutes,
    "p": parsePeriod,
    "S": parseSeconds,
    "U": parseWeekNumberSunday,
    "w": parseWeekdayNumber,
    "W": parseWeekNumberMonday,
    "x": parseLocaleDate,
    "X": parseLocaleTime,
    "y": parseYear,
    "Y": parseFullYear,
    "Z": parseZone,
    "%": parseLiteralPercent
  };

  // These recursive directive definitions must be deferred.
  formats.x = newFormat(locale_date, formats);
  formats.X = newFormat(locale_time, formats);
  formats.c = newFormat(locale_dateTime, formats);
  utcFormats.x = newFormat(locale_date, utcFormats);
  utcFormats.X = newFormat(locale_time, utcFormats);
  utcFormats.c = newFormat(locale_dateTime, utcFormats);

  function newFormat(specifier, formats) {
    return function(date) {
      var string = [],
          i = -1,
          j = 0,
          n = specifier.length,
          c,
          pad,
          format;

      if (!(date instanceof Date)) date = new Date(+date);

      while (++i < n) {
        if (specifier.charCodeAt(i) === 37) {
          string.push(specifier.slice(j, i));
          if ((pad = pads[c = specifier.charAt(++i)]) != null) c = specifier.charAt(++i);
          else pad = c === "e" ? " " : "0";
          if (format = formats[c]) c = format(date, pad);
          string.push(c);
          j = i + 1;
        }
      }

      string.push(specifier.slice(j, i));
      return string.join("");
    };
  }

  function newParse(specifier, newDate) {
    return function(string) {
      var d = newYear(1900),
          i = parseSpecifier(d, specifier, string += "", 0);
      if (i != string.length) return null;

      // The am-pm flag is 0 for AM, and 1 for PM.
      if ("p" in d) d.H = d.H % 12 + d.p * 12;

      // Convert day-of-week and week-of-year to day-of-year.
      if ("W" in d || "U" in d) {
        if (!("w" in d)) d.w = "W" in d ? 1 : 0;
        var day$$1 = "Z" in d ? utcDate(newYear(d.y)).getUTCDay() : newDate(newYear(d.y)).getDay();
        d.m = 0;
        d.d = "W" in d ? (d.w + 6) % 7 + d.W * 7 - (day$$1 + 5) % 7 : d.w + d.U * 7 - (day$$1 + 6) % 7;
      }

      // If a time zone is specified, all fields are interpreted as UTC and then
      // offset according to the specified time zone.
      if ("Z" in d) {
        d.H += d.Z / 100 | 0;
        d.M += d.Z % 100;
        return utcDate(d);
      }

      // Otherwise, all fields are in local time.
      return newDate(d);
    };
  }

  function parseSpecifier(d, specifier, string, j) {
    var i = 0,
        n = specifier.length,
        m = string.length,
        c,
        parse;

    while (i < n) {
      if (j >= m) return -1;
      c = specifier.charCodeAt(i++);
      if (c === 37) {
        c = specifier.charAt(i++);
        parse = parses[c in pads ? specifier.charAt(i++) : c];
        if (!parse || ((j = parse(d, string, j)) < 0)) return -1;
      } else if (c != string.charCodeAt(j++)) {
        return -1;
      }
    }

    return j;
  }

  function parsePeriod(d, string, i) {
    var n = periodRe.exec(string.slice(i));
    return n ? (d.p = periodLookup[n[0].toLowerCase()], i + n[0].length) : -1;
  }

  function parseShortWeekday(d, string, i) {
    var n = shortWeekdayRe.exec(string.slice(i));
    return n ? (d.w = shortWeekdayLookup[n[0].toLowerCase()], i + n[0].length) : -1;
  }

  function parseWeekday(d, string, i) {
    var n = weekdayRe.exec(string.slice(i));
    return n ? (d.w = weekdayLookup[n[0].toLowerCase()], i + n[0].length) : -1;
  }

  function parseShortMonth(d, string, i) {
    var n = shortMonthRe.exec(string.slice(i));
    return n ? (d.m = shortMonthLookup[n[0].toLowerCase()], i + n[0].length) : -1;
  }

  function parseMonth(d, string, i) {
    var n = monthRe.exec(string.slice(i));
    return n ? (d.m = monthLookup[n[0].toLowerCase()], i + n[0].length) : -1;
  }

  function parseLocaleDateTime(d, string, i) {
    return parseSpecifier(d, locale_dateTime, string, i);
  }

  function parseLocaleDate(d, string, i) {
    return parseSpecifier(d, locale_date, string, i);
  }

  function parseLocaleTime(d, string, i) {
    return parseSpecifier(d, locale_time, string, i);
  }

  function formatShortWeekday(d) {
    return locale_shortWeekdays[d.getDay()];
  }

  function formatWeekday(d) {
    return locale_weekdays[d.getDay()];
  }

  function formatShortMonth(d) {
    return locale_shortMonths[d.getMonth()];
  }

  function formatMonth(d) {
    return locale_months[d.getMonth()];
  }

  function formatPeriod(d) {
    return locale_periods[+(d.getHours() >= 12)];
  }

  function formatUTCShortWeekday(d) {
    return locale_shortWeekdays[d.getUTCDay()];
  }

  function formatUTCWeekday(d) {
    return locale_weekdays[d.getUTCDay()];
  }

  function formatUTCShortMonth(d) {
    return locale_shortMonths[d.getUTCMonth()];
  }

  function formatUTCMonth(d) {
    return locale_months[d.getUTCMonth()];
  }

  function formatUTCPeriod(d) {
    return locale_periods[+(d.getUTCHours() >= 12)];
  }

  return {
    format: function(specifier) {
      var f = newFormat(specifier += "", formats);
      f.toString = function() { return specifier; };
      return f;
    },
    parse: function(specifier) {
      var p = newParse(specifier += "", localDate);
      p.toString = function() { return specifier; };
      return p;
    },
    utcFormat: function(specifier) {
      var f = newFormat(specifier += "", utcFormats);
      f.toString = function() { return specifier; };
      return f;
    },
    utcParse: function(specifier) {
      var p = newParse(specifier, utcDate);
      p.toString = function() { return specifier; };
      return p;
    }
  };
}

var pads = {"-": "", "_": " ", "0": "0"};
var numberRe = /^\s*\d+/;
var percentRe = /^%/;
var requoteRe = /[\\\^\$\*\+\?\|\[\]\(\)\.\{\}]/g;

function pad(value, fill, width) {
  var sign = value < 0 ? "-" : "",
      string = (sign ? -value : value) + "",
      length = string.length;
  return sign + (length < width ? new Array(width - length + 1).join(fill) + string : string);
}

function requote(s) {
  return s.replace(requoteRe, "\\$&");
}

function formatRe(names) {
  return new RegExp("^(?:" + names.map(requote).join("|") + ")", "i");
}

function formatLookup(names) {
  var map = {}, i = -1, n = names.length;
  while (++i < n) map[names[i].toLowerCase()] = i;
  return map;
}

function parseWeekdayNumber(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 1));
  return n ? (d.w = +n[0], i + n[0].length) : -1;
}

function parseWeekNumberSunday(d, string, i) {
  var n = numberRe.exec(string.slice(i));
  return n ? (d.U = +n[0], i + n[0].length) : -1;
}

function parseWeekNumberMonday(d, string, i) {
  var n = numberRe.exec(string.slice(i));
  return n ? (d.W = +n[0], i + n[0].length) : -1;
}

function parseFullYear(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 4));
  return n ? (d.y = +n[0], i + n[0].length) : -1;
}

function parseYear(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 2));
  return n ? (d.y = +n[0] + (+n[0] > 68 ? 1900 : 2000), i + n[0].length) : -1;
}

function parseZone(d, string, i) {
  var n = /^(Z)|([+-]\d\d)(?:\:?(\d\d))?/.exec(string.slice(i, i + 6));
  return n ? (d.Z = n[1] ? 0 : -(n[2] + (n[3] || "00")), i + n[0].length) : -1;
}

function parseMonthNumber(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 2));
  return n ? (d.m = n[0] - 1, i + n[0].length) : -1;
}

function parseDayOfMonth(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 2));
  return n ? (d.d = +n[0], i + n[0].length) : -1;
}

function parseDayOfYear(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 3));
  return n ? (d.m = 0, d.d = +n[0], i + n[0].length) : -1;
}

function parseHour24(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 2));
  return n ? (d.H = +n[0], i + n[0].length) : -1;
}

function parseMinutes(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 2));
  return n ? (d.M = +n[0], i + n[0].length) : -1;
}

function parseSeconds(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 2));
  return n ? (d.S = +n[0], i + n[0].length) : -1;
}

function parseMilliseconds(d, string, i) {
  var n = numberRe.exec(string.slice(i, i + 3));
  return n ? (d.L = +n[0], i + n[0].length) : -1;
}

function parseLiteralPercent(d, string, i) {
  var n = percentRe.exec(string.slice(i, i + 1));
  return n ? i + n[0].length : -1;
}

function formatDayOfMonth(d, p) {
  return pad(d.getDate(), p, 2);
}

function formatHour24(d, p) {
  return pad(d.getHours(), p, 2);
}

function formatHour12(d, p) {
  return pad(d.getHours() % 12 || 12, p, 2);
}

function formatDayOfYear(d, p) {
  return pad(1 + day.count(year(d), d), p, 3);
}

function formatMilliseconds(d, p) {
  return pad(d.getMilliseconds(), p, 3);
}

function formatMonthNumber(d, p) {
  return pad(d.getMonth() + 1, p, 2);
}

function formatMinutes(d, p) {
  return pad(d.getMinutes(), p, 2);
}

function formatSeconds(d, p) {
  return pad(d.getSeconds(), p, 2);
}

function formatWeekNumberSunday(d, p) {
  return pad(sunday.count(year(d), d), p, 2);
}

function formatWeekdayNumber(d) {
  return d.getDay();
}

function formatWeekNumberMonday(d, p) {
  return pad(monday.count(year(d), d), p, 2);
}

function formatYear(d, p) {
  return pad(d.getFullYear() % 100, p, 2);
}

function formatFullYear(d, p) {
  return pad(d.getFullYear() % 10000, p, 4);
}

function formatZone(d) {
  var z = d.getTimezoneOffset();
  return (z > 0 ? "-" : (z *= -1, "+"))
      + pad(z / 60 | 0, "0", 2)
      + pad(z % 60, "0", 2);
}

function formatUTCDayOfMonth(d, p) {
  return pad(d.getUTCDate(), p, 2);
}

function formatUTCHour24(d, p) {
  return pad(d.getUTCHours(), p, 2);
}

function formatUTCHour12(d, p) {
  return pad(d.getUTCHours() % 12 || 12, p, 2);
}

function formatUTCDayOfYear(d, p) {
  return pad(1 + utcDay.count(utcYear(d), d), p, 3);
}

function formatUTCMilliseconds(d, p) {
  return pad(d.getUTCMilliseconds(), p, 3);
}

function formatUTCMonthNumber(d, p) {
  return pad(d.getUTCMonth() + 1, p, 2);
}

function formatUTCMinutes(d, p) {
  return pad(d.getUTCMinutes(), p, 2);
}

function formatUTCSeconds(d, p) {
  return pad(d.getUTCSeconds(), p, 2);
}

function formatUTCWeekNumberSunday(d, p) {
  return pad(utcSunday.count(utcYear(d), d), p, 2);
}

function formatUTCWeekdayNumber(d) {
  return d.getUTCDay();
}

function formatUTCWeekNumberMonday(d, p) {
  return pad(utcMonday.count(utcYear(d), d), p, 2);
}

function formatUTCYear(d, p) {
  return pad(d.getUTCFullYear() % 100, p, 2);
}

function formatUTCFullYear(d, p) {
  return pad(d.getUTCFullYear() % 10000, p, 4);
}

function formatUTCZone() {
  return "+0000";
}

function formatLiteralPercent() {
  return "%";
}

var locale$2;
var timeFormat;
var timeParse;
var utcFormat;
var utcParse;

defaultLocale$1({
  dateTime: "%x, %X",
  date: "%-m/%-d/%Y",
  time: "%-I:%M:%S %p",
  periods: ["AM", "PM"],
  days: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
  shortDays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
  months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
  shortMonths: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
});

function defaultLocale$1(definition) {
  locale$2 = formatLocale$1(definition);
  timeFormat = locale$2.format;
  timeParse = locale$2.parse;
  utcFormat = locale$2.utcFormat;
  utcParse = locale$2.utcParse;
  return locale$2;
}

var isoSpecifier = "%Y-%m-%dT%H:%M:%S.%LZ";

function formatIsoNative(date) {
  return date.toISOString();
}

var formatIso = Date.prototype.toISOString
    ? formatIsoNative
    : utcFormat(isoSpecifier);

function parseIsoNative(string) {
  var date = new Date(string);
  return isNaN(date) ? null : date;
}

var parseIso = +new Date("2000-01-01T00:00:00.000Z")
    ? parseIsoNative
    : utcParse(isoSpecifier);

var colors = function(s) {
  return s.match(/.{6}/g).map(function(x) {
    return "#" + x;
  });
};

colors("1f77b4ff7f0e2ca02cd627289467bd8c564be377c27f7f7fbcbd2217becf");

colors("393b795254a36b6ecf9c9ede6379398ca252b5cf6bcedb9c8c6d31bd9e39e7ba52e7cb94843c39ad494ad6616be7969c7b4173a55194ce6dbdde9ed6");

colors("3182bd6baed69ecae1c6dbefe6550dfd8d3cfdae6bfdd0a231a35474c476a1d99bc7e9c0756bb19e9ac8bcbddcdadaeb636363969696bdbdbdd9d9d9");

colors("1f77b4aec7e8ff7f0effbb782ca02c98df8ad62728ff98969467bdc5b0d58c564bc49c94e377c2f7b6d27f7f7fc7c7c7bcbd22dbdb8d17becf9edae5");

cubehelixLong(cubehelix(300, 0.5, 0.0), cubehelix(-240, 0.5, 1.0));

var warm = cubehelixLong(cubehelix(-100, 0.75, 0.35), cubehelix(80, 1.50, 0.8));

var cool = cubehelixLong(cubehelix(260, 0.75, 0.35), cubehelix(80, 1.50, 0.8));

var rainbow = cubehelix();

var constant$10 = function(x) {
  return function constant() {
    return x;
  };
};

var abs$1 = Math.abs;
var atan2$1 = Math.atan2;
var cos$2 = Math.cos;
var max$2 = Math.max;
var min$1 = Math.min;
var sin$2 = Math.sin;
var sqrt$2 = Math.sqrt;

var epsilon$3 = 1e-12;
var pi$4 = Math.PI;
var halfPi$3 = pi$4 / 2;
var tau$4 = 2 * pi$4;

function acos$1(x) {
  return x > 1 ? 0 : x < -1 ? pi$4 : Math.acos(x);
}

function asin$1(x) {
  return x >= 1 ? halfPi$3 : x <= -1 ? -halfPi$3 : Math.asin(x);
}

function arcInnerRadius(d) {
  return d.innerRadius;
}

function arcOuterRadius(d) {
  return d.outerRadius;
}

function arcStartAngle(d) {
  return d.startAngle;
}

function arcEndAngle(d) {
  return d.endAngle;
}

function arcPadAngle(d) {
  return d && d.padAngle; // Note: optional!
}

function intersect(x0, y0, x1, y1, x2, y2, x3, y3) {
  var x10 = x1 - x0, y10 = y1 - y0,
      x32 = x3 - x2, y32 = y3 - y2,
      t = (x32 * (y0 - y2) - y32 * (x0 - x2)) / (y32 * x10 - x32 * y10);
  return [x0 + t * x10, y0 + t * y10];
}

// Compute perpendicular offset line of length rc.
// http://mathworld.wolfram.com/Circle-LineIntersection.html
function cornerTangents(x0, y0, x1, y1, r1, rc, cw) {
  var x01 = x0 - x1,
      y01 = y0 - y1,
      lo = (cw ? rc : -rc) / sqrt$2(x01 * x01 + y01 * y01),
      ox = lo * y01,
      oy = -lo * x01,
      x11 = x0 + ox,
      y11 = y0 + oy,
      x10 = x1 + ox,
      y10 = y1 + oy,
      x00 = (x11 + x10) / 2,
      y00 = (y11 + y10) / 2,
      dx = x10 - x11,
      dy = y10 - y11,
      d2 = dx * dx + dy * dy,
      r = r1 - rc,
      D = x11 * y10 - x10 * y11,
      d = (dy < 0 ? -1 : 1) * sqrt$2(max$2(0, r * r * d2 - D * D)),
      cx0 = (D * dy - dx * d) / d2,
      cy0 = (-D * dx - dy * d) / d2,
      cx1 = (D * dy + dx * d) / d2,
      cy1 = (-D * dx + dy * d) / d2,
      dx0 = cx0 - x00,
      dy0 = cy0 - y00,
      dx1 = cx1 - x00,
      dy1 = cy1 - y00;

  // Pick the closer of the two intersection points.
  // TODO Is there a faster way to determine which intersection to use?
  if (dx0 * dx0 + dy0 * dy0 > dx1 * dx1 + dy1 * dy1) cx0 = cx1, cy0 = cy1;

  return {
    cx: cx0,
    cy: cy0,
    x01: -ox,
    y01: -oy,
    x11: cx0 * (r1 / r - 1),
    y11: cy0 * (r1 / r - 1)
  };
}

var arc = function() {
  var innerRadius = arcInnerRadius,
      outerRadius = arcOuterRadius,
      cornerRadius = constant$10(0),
      padRadius = null,
      startAngle = arcStartAngle,
      endAngle = arcEndAngle,
      padAngle = arcPadAngle,
      context = null;

  function arc() {
    var buffer,
        r,
        r0 = +innerRadius.apply(this, arguments),
        r1 = +outerRadius.apply(this, arguments),
        a0 = startAngle.apply(this, arguments) - halfPi$3,
        a1 = endAngle.apply(this, arguments) - halfPi$3,
        da = abs$1(a1 - a0),
        cw = a1 > a0;

    if (!context) context = buffer = path();

    // Ensure that the outer radius is always larger than the inner radius.
    if (r1 < r0) r = r1, r1 = r0, r0 = r;

    // Is it a point?
    if (!(r1 > epsilon$3)) context.moveTo(0, 0);

    // Or is it a circle or annulus?
    else if (da > tau$4 - epsilon$3) {
      context.moveTo(r1 * cos$2(a0), r1 * sin$2(a0));
      context.arc(0, 0, r1, a0, a1, !cw);
      if (r0 > epsilon$3) {
        context.moveTo(r0 * cos$2(a1), r0 * sin$2(a1));
        context.arc(0, 0, r0, a1, a0, cw);
      }
    }

    // Or is it a circular or annular sector?
    else {
      var a01 = a0,
          a11 = a1,
          a00 = a0,
          a10 = a1,
          da0 = da,
          da1 = da,
          ap = padAngle.apply(this, arguments) / 2,
          rp = (ap > epsilon$3) && (padRadius ? +padRadius.apply(this, arguments) : sqrt$2(r0 * r0 + r1 * r1)),
          rc = min$1(abs$1(r1 - r0) / 2, +cornerRadius.apply(this, arguments)),
          rc0 = rc,
          rc1 = rc,
          t0,
          t1;

      // Apply padding? Note that since r1 ≥ r0, da1 ≥ da0.
      if (rp > epsilon$3) {
        var p0 = asin$1(rp / r0 * sin$2(ap)),
            p1 = asin$1(rp / r1 * sin$2(ap));
        if ((da0 -= p0 * 2) > epsilon$3) p0 *= (cw ? 1 : -1), a00 += p0, a10 -= p0;
        else da0 = 0, a00 = a10 = (a0 + a1) / 2;
        if ((da1 -= p1 * 2) > epsilon$3) p1 *= (cw ? 1 : -1), a01 += p1, a11 -= p1;
        else da1 = 0, a01 = a11 = (a0 + a1) / 2;
      }

      var x01 = r1 * cos$2(a01),
          y01 = r1 * sin$2(a01),
          x10 = r0 * cos$2(a10),
          y10 = r0 * sin$2(a10);

      // Apply rounded corners?
      if (rc > epsilon$3) {
        var x11 = r1 * cos$2(a11),
            y11 = r1 * sin$2(a11),
            x00 = r0 * cos$2(a00),
            y00 = r0 * sin$2(a00);

        // Restrict the corner radius according to the sector angle.
        if (da < pi$4) {
          var oc = da0 > epsilon$3 ? intersect(x01, y01, x00, y00, x11, y11, x10, y10) : [x10, y10],
              ax = x01 - oc[0],
              ay = y01 - oc[1],
              bx = x11 - oc[0],
              by = y11 - oc[1],
              kc = 1 / sin$2(acos$1((ax * bx + ay * by) / (sqrt$2(ax * ax + ay * ay) * sqrt$2(bx * bx + by * by))) / 2),
              lc = sqrt$2(oc[0] * oc[0] + oc[1] * oc[1]);
          rc0 = min$1(rc, (r0 - lc) / (kc - 1));
          rc1 = min$1(rc, (r1 - lc) / (kc + 1));
        }
      }

      // Is the sector collapsed to a line?
      if (!(da1 > epsilon$3)) context.moveTo(x01, y01);

      // Does the sector’s outer ring have rounded corners?
      else if (rc1 > epsilon$3) {
        t0 = cornerTangents(x00, y00, x01, y01, r1, rc1, cw);
        t1 = cornerTangents(x11, y11, x10, y10, r1, rc1, cw);

        context.moveTo(t0.cx + t0.x01, t0.cy + t0.y01);

        // Have the corners merged?
        if (rc1 < rc) context.arc(t0.cx, t0.cy, rc1, atan2$1(t0.y01, t0.x01), atan2$1(t1.y01, t1.x01), !cw);

        // Otherwise, draw the two corners and the ring.
        else {
          context.arc(t0.cx, t0.cy, rc1, atan2$1(t0.y01, t0.x01), atan2$1(t0.y11, t0.x11), !cw);
          context.arc(0, 0, r1, atan2$1(t0.cy + t0.y11, t0.cx + t0.x11), atan2$1(t1.cy + t1.y11, t1.cx + t1.x11), !cw);
          context.arc(t1.cx, t1.cy, rc1, atan2$1(t1.y11, t1.x11), atan2$1(t1.y01, t1.x01), !cw);
        }
      }

      // Or is the outer ring just a circular arc?
      else context.moveTo(x01, y01), context.arc(0, 0, r1, a01, a11, !cw);

      // Is there no inner ring, and it’s a circular sector?
      // Or perhaps it’s an annular sector collapsed due to padding?
      if (!(r0 > epsilon$3) || !(da0 > epsilon$3)) context.lineTo(x10, y10);

      // Does the sector’s inner ring (or point) have rounded corners?
      else if (rc0 > epsilon$3) {
        t0 = cornerTangents(x10, y10, x11, y11, r0, -rc0, cw);
        t1 = cornerTangents(x01, y01, x00, y00, r0, -rc0, cw);

        context.lineTo(t0.cx + t0.x01, t0.cy + t0.y01);

        // Have the corners merged?
        if (rc0 < rc) context.arc(t0.cx, t0.cy, rc0, atan2$1(t0.y01, t0.x01), atan2$1(t1.y01, t1.x01), !cw);

        // Otherwise, draw the two corners and the ring.
        else {
          context.arc(t0.cx, t0.cy, rc0, atan2$1(t0.y01, t0.x01), atan2$1(t0.y11, t0.x11), !cw);
          context.arc(0, 0, r0, atan2$1(t0.cy + t0.y11, t0.cx + t0.x11), atan2$1(t1.cy + t1.y11, t1.cx + t1.x11), cw);
          context.arc(t1.cx, t1.cy, rc0, atan2$1(t1.y11, t1.x11), atan2$1(t1.y01, t1.x01), !cw);
        }
      }

      // Or is the inner ring just a circular arc?
      else context.arc(0, 0, r0, a10, a00, cw);
    }

    context.closePath();

    if (buffer) return context = null, buffer + "" || null;
  }

  arc.centroid = function() {
    var r = (+innerRadius.apply(this, arguments) + +outerRadius.apply(this, arguments)) / 2,
        a = (+startAngle.apply(this, arguments) + +endAngle.apply(this, arguments)) / 2 - pi$4 / 2;
    return [cos$2(a) * r, sin$2(a) * r];
  };

  arc.innerRadius = function(_) {
    return arguments.length ? (innerRadius = typeof _ === "function" ? _ : constant$10(+_), arc) : innerRadius;
  };

  arc.outerRadius = function(_) {
    return arguments.length ? (outerRadius = typeof _ === "function" ? _ : constant$10(+_), arc) : outerRadius;
  };

  arc.cornerRadius = function(_) {
    return arguments.length ? (cornerRadius = typeof _ === "function" ? _ : constant$10(+_), arc) : cornerRadius;
  };

  arc.padRadius = function(_) {
    return arguments.length ? (padRadius = _ == null ? null : typeof _ === "function" ? _ : constant$10(+_), arc) : padRadius;
  };

  arc.startAngle = function(_) {
    return arguments.length ? (startAngle = typeof _ === "function" ? _ : constant$10(+_), arc) : startAngle;
  };

  arc.endAngle = function(_) {
    return arguments.length ? (endAngle = typeof _ === "function" ? _ : constant$10(+_), arc) : endAngle;
  };

  arc.padAngle = function(_) {
    return arguments.length ? (padAngle = typeof _ === "function" ? _ : constant$10(+_), arc) : padAngle;
  };

  arc.context = function(_) {
    return arguments.length ? ((context = _ == null ? null : _), arc) : context;
  };

  return arc;
};

function Linear(context) {
  this._context = context;
}

Linear.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._point = 0;
  },
  lineEnd: function() {
    if (this._line || (this._line !== 0 && this._point === 1)) this._context.closePath();
    this._line = 1 - this._line;
  },
  point: function(x, y) {
    x = +x, y = +y;
    switch (this._point) {
      case 0: this._point = 1; this._line ? this._context.lineTo(x, y) : this._context.moveTo(x, y); break;
      case 1: this._point = 2; // proceed
      default: this._context.lineTo(x, y); break;
    }
  }
};

var curveLinear = function(context) {
  return new Linear(context);
};

function x$3(p) {
  return p[0];
}

function y$3(p) {
  return p[1];
}

function sign$1(x) {
  return x < 0 ? -1 : 1;
}

// Calculate the slopes of the tangents (Hermite-type interpolation) based on
// the following paper: Steffen, M. 1990. A Simple Method for Monotonic
// Interpolation in One Dimension. Astronomy and Astrophysics, Vol. 239, NO.
// NOV(II), P. 443, 1990.
function slope3(that, x2, y2) {
  var h0 = that._x1 - that._x0,
      h1 = x2 - that._x1,
      s0 = (that._y1 - that._y0) / (h0 || h1 < 0 && -0),
      s1 = (y2 - that._y1) / (h1 || h0 < 0 && -0),
      p = (s0 * h1 + s1 * h0) / (h0 + h1);
  return (sign$1(s0) + sign$1(s1)) * Math.min(Math.abs(s0), Math.abs(s1), 0.5 * Math.abs(p)) || 0;
}

// Calculate a one-sided slope.
function slope2(that, t) {
  var h = that._x1 - that._x0;
  return h ? (3 * (that._y1 - that._y0) / h - t) / 2 : t;
}

// According to https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Representations
// "you can express cubic Hermite interpolation in terms of cubic Bézier curves
// with respect to the four values p0, p0 + m0 / 3, p1 - m1 / 3, p1".
function point$5(that, t0, t1) {
  var x0 = that._x0,
      y0 = that._y0,
      x1 = that._x1,
      y1 = that._y1,
      dx = (x1 - x0) / 3;
  that._context.bezierCurveTo(x0 + dx, y0 + dx * t0, x1 - dx, y1 - dx * t1, x1, y1);
}

function MonotoneX(context) {
  this._context = context;
}

MonotoneX.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 =
    this._y0 = this._y1 =
    this._t0 = NaN;
    this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 2: this._context.lineTo(this._x1, this._y1); break;
      case 3: point$5(this, this._t0, slope2(this, this._t0)); break;
    }
    if (this._line || (this._line !== 0 && this._point === 1)) this._context.closePath();
    this._line = 1 - this._line;
  },
  point: function(x, y) {
    var t1 = NaN;

    x = +x, y = +y;
    if (x === this._x1 && y === this._y1) return; // Ignore coincident points.
    switch (this._point) {
      case 0: this._point = 1; this._line ? this._context.lineTo(x, y) : this._context.moveTo(x, y); break;
      case 1: this._point = 2; break;
      case 2: this._point = 3; point$5(this, slope2(this, t1 = slope3(this, x, y)), t1); break;
      default: point$5(this, this._t0, t1 = slope3(this, x, y)); break;
    }

    this._x0 = this._x1, this._x1 = x;
    this._y0 = this._y1, this._y1 = y;
    this._t0 = t1;
  }
};

function MonotoneY(context) {
  this._context = new ReflectContext(context);
}

(MonotoneY.prototype = Object.create(MonotoneX.prototype)).point = function(x, y) {
  MonotoneX.prototype.point.call(this, y, x);
};

function ReflectContext(context) {
  this._context = context;
}

ReflectContext.prototype = {
  moveTo: function(x, y) { this._context.moveTo(y, x); },
  closePath: function() { this._context.closePath(); },
  lineTo: function(x, y) { this._context.lineTo(y, x); },
  bezierCurveTo: function(x1, y1, x2, y2, x, y) { this._context.bezierCurveTo(y1, x1, y2, x2, y, x); }
};

function createBorderEdge(left, v0, v1) {
  var edge = [v0, v1];
  edge.left = left;
  return edge;
}



// Liang–Barsky line clipping.
function clipEdge(edge, x0, y0, x1, y1) {
  var a = edge[0],
      b = edge[1],
      ax = a[0],
      ay = a[1],
      bx = b[0],
      by = b[1],
      t0 = 0,
      t1 = 1,
      dx = bx - ax,
      dy = by - ay,
      r;

  r = x0 - ax;
  if (!dx && r > 0) return;
  r /= dx;
  if (dx < 0) {
    if (r < t0) return;
    if (r < t1) t1 = r;
  } else if (dx > 0) {
    if (r > t1) return;
    if (r > t0) t0 = r;
  }

  r = x1 - ax;
  if (!dx && r < 0) return;
  r /= dx;
  if (dx < 0) {
    if (r > t1) return;
    if (r > t0) t0 = r;
  } else if (dx > 0) {
    if (r < t0) return;
    if (r < t1) t1 = r;
  }

  r = y0 - ay;
  if (!dy && r > 0) return;
  r /= dy;
  if (dy < 0) {
    if (r < t0) return;
    if (r < t1) t1 = r;
  } else if (dy > 0) {
    if (r > t1) return;
    if (r > t0) t0 = r;
  }

  r = y1 - ay;
  if (!dy && r < 0) return;
  r /= dy;
  if (dy < 0) {
    if (r > t1) return;
    if (r > t0) t0 = r;
  } else if (dy > 0) {
    if (r < t0) return;
    if (r < t1) t1 = r;
  }

  if (!(t0 > 0) && !(t1 < 1)) return true; // TODO Better check?

  if (t0 > 0) edge[0] = [ax + t0 * dx, ay + t0 * dy];
  if (t1 < 1) edge[1] = [ax + t1 * dx, ay + t1 * dy];
  return true;
}

function connectEdge(edge, x0, y0, x1, y1) {
  var v1 = edge[1];
  if (v1) return true;

  var v0 = edge[0],
      left = edge.left,
      right = edge.right,
      lx = left[0],
      ly = left[1],
      rx = right[0],
      ry = right[1],
      fx = (lx + rx) / 2,
      fy = (ly + ry) / 2,
      fm,
      fb;

  if (ry === ly) {
    if (fx < x0 || fx >= x1) return;
    if (lx > rx) {
      if (!v0) v0 = [fx, y0];
      else if (v0[1] >= y1) return;
      v1 = [fx, y1];
    } else {
      if (!v0) v0 = [fx, y1];
      else if (v0[1] < y0) return;
      v1 = [fx, y0];
    }
  } else {
    fm = (lx - rx) / (ry - ly);
    fb = fy - fm * fx;
    if (fm < -1 || fm > 1) {
      if (lx > rx) {
        if (!v0) v0 = [(y0 - fb) / fm, y0];
        else if (v0[1] >= y1) return;
        v1 = [(y1 - fb) / fm, y1];
      } else {
        if (!v0) v0 = [(y1 - fb) / fm, y1];
        else if (v0[1] < y0) return;
        v1 = [(y0 - fb) / fm, y0];
      }
    } else {
      if (ly < ry) {
        if (!v0) v0 = [x0, fm * x0 + fb];
        else if (v0[0] >= x1) return;
        v1 = [x1, fm * x1 + fb];
      } else {
        if (!v0) v0 = [x1, fm * x1 + fb];
        else if (v0[0] < x0) return;
        v1 = [x0, fm * x0 + fb];
      }
    }
  }

  edge[0] = v0;
  edge[1] = v1;
  return true;
}

function cellHalfedgeStart(cell, edge) {
  return edge[+(edge.left !== cell.site)];
}

function cellHalfedgeEnd(cell, edge) {
  return edge[+(edge.left === cell.site)];
}

var epsilon$4 = 1e-6;


var cells;

var edges;

function triangleArea(a, b, c) {
  return (a[0] - c[0]) * (b[1] - a[1]) - (a[0] - b[0]) * (c[1] - a[1]);
}

/**
 * Example params:
 * {
 *   inputSize: 100,
 *   activeBits: [42, 45]
 * }
 */
function arrayOfAxonsChart() {
  let width,
      height;

  let chart = function(selection$$1) {
    selection$$1.each(function(axonsData) {
      let x = linear$2()
          .domain([0, axonsData.inputSize])
          .range([4, width - 4]);

      select(this)
        .selectAll('.border')
        .data([null])
        .enter().append('rect')
        .attr('class', 'border')
        .attr('fill', 'none')
        .attr('stroke', 'black')
        .attr('width', width)
        .attr('height', height);

      let activeAxon = select(this).selectAll('.activeAxon')
          .data(d => d.activeBits);

      activeAxon.enter().append('g')
        .attr('class', 'activeAxon')
        .call(enter => {
          enter.append('circle')
            .attr('r', 1.5)
            .attr('stroke', 'none')
            .attr('fill', 'black');
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

  return chart;
}

/**
 * Example data:
 *    {name: 'A'}
 */
function featureChart() {
  let width,
      height,
      color$$1;

  let chart = function(selection$$1) {
    let featureColor = selection$$1.selectAll('.featureColor')
        .data(d => d.name != null ? [d.name] : []);

    featureColor.exit().remove();

    featureColor.enter()
      .append('rect')
      .attr('class', 'featureColor')
      .attr('width', width)
      .attr('height', height)
      .attr('stroke', 'none')
      .merge(featureColor)
      .attr('fill', d => color$$1(d));

    let featureText = selection$$1.selectAll('.featureText')
        .data(d => d.name != null ? [d.name] : []);

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
    if (!arguments.length) return color$$1;
    color$$1 = _;
    return chart;
  };

  return chart;
}

function partition$1(arr, n) {
  let result = [];

  for (let i = 0; i < arr.length; i += n) {
    result.push(arr.slice(i, i+n));
  }

  return result;
}

/**
 * Example data:
 *    {decodings: [{objectName: 'Object 1', top: 42.0, left: 17.2, amountContained: 0.95}],
 *     objects: {'Object 1': [{name: 'A', left: 11.2, top:12.0, width: 12.2, height: 7.2}],
 *               'Object 2': []}}
 */
function decodedLocationsChart() {
  let width,
      height,
      color$$1,
      minimumMatch = 0.25;

  let chart = function(selection$$1) {

    selection$$1.each(function(decodedLocationsData) {
      let decodingsByObject = {};
      decodedLocationsData.decodings.forEach(d => {
        if (d.amountContained >= minimumMatch) {
          if (!decodingsByObject.hasOwnProperty(d.objectName)) {
            decodingsByObject[d.objectName] = [];
          }

          decodingsByObject[d.objectName].push(d);
        }
      });

      let decodings = [],
          maxWidth = 0,
          maxHeight = 0;
      for (let objectName in decodingsByObject) {
        decodings.push([objectName, decodingsByObject[objectName]]);

        decodedLocationsData.objects[objectName].forEach(d => {
          maxWidth = Math.max(maxWidth, d.left + d.width);
          maxHeight = Math.max(maxHeight, d.top + d.height);
        });
      }

      // Sort by object name.
      decodings.sort((a,b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);

      let rows = partition$1(decodings, 3);

      let decodedObjectRow = select(this).selectAll('.decodedObjectRow')
          .data(rows);

      decodedObjectRow.exit().remove();

      decodedObjectRow = decodedObjectRow.enter().append('g')
        .attr('class', 'decodedObjectRow')
        .attr('transform', (d,i) => `translate(0,${i == 0 ? 0 : i*height/2.5 + 10})`)
        .merge(decodedObjectRow);

      let decodedObject = decodedObjectRow.selectAll('.decodedObject')
          .data(d => d);

      decodedObject.exit().remove();

      decodedObject = decodedObject.enter().append('g')
        .attr('class', 'decodedObject')
        .attr('transform', (d, i) => `translate(${i*width/3},0)`)
        .call(enter => {
          enter.append('g')
            .attr('class', 'features');
          enter.append('g')
              .attr('class', 'points')
            .append('rect')
              .attr('width', width/3)
              .attr('height', height/2.5)
              .attr('fill', 'white')
              .attr('fill-opacity', 0.7);
        })
        .merge(decodedObject);

      decodedObject.each(function([objectName, decodedLocations]) {

        let cmMax = Math.max(maxWidth, maxHeight);
        let pxMax = Math.min(width/3, height/2.5);
        let x = linear$2()
            .domain([0, cmMax])
            .range([0, pxMax]);
        let y = linear$2()
            .domain([0, cmMax])
            .range([0, pxMax]);

        let feature = select(this).select('.features').selectAll('.feature')
            .data(decodedLocationsData.objects[objectName]);

        feature.exit().remove();

        feature = feature.enter()
          .append('g')
          .attr('class', 'feature')
          .merge(feature)
          .attr('transform', d => `translate(${x(d.left)},${y(d.top)})`)
          .each(function(featureData) {
            select(this)
              .call(featureChart()
                    .width(x(featureData.width))
                    .height(y(featureData.height))
                    .color(color$$1));
          });

        let point = select(this).select('.points').selectAll('.point')
            .data(decodedLocations);

        point.exit().remove();

        point = point.enter().append('g')
            .attr('class', 'point')
          .call(enter => {
            enter.append('circle')
              .attr('r', 4)
              .attr('fill', 'white')
              .attr('stroke', 'none');

            enter.append('path')
              .attr('fill', 'black')
              .attr('stroke', 'none');
          }).merge(point)
            .attr('transform', d => `translate(${x(d.left)},${y(d.top)})`);

        point.select('path')
          .attr('d', arc()
                .innerRadius(0)
                .outerRadius(4)
                .startAngle(0)
                .endAngle(d => d.amountContained / 1.0 * 2 * Math.PI));
      });
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

  chart.color = function(_) {
    if (!arguments.length) return color$$1;
    color$$1 = _;
    return chart;
  };

  chart.minimumMatch = function(_) {
    if (!arguments.length) return minimumMatch;
    minimumMatch = _;
    return chart;
  };

  return chart;
}

function partition$2(arr, n) {
  let result = [];

  for (let i = 0; i < arr.length; i += n) {
    result.push(arr.slice(i, i+n));
  }

  return result;
}

/**
 * Example data:
 *    {decodings: ['Object 1', 'Object 2'],
 *     objects: {'Object 1': [{name: 'A', top: 42.0, left: 17.2, width: 12.2, height: 7.2}],
 *               'Object 2': []}}
 */
function decodedObjectsChart() {
  let width,
      height,
      color$$1;

  let chart = function(selection$$1) {

    selection$$1.each(function(decodedObjectsData) {

      let maxWidth = 0,
          maxHeight = 0;
      decodedObjectsData.decodings.forEach(objectName => {
        decodedObjectsData.objects[objectName].forEach(d => {
          maxWidth = Math.max(maxWidth, d.left + d.width);
          maxHeight = Math.max(maxHeight, d.top + d.height);
        });
      });

      let decodings = decodedObjectsData.decodings.slice();

      // Sort by object name.
      decodings.sort();

      let rows = partition$2(decodings, 3);

      let decodedObjectRow = select(this).selectAll('.decodedObjectRow')
          .data(rows);

      decodedObjectRow.exit().remove();

      decodedObjectRow = decodedObjectRow.enter().append('g')
        .attr('class', 'decodedObjectRow')
        .attr('transform', (d,i) => `translate(0,${i == 0 ? 0 : i*height/3 + 10})`)
        .merge(decodedObjectRow);

      let decodedObject = decodedObjectRow.selectAll('.decodedObject')
          .data(d => d);

      decodedObject.exit().remove();

      decodedObject = decodedObject.enter().append('g')
        .attr('class', 'decodedObject')
        .attr('transform', (d, i) => `translate(${i*width/3},0)`)
        .merge(decodedObject);

      decodedObject.each(function(objectName) {
        let cmMax = Math.max(maxWidth, maxHeight);
        let pxMax = Math.min(width/3, height/3);
        let x = linear$2()
            .domain([0, cmMax])
            .range([0, pxMax]);
        let y = linear$2()
            .domain([0, cmMax])
            .range([0, pxMax]);

        let feature = select(this).selectAll('.feature')
            .data(decodedObjectsData.objects[objectName]);

        feature.exit().remove();

        feature.enter()
          .append('g')
          .attr('class', 'feature')
          .merge(feature)
          .attr('transform', d => `translate(${x(d.left)},${y(d.top)})`)
          .each(function(featureData) {
            select(this)
              .call(featureChart()
                    .width(x(featureData.width))
                    .height(y(featureData.height))
                    .color(color$$1));
          });
      });
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

  chart.color = function(_) {
    if (!arguments.length) return color$$1;
    color$$1 = _;
    return chart;
  };

  return chart;
}

/**
 * Example params:
 * {
 *   dimensions: {rows: 5, cols: 5},
 *   cells: [{cell: 42, state: 'active'},
 *           {cell: 43, state: 'predicted'},
 *           {cell: 44, state: 'predicted-active'}]
 *   highlightedCells: [42]
 * }
 */
function layerOfCellsChart() {
  let width,
      height,
      color$$1 = ordinal()
        .domain(['active', 'predicted', 'predicted-active'])
        .range(['orangered', 'rgba(0, 127, 255, 0.498)', 'black']),
      stroke = "black",
      onCellSelected = selectedCell => {},
      columnMajorIndexing = false;

  let drawHighlightedCells = function(selection$$1) {
    selection$$1.each(function(layerData) {
      select(this).selectAll('.cell')
        .attr('stroke-width', d =>
              layerData.highlightedCells.indexOf(d.cell) != -1
              ? 2
              : 0);
    });
  };

  let chart = function(selection$$1) {
    selection$$1.each(function(layerData) {
      let layerNode = this,
          layer = select(layerNode),
          xScale = linear$2()
            .domain([0, layerData.dimensions.cols - 1])
            .range([5, width - 5]),
          yScale = linear$2()
            .domain([0, layerData.dimensions.rows - 1])
            .range([5, height - 5]);

      let x, y;
      if (columnMajorIndexing) {
        x = d => xScale(Math.floor(d.cell / layerData.dimensions.rows));
        y = d => yScale(d.cell % layerData.dimensions.rows);
      } else {
        x = d => xScale(d.cell % layerData.dimensions.cols);
        y = d => yScale(Math.floor(d.cell / layerData.dimensions.cols));
      }

      // For each layer, keep:
      // - the selected cell, so that we fire events only when the selection
      //   changes.
      // - the mouse position from the most recent mousemove, so that we can
      //   reevaluate the selected cell when the data changes.
      if (layerNode._selectedCell === undefined) {
        layerNode._selectedCell = null;
      }
      if (layerNode._mousePosition === undefined) {
        layerNode._mousePosition = null;
      }

      layer.selectAll('.border')
        .data([null])
        .enter().append('rect')
          .attr('class', 'border')
          .attr('stroke', stroke)
          .attr('stroke-width', 1)
          .attr('fill', 'none')
          .attr('width', width)
          .attr('height', height);

      let cells = layer.selectAll('.cells')
          .data([layerData.cells]);

      cells = cells.enter()
        .append('g')
          .attr('class', 'cells')
        .merge(cells);

      let cell = cells.selectAll('.cell')
          .data(d => d);

      cell.exit().remove();

      cell = cell.enter()
        .append('polygon')
          .attr('class', 'cell')
          .attr('stroke', 'goldenrod')
        .merge(cell)
          .attr('fill', d => color$$1(d.state))
          .attr('stroke-width', d =>
                layerData.highlightedCells.indexOf(d.cell) != -1
                ? 2
                : 0);

      // Enable fast lookup of the cell nearest to the cursor.
      let quadtree$$1 = quadtree()
          .extent([[0, 0], [width, height]])
          .x(x)
          .y(y)
          .addAll(layerData.cells);

      let mouseEvents = layer.selectAll('.mouseEvents')
          .data([null]);

      mouseEvents.enter()
        .append('rect')
          .attr('class', 'mouseEvents')
          .attr('stroke', 'transparent')
          .attr('fill', 'transparent')
          .attr('width', width)
          .attr('height', height)
        .merge(mouseEvents)
        .on('mousemove', function() {
          layerNode._mousePosition = mouse(this);

          let p = quadtree$$1.find(layerNode._mousePosition[0],
                                layerNode._mousePosition[1]);

          if (p !== layerNode._selectedCell) {
            layerNode._selectedCell = p;
            draw();
            onCellSelected(p.cell);
          }
        })
        .on('mouseleave', () => {
          layerNode._mousePosition = null;

          if (layerNode._selectedCell !== null) {
            layerNode._selectedCell = null;
            draw();
            onCellSelected(null);
          }
        });

      // If we're rerendering, check if it has caused the nearest cell to
      // change.
      if (layerNode._mousePosition !== null) {
        let p = quadtree$$1.find(layerNode._mousePosition[0],
                              layerNode._mousePosition[1]);
        if (p !== layerNode._selectedCell) {
          layerNode._selectedCell = p;
          onCellSelected(p.cell);
        }
      }

      draw();


      function draw() {
        cell
          .attr('transform', d => `translate(${x(d)},${y(d)})`)
          .attr('points', cell =>
                cell == layerNode._selectedCell
                ? '0,-6 5,6 -5,6'
                : (cell.state == 'predicted')
                ? '0,-3 1.5,1.5 -1.5,1.5'
                : '0,-4 2,2 -2,2');
      }
    });
  };

  chart.drawHighlightedCells = drawHighlightedCells;

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

  chart.stroke = function(_) {
    if (!arguments.length) return stroke;
    stroke = _;
    return chart;
  };

  chart.onCellSelected = function(_) {
    if (!arguments.length) return onCellSelected;
    onCellSelected = _;
    return chart;
  };

  chart.columnMajorIndexing = function(_) {
    if (!arguments.length) return columnMajorIndexing;
    columnMajorIndexing = _;
    return chart;
  };

  return chart;
}

/**
 * Data:
 * {modules: [{dimensions: {rows: 5, cols: 5},
 *             scale: 3.2,
 *             orientation: 0.834,
 *             cells: [{cell: 2, state: 'active'}],
 *            ...],
 *  highlightedCells: [10, 46]}
 */
function locationModulesChart() {
  var width,
      height,
      onCellSelected = (iModule, selectedCell) => {};

  let drawHighlightedCells = function(selection$$1) {
    selection$$1.each(function(moduleArrayData) {
      let moduleWidth = width / 6,
          moduleHeight = height / 3,
          highlightedCellsByModule = [];

      let base = 0;
      moduleArrayData.modules.forEach(module => {
        let end = base + module.dimensions.rows*module.dimensions.cols;

        let highlightedInModule = [];
        moduleArrayData.highlightedCells.forEach(cell => {
          if (cell >= base && cell < end) {
            highlightedInModule.push(cell - base);
          }
        });

        highlightedCellsByModule.push(highlightedInModule);

        base = end;
      });

      let module = select(this)
          .selectAll('.module')
          .datum((d, i) => {
            d.highlightedCells = highlightedCellsByModule[i];
            return d;
          })
          .each(function(d, i) {
            select(this).call(
              layerOfCellsChart()
                .width(moduleWidth)
                .height(moduleHeight)
                .stroke('lightgray')
                .onCellSelected(
                  cell => onCellSelected(cell !== null ? i : null,
                                         cell))
                .drawHighlightedCells);
          });
    });
  };

  var chart = function(selection$$1) {
    let modules = selection$$1.selectAll('.modules')
        .data(d => [d]);

    modules = modules.enter()
      .append('g')
      .attr('class', 'modules')
      .merge(modules);

    selection$$1.selectAll('.boundary')
      .data([null])
      .enter().append('rect')
      .attr('class', 'boundary')
      .attr('fill', 'none')
      .attr('stroke', 'black')
      .attr('width', width)
      .attr('height', height);

    modules.each(function(moduleArrayData) {
      // TODO: stop hardcoding 18 modules
      let moduleWidth = width / 6,
          moduleHeight = height / 3;

      let module = select(this)
          .selectAll('.module')
          .data(moduleArrayData.modules.map(m => {
            return Object.assign({highlightedCells: []}, m);
          }));

      module.exit().remove();

      module = module.enter().append('g')
        .attr('class', 'module')
        .merge(module)
        .attr('transform',
              (d, i) => `translate(${Math.floor(i/3) * moduleWidth},${(i%3)*moduleHeight})`)
        .each(function(d, i) {
          select(this)
            .call(layerOfCellsChart()
                  .width(moduleWidth)
                  .height(moduleHeight)
                  .stroke('lightgray')
                  .onCellSelected(
                    cell => onCellSelected(cell !== null ? i : null,
                                           cell)));
        });
    });
  };

  chart.drawHighlightedCells = drawHighlightedCells;

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

  chart.onCellSelected = function(_) {
    if (!arguments.length) return onCellSelected;
    onCellSelected = _;
    return chart;
  };

  return chart;
}

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

function motionChart() {
  let chart = function(selection$$1) {
    selection$$1.each(function(deltaLocation) {
      let data = deltaLocation != null ? [deltaLocation] : [];

      let arrow = select(this).selectAll('.arrow')
          .data(data);

      arrow.exit().remove();

      let arrowLength,
          correctedArrowLength,
          correctionFactor;
      if (deltaLocation != null) {
        arrowLength = Math.sqrt(Math.pow(deltaLocation.top, 2) +
                                Math.pow(deltaLocation.left, 2));
        correctedArrowLength = Math.min(900, Math.max(15, arrowLength));
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

      let leftLabel = select(this).selectAll('.leftLabel')
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
          return `translate(0,${Math.abs(d.top/2) + 10})`;
        });

      leftLabel.select('text')
        .text(d => format('.0f')(Math.abs(d.left)));

      leftLabel.select('.beginCap')
        .attr('x1', d => -correctionFactor*d.left/2)
        .attr('x2', d => -correctionFactor*d.left/2);

      leftLabel.select('.endCap')
        .attr('x1', d => correctionFactor*d.left/2)
        .attr('x2', d => correctionFactor*d.left/2);

      leftLabel.select('.line')
        .attr('x1', d => -correctionFactor*d.left/2)
        .attr('x2', d => correctionFactor*d.left/2);

      let topLabel = select(this).selectAll('.topLabel')
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
          return `translate(${Math.abs(d.left/2) + 12},0)`;
        });

      topLabel.select('text')
        .text(d => format('.0f')(Math.abs(d.top)));

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

  return chart;
}

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

  let drawSelectedStep = function(selection$$1) {
    selection$$1.each(function(timelineData) {
      select(this).selectAll('.colorWithSelection')
        .attr('fill', d => d.iTimestep == timelineData.selectedIndex
              ? 'black'
              : 'lightgray');

      select(this).selectAll('.move.colorWithSelection')
        .attr('stroke', d => d.iTimestep == timelineData.selectedIndex
              ? 'black'
              : 'lightgray');

      select(this).selectAll('.selectedText')
        .style('visibility', d => d.iTimestep == timelineData.selectedIndex
               ? 'visible'
               : 'hidden');
    });
  };

  var chart = function(selection$$1) {
    let timeline = selection$$1.selectAll('.timeline')
        .data(d => [d]);

    timeline = timeline.enter()
      .append('g')
      .attr('class', 'timeline')
      .merge(timeline);

    timeline.each(function(timelineData) {
      let timelineNode = select(this);

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
            timesteps: [o]
          });

          break;
        case 'move':
          o.text = 'Move';
          shapes.push(o);
          break;
        case 'repeat': {
          let touchData = shapes[shapes.length-1];

          if (touchData.type != 'sense') {
            throw `Invalid data ${touchData.type}`;
          }

          o.text = 'Settle';
          touchData.timesteps.push(o);
          break;
        }
        default:
          throw `Unrecognized ${o.type}`;
        }
      });

      let onchangeFn = onchange
          ? d => onchange(d.iTimestep)
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

    drawSelectedStep(selection$$1);
  };

  chart.drawSelectedStep = drawSelectedStep;

  chart.onchange = function(_) {
    if (!arguments.length) return onchange;
    onchange = _;
    return chart;
  };

  return chart;
}

const CURSOR_URL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAA1CAYAAAADOrgJAAAAAXNSR0IArs4c6QAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAnRpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4wIj4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAgeG1sbnM6dGlmZj0iaHR0cDovL25zLmFkb2JlLmNvbS90aWZmLzEuMC8iCiAgICAgICAgICAgIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyI+CiAgICAgICAgIDx0aWZmOllSZXNvbHV0aW9uPjMwMDwvdGlmZjpZUmVzb2x1dGlvbj4KICAgICAgICAgPHRpZmY6Q29tcHJlc3Npb24+NTwvdGlmZjpDb21wcmVzc2lvbj4KICAgICAgICAgPHRpZmY6WFJlc29sdXRpb24+MzAwPC90aWZmOlhSZXNvbHV0aW9uPgogICAgICAgICA8eG1wOkNyZWF0b3JUb29sPkZseWluZyBNZWF0IEFjb3JuIDUuNi40PC94bXA6Q3JlYXRvclRvb2w+CiAgICAgICAgIDx4bXA6TW9kaWZ5RGF0ZT4yMDE3LTA2LTEyVDExOjI2OjI5PC94bXA6TW9kaWZ5RGF0ZT4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+CrudzKAAAAm6SURBVGgF7ZhPbJVZGca/2z9MYaDQ4vAn40yZdmIGjaBBVxAkGiAhRFxgCLKYZFgYFuBi4gIWJi5dmIDBQGKiC2IMRsQY2Zk4akxQF0BYGFkACs0EbCkpLbbQe795fqff891zv/vd9t6CE2N8k6fv+fOec97nfc8537lNkv/Lf1cEKi/ZnbL50pe8Rul0ZQuXGi7Q2Mkc/zFSnThRxsXjizq2tfPW9MXl2HbJZTuwlAk8Fg26Mu125sThMrgP/VIkXrSTCT3OzqNBj9AtuL+m8nMBbZiYml5eZli0U7GTReeXaSKcnRWqAgTQrwiM8bh4W7lN3S8mRLAT8cJ2jECAV4VnwteFrwmTwgrhvvB9AXIAGwSCSExqvuVj+gsBbyEiDYFBYUBAzgjeOugPhdeE5cJaoV/oE3oF5nFAVHwx6TQjXs2E2E7IV4U3du3a9Znt27cnjx8/frpixYq+qamp6rlz576lvinhkvBQgITJshUpMx/6YxFHj61ENlYK6wWy8UchPX/+fCqp8ScTtpCd3pHZkkG2HYSYK84MayxJOs2IyaBxgvE48m8hGRsbm5PK53z27BkRx/apgOMEAGK+yVTMiZZlpKyNMU2SL9rU07oBx3AeQMaEEmWBvlyyuklDApAlSCH0Fa9l2pxFz7cooaUQsQMmwmJhwVoNn+oS1ennkHPoIYKhx7tOW7wVTQa9KKF2iXgitOGsoEO/MqBiXQp11gLYkhEc9wB0GTHaYzDWY1SsSztEgpMagi7CZGxTn1mlAhGu3w0CBDgzfFM4K5wrMsF3htuNcgwTtmatJjILEYmdo2yny7S6myU7IyzK2fiegNOIo0yZtlXCH4T3BGxNkD4TVTHPYhOZVkRMAm0S6PhwM5Y6xGyvYl2iM0Lj6npPaelNtXpOok9W2ILM7+eOiuVkMCqKnTIBO8/Hj8PKNwDtLzSLe4yKpZJev379+YMHD+Zu3bo1Nzs7O3fgwAGcQ64K3xR+K/xG+LnwK+HXwjsCZFjPJEsDR2csdigmwUAmYzs8ESDkCPHkaEkkPiNbt24NduvWrdOQJBkaGiLiyLjwgcCH8nNCLIyZFFjHPnhbovETHZxAxxKTIBtEHvm0cEzAARYAvxDuCyzSJHNzbO/6YrGBnjGukuEhYQ0NK1eufH7x4sWu7u7u2uHDh789MTHBxfAj4c8C67DtQE5C5QYidCAmQiaIPvUJgVB+Q4iF7fB3IUQl7qD89Ck+zPdVq9VEzlEPcv8+/IMQGG6skCG9zyr79u1j7a7+/v79IoLR7wS2oHcD/ayZr0tUY8FpE2EQk39B+PLGjRs/eezYsVQLEeau3t7e5MyZM/tU/7zqrwuJMsACuaxdu7ais1BRlNOYBAZHjx7tWrVqVbJ58+bhbdu2vXv37t23b9y4kYyMjDAHPtQUCK5objEEXXa5YJsTwpAGDEkf6f6EwCH7jpDu3LlTW77hQVgbHBx0VNC1U6dOYdOJxA9MxsX16sDAAGeRuY8LfIMImH8KkASTDgXVAwk0AiGAEVkJP4YU+bBPVM9F5OY2bdpUrVQqLFYpXLe53QIFR5PMe9/n5tF8BNjZyJ3PDVWgsSgxEcqQYX/70Nu+cvny5Z47d+507927FzufCfe3qxlrJxvGKEOu4yc29s3tuS4jkneqQD9PdLLCqfONoWJd1qxZE7ZWttXqHSo9efKEvZ57pHOUTk5OMo/bQp32hoGNFRMIAVOX641WUQesyQA/YXkbvSFwmHcIJwUWe17Yy97b7O94j6e6mdgyAe7TYeayqF24cIFx6enTp5mztmXLloax6qrq1vIZeV82Qxn4IccZ5iwT6EDOtxaT0YA27MS02gAfJoTBRDSfRGVHSsW6ZDcVdswZRNlpiGZ2DirZVW2zoLu6GBrEPrWqB2fcac0gHIUIEWFbkSW+Fzwlvrthw4aZ1atXV3THY9NSuKIlOQkqXLnI8uVciHXt9tA4/6ciIh7rLV0klJs7I26wYUyEjxUejQk8J0b0ZpIK342gW/3JIupMBzPdcKU6ir6nS/UPjN5Hjx5RJzVsSba+fVSxLjERDLyoiTCYjHDgITMfxvlJk76+PsYsSYqEdCY8T23//v2JslnTGfuxGkeFmwLr4wu+mUw+KCai/iB0YgwpiBANIgEZthI6PMm1HfJNrLYmyZzLF8PADhe1ifGUuXLlCvOy/u+Fvwqus9VNRMUgYf6iI17UZHDcWWGLERFy/UPhp8ePH5/QsyW5ffs2kzdJ5iwONUn0sQt9fsJIs7bX5WxyQ9EGCdpNxL6qqfHRGBr0BwMWRwNPChEyA5EfCANnz57dLZ0cOXKka3h42ONoCjI9zWXXKPfu3QsNfv1mj8JE3xYbOvqsReCYBBIxkabAYVwmcRRdRgPG8J3xv0vH9Ezpe/jw4avsaz0U8yxza127di3VozA9ePBgGK8s1SBIJtevX1/p6elJRS7dvXs3znVdvXp1+tKlS39S+W/CB8K/BAgRSJNxkNW0uMSOO8X8ZuDxNixsEb4obBP+IqQnT56Un0uT7OOJgx8KewV+ZH1WeEt4TeDe5gXMuXbWVJyXPHpuiDSTImiixRYjIkRnRuARSR8T/0O4p5+wYzMzM4m2S1Pq1d8gPF1u3rxZ1Zhgq0ep+wkgNxTB8/lEtzwf6ltUnBUI4zAPR/7ny/PldeFTAk+YLwk7hJ8J6Z49e+YWy8uhQ4cIQnrixIlgOjo6CiHayMhXBLIxIvCDrl9omQ31NfxCpF4UJoZMWFSaqCAmaE0EiRpIli1bhv1CWanIhqzO6pbCwb7x8XHmJliM8w3JDnA27AN6yWKHOeg4TWaIEnv3TYHovSP8RGAhHMChMuAYNmTvoPDL7CnClqV9VODsvSVsFPhm8SEuPRtqD0JnO8ICkEHjHIKzJsh+JsK0IU2Hcb45/CUYCPftP4Wp7JsSn1dnxNlgTdYGpYIj7Ypt7byzAwkyhB7KQB91dEzKGWIOHmzjwqDA0xzn6YcgV++0YEIEiK3Xkoydk01bYns0DpqMtxsZpg1itMXbwU6YjEn6G0E7ZWe1SIJ+z6Fio7S7tTyKiSDhCZk8HPDMgIMLQbfRb/KYMM4O+TqlHXE7ZUg4C25nbEvplAgTxWRYxGIncYBoO+JFIti1AvNBMIZt1RzGoZtkKUSYhMktJkObHfG5QBfFjnmOWHsO5gGu26Y4V16Po5U3tlnwWHQRJsJUtoudcdnaS1IvA/1FW48J2os0NHZQice7jHaZqeJy0ZliHXu3WcdtlEslXqTUoM3G4jzF+kLTxA6X2S3WH8Z0smDZImVtLzpnW46XLfw/0fYRMvJsGfpXau0AAAAASUVORK5CYII=';

/**
 * Data:
 * {location: [12.0, 16.0],
 *  features: [{name: 'A', location: {}, width: 12.2, height: 7.2}],
 *  selectedLocationCell: 42,
 *  selectedLocationModule: {
 *    cellDimensions: [5, 5],
 *    moduleMapDimensions: [20.0, 20.0],
 *    orientation: 0.834
 *    activeCells: [42],
 *    activePoints: [[11.0, 12.0]]
 *  }}
 */
function worldChart() {
  var width,
      height,
      color$$1,
      t = 0;

  let drawFiringFields = function(selection$$1) {
    selection$$1.each(function(worldData) {
      let worldBackground = select(this).select('.worldBackground'),
          firingFields = worldBackground.select('.firingFields');

      let xScale = linear$2()
            .domain([0, worldData.dims.width])
            .range([0, width]),
          x = location => xScale(location.left),
          yScale =  linear$2()
            .domain([0, worldData.dims.height])
            .range([0, height]),
          y = location => yScale(location.top);

      let pattern = worldBackground.select('.myDefs')
          .selectAll('pattern')
          .data(worldData.selectedLocationModule
                ? [worldData.selectedLocationModule.activePoints]
                : []);

      pattern.exit().remove();

      if (worldData.selectedLocationModule) {
        let config = worldData.selectedLocationModule,
            distancePerCell = [config.moduleMapDimensions[0] / config.cellDimensions[0],
                               config.moduleMapDimensions[1] / config.cellDimensions[1]],
            pixelsPerCell = [height * (distancePerCell[0] / worldData.dims.height),
                             width * (distancePerCell[1] / worldData.dims.width)];

        pattern.enter().append('pattern')
          .attr('id', 'FiringField')
          .attr('patternUnits', 'userSpaceOnUse')
          .call(enter => {
            enter.append('path')
              .attr('fill', 'black')
              .attr('fill-opacity', 0.6)
              .attr('stroke', 'none');
          })
          .merge(pattern)
          .attr('width', config.cellDimensions[1])
          .attr('height', config.cellDimensions[0])
          .call(p => {
            p.select('path')
              .attr('d', points => {

                let squares = [];

                for (let i = -1; i < 2; i++) {
                  for (let j = -1; j < 2; j++) {
                    points.forEach(point => {

                      let cellFieldOrigin = [
                        Math.floor(worldData.selectedLocationCell / config.cellDimensions[1]),
                        worldData.selectedLocationCell % config.cellDimensions[1]];

                      let top = i*config.cellDimensions[1] + (cellFieldOrigin[0] - point[0]),
                          left = j*config.cellDimensions[0] + (cellFieldOrigin[1] - point[1]);

                      squares.push(`M ${left} ${top} l 1 0 l 0 1 l -1 0 Z`);

                    });
                  }
                }

                return squares.join(' ');
              });
          })
          .attr('patternTransform', point => {
            let translateLocation = {
              top: worldData.location.top,
              left: worldData.location.left
            };

            return `translate(${x(translateLocation)},${y(translateLocation)}) `
              + `rotate(${180 * config.orientation / Math.PI}, 0, 0) `
              + `scale(${pixelsPerCell[1]},${pixelsPerCell[0]})`;
          });
      }

      let firingField = firingFields.selectAll('.firingField')
          .data(worldData.selectedLocationModule
                ? [worldData.selectedLocationModule.activePoints]
                : []);

      firingField.enter().append('rect')
        .attr('class', 'firingField')
        .attr('fill', (d,i) => `url(#FiringField)`)
        .attr('stroke', 'none')
        .attr('width', width)
        .attr('height', height);

      firingField.exit().remove();
    });
  };

  var chart = function(selection$$1) {
    selection$$1.each(function(worldData) {
      let worldNode = select(this);

      let xScale = linear$2()
            .domain([0, worldData.dims.width])
            .range([0, width]),
          x = location => xScale(location.left),
          yScale =  linear$2()
            .domain([0, worldData.dims.height])
            .range([0, height]),
          y = location => yScale(location.top);


      let features = worldNode.selectAll('.features')
        .data(d => [d]);

      features = features.enter().append('g')
        .attr('class', 'features')
        .merge(features);

      let feature = features.selectAll('feature')
          .data(d => d.features ? d.features : []);

      feature.exit().remove();

      feature.enter()
        .append('g')
        .attr('class', 'feature')
        .merge(feature)
        .attr('transform', d => `translate(${x(d)},${y(d)})`)
        .each(function(featureData) {
          select(this)
            .call(featureChart()
                  .width(xScale(featureData.width))
                  .height(yScale(featureData.height))
                  .color(color$$1));
        });

      let worldBackground = worldNode.selectAll('.worldBackground')
        .data(d => [d]);

      worldBackground = worldBackground
        .enter().append('g')
          .attr('class', 'worldBackground')
        .call(enter => {
          enter.append('defs')
              .attr('class', 'myDefs');

          enter.append('rect').attr('fill', 'none')
            .attr('width', width)
            .attr('height', height)
            .attr('stroke', 'lightgray')
            .attr('stroke-width', 1);

          enter.append('g').attr('class', 'firingFields');

        })
        .merge(worldBackground);

      let currentLocation = select(this).selectAll('.currentLocation')
          .data([null]);

      currentLocation.enter().append('g')
        .attr('class', 'currentLocation')
        .call(enter => {
          enter.append('image')
            .attr('x', -21)
            .attr('y', -10)
            .attr('width', 50)
            .attr('height', 53)
            .attr('xlink:href', CURSOR_URL);
        })
        .merge(currentLocation)
        .attr('transform', `translate(${x(worldData.location)},${y(worldData.location)})`);
    });

    drawFiringFields(selection$$1);
  };

  chart.drawFiringFields = drawFiringFields;

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
    if (!arguments.length) return color$$1;
    color$$1 = _;
    return chart;
  };


  return chart;
}

/**
 *
 * Example timestep:
 * {
 *   worldLocation: {left: 42.0, top: 12.0},
 *   reset: null,
 *   locationLayer: {
 *     modules: [
 *       {activeCells: [],
 *        activePoints: [],
 *        activeSynapsesByCell: {}},
 *     ]
 *   },
 *   inputLayer: {
 *     activeCells: [],
 *     decodings: [],
 *     activeSynapsesByCell: {
 *       42: {
 *         locationLayer: [12, 17, 29],
 *         objectLayer: [42, 45]
 *       }
 *     }
 *   },
 *   objectLayer: {
 *     activeCells: [],
 *     decodings: [],
 *     activeSynapsesByCell: {}
 *   },
 *   deltaLocationInput: {
 *   },
 *   featureInput: {
 *     inputSize: 150,
 *     activeBits: [],
 *     decodings: []
 *   }
 * };
 */
function parseData(text$$1) {
  let featureColor = ordinal(),
      timesteps = [],
      rows = text$$1.split('\n');

  // Row: world dimensions
  let worldDims = JSON.parse(rows[0]);

  // Row: Features and colors
  //   {'A': 'red',
  //    'B': 'blue',
  //    'C': 'gray'}
  let featureColorMapping = JSON.parse(rows[1]);

  var features = [],
      colors = [];

  for (var feature in featureColorMapping) {
    features.push(feature);
    colors.push(featureColorMapping[feature]);
  }

  featureColor
    .domain(features)
    .range(colors);

  // Third row: Objects
  // {
  //   'Object 1': [
  //     {top: 12.0, left: 11.2, width: 5.2, height: 19, name: 'A'},
  //     ...
  //   ],
  //   'Object 2': []
  // };
  let objects = JSON.parse(rows[2]);

  let currentTimestep = null,
      didReset = false,
      objectPlacements = null,
      worldFeatures = null,
      locationInWorld = null;

  // [{cellDimensions: [5,5], moduleMapDimensions: [20.0, 20.0], orientation: 0.2},
  //  ...]
  let configByModule = JSON.parse(rows[3]).map(d => {
    d.dimensions = {rows: d.cellDimensions[0], cols: d.cellDimensions[1]};
    return d;
  });

  function endTimestep() {
    if (currentTimestep !== null) {
      currentTimestep.objectPlacements = objectPlacements;
      currentTimestep.worldFeatures = worldFeatures;

      if (currentTimestep.type == 'move') {
        currentTimestep.worldLocation = locationInWorld;
        currentTimestep.featureInput = {
          inputSize: 150,
          activeBits: [],
          decodings: []
        };

        // Continue showing the previous object layer.
        for (let i = timesteps.length - 1; i >= 0; i--) {
          if (timesteps[i].type !== 'move') {
            currentTimestep.objectLayer =
              Object.assign({}, timesteps[i].objectLayer);
            currentTimestep.objectLayer.activeSynapsesByCell = {};
            break;
          }
        }
      }

      timesteps.push(currentTimestep);
    }

    currentTimestep = null;
  }

  function beginNewTimestep(type) {
    endTimestep();

    currentTimestep = {
      worldLocation: locationInWorld,
      type
    };

    if (didReset) {
      currentTimestep.reset = true;
      didReset = false;
    }
  }

  let i = 4;
  while (i < rows.length) {
    switch (rows[i]) {
    case 'reset':
      didReset = true;
      i++;
      break;
    case 'sense':
      beginNewTimestep('sense');

      currentTimestep.featureInput = {
        inputSize: 150,
        activeBits: JSON.parse(rows[i+1]),
        decodings: JSON.parse(rows[i+2])
      };

      i += 3;
      break;
    case 'sensoryRepetition':
      beginNewTimestep('repeat');
      currentTimestep.featureInput = timesteps[timesteps.length - 1].featureInput;
      i++;
      break;
    case 'move': {
      beginNewTimestep('move');
      let deltaLocation = JSON.parse(rows[i+1]);

      currentTimestep.deltaLocation = {
        top: deltaLocation[0],
        left: deltaLocation[1]
      };

      i += 2;
      break;
    }
    case 'locationInWorld': {
      let location = JSON.parse(rows[i+1]);

      locationInWorld = {top: location[0], left: location[1]};

      i += 2;
      break;
    }
    case 'shift': {
      let modules = [];
      JSON.parse(rows[i+1]).forEach((activeCells, i) => {
        let cells = activeCells.map(cell => {
          return {
            cell,
            state: 'predicted-active'
          };
        });

        modules.push(Object.assign({cells,
                                    activeSynapsesByCell: {}},
                                   configByModule[i]));
      });

      JSON.parse(rows[i+2]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      let decodings = JSON.parse(rows[i+3]).map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });
      currentTimestep.locationLayer = { modules, decodings };

      i += 4;
      break;
    }
    case 'locationLayer': {
      let modules = [];

      JSON.parse(rows[i+1]).forEach((module, i) => {
        let [activeCells, segmentsForActiveCells] = module;

        let prevActiveCells = (currentTimestep.reset || timesteps.length == 0)
            ? []
            : timesteps[timesteps.length-1].locationLayer.modules[i].cells.map(
              d => d.cell);

        let cells = activeCells.map(cell => {
          return {
            cell,
            state: prevActiveCells.indexOf(cell) != -1
              ? 'predicted-active'
              : 'active'
          };
        });

        let activeSynapsesByCell = {};

        if (segmentsForActiveCells) {

          activeCells.forEach(cell => {
            activeSynapsesByCell[cell] = {};
          });

          for (let presynapticLayer in segmentsForActiveCells) {
            segmentsForActiveCells[presynapticLayer].forEach((segments, ci) => {
              let synapses = [];
              segments.forEach(presynapticCells => {
                synapses = synapses.concat(presynapticCells);
              });

              activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
            });
          }
        }

        modules.push(Object.assign({cells, activeSynapsesByCell},
                                   configByModule[i]));
      });

      JSON.parse(rows[i+2]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      let decodings = JSON.parse(rows[i+3]).map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });
      currentTimestep.locationLayer = { modules, decodings };

      i += 4;
      break;
    }
    case 'inputLayer': {
      let activeSynapsesByCell = {};

      let [activeCells, predictedCells, segmentsForActiveCells,
           segmentsForPredictedCells] = JSON.parse(rows[i+1]);

      let cells = activeCells.map(cell => {
        return {
          cell,
          state: predictedCells.indexOf(cell) != -1
            ? 'predicted-active'
            : 'active'
        };
      });

      if (segmentsForActiveCells) {
        activeCells.forEach(cell => {
          activeSynapsesByCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForActiveCells) {
          segmentsForActiveCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
          });
        }
      }

      let {activeCellDecodings, predictedCellDecodings} = JSON.parse(rows[i+2]);

      let activeCellDecodings2 = activeCellDecodings.map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });
      let predictedCellDecodings2 = predictedCellDecodings.map(
        ([objectName, top, left, amountContained]) => {
          return { objectName, top, left, amountContained };
        });

      currentTimestep.inputLayer = {
        cells, activeSynapsesByCell,
        decodings: activeCellDecodings2,
        dimensions: {rows: 32, cols: 150},
        predictedCells: []
      };

      if (timesteps.length > 0 &&
          timesteps[timesteps.length - 1].type == 'move') {
        let prevTimestep = timesteps[timesteps.length - 1];

        let synapsesByPredictedCell = {};

        let cells2 = predictedCells.map(cell => {
          return {
            cell,
            state: 'predicted'
          };
        });

        predictedCells.forEach(cell => {
          synapsesByPredictedCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForPredictedCells) {
          segmentsForPredictedCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            synapsesByPredictedCell[predictedCells[ci]][presynapticLayer] = synapses;
          });
        }

        prevTimestep.inputLayer = {
          predictedCells,
          activeSynapsesByCell: synapsesByPredictedCell,
          decodings: predictedCellDecodings2,
          cells: cells2,
          dimensions: {rows: 32, cols: 150}
        };
      }

      i += 3;
      break;
    }
    case 'objectLayer': {
      let [activeCells, segmentsForActiveCells] = JSON.parse(rows[i+1]);

      let prevActiveCells = (currentTimestep.reset || timesteps.length == 0)
          ? []
          : timesteps[timesteps.length-1].objectLayer.cells.map(d => d.cell);

      let cells = activeCells.map(cell => {
        return {
          cell,
          state: prevActiveCells.indexOf(cell) != -1
            ? 'predicted-active'
            : 'active'
        };
      });

      let activeSynapsesByCell = {};
      if (segmentsForActiveCells) {

        activeCells.forEach(cell => {
          activeSynapsesByCell[cell] = {};
        });

        for (let presynapticLayer in segmentsForActiveCells) {
          segmentsForActiveCells[presynapticLayer].forEach((segments, ci) => {
            let synapses = [];
            segments.forEach(presynapticCells => {
              synapses = synapses.concat(presynapticCells);
            });

            activeSynapsesByCell[activeCells[ci]][presynapticLayer] = synapses;
          });
        }
      }

      let decodings = JSON.parse(rows[i+2]);
      currentTimestep.objectLayer = Object.assign(
        {cells, activeSynapsesByCell, decodings},
        {dimensions: {rows: 16, cols: 256}});
      i += 3;
      break;
    }
    case 'objectPlacements': {
      objectPlacements = JSON.parse(rows[i+1]);

      worldFeatures = [];
      for (let objectName in objects) {
        let objectPlacement = objectPlacements[objectName];
        objects[objectName].forEach(({name, top, left, width, height}) => {
          worldFeatures.push({
            name, width, height,
            top: top + objectPlacement[0],
            left: left + objectPlacement[1]
          });
        });
      }

      i += 2;
      break;
    }
    default:
      if (rows[i] == null || rows[i] == '') {
        i++;
      } else {
        throw `Unrecognized: ${rows[i]}`;
      }
    }
  }

  endTimestep();

  return {
    timesteps, worldDims, configByModule, featureColor, objects
  };
}

let secondColumnLeft = 180;
let secondRowTop = 220;
let thirdRowTop = 428;
let columnWidth = 170;

let boxes = {
  location: {
    left: 0, top: secondRowTop, width: columnWidth, height: 180, text: 'location layer',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 60,
    decodingsLeft: 20, decodingsTop: 85, decodingsWidth: 148, decodingsHeight: 90
  },
  input: {
    left: secondColumnLeft, top: secondRowTop, width: columnWidth, height: 180, text: 'feature-location pair layer',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 60,
    decodingsLeft: 20, decodingsTop: 85, decodingsWidth: 148, decodingsHeight: 90
  },
  object: {
    left: secondColumnLeft, top: 12, width: columnWidth, height: 180, text: 'object layer',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 60,
    decodingsLeft: 20, decodingsTop: 85, decodingsWidth: 148, decodingsHeight: 90
  },
  motion: {
    left: 0, top: thirdRowTop, width: columnWidth, height: 81, text: 'motion input',
    bitsLeft: 0, bitsTop: 0,
    decodingsLeft: 85, decodingsTop: 36,
    secondary: true
  },
  feature: {
    left: secondColumnLeft, top: thirdRowTop, width: columnWidth, height: 81, text: 'feature input',
    bitsLeft: 10, bitsTop: 10, bitsWidth: 150, bitsHeight: 5,
    decodingsLeft: 65, decodingsTop: 30,
    secondary: true
  },
  world: {
    left: 370, top: 120, width: 230, height: 230, text: 'the world'
  }
};

function printRecording(node, text$$1) {
  // Constants
  let margin = {top: 5, right: 5, bottom: 15, left: 5},
      width = 600,
      height = 495,
      parsed = parseData(text$$1);

  // Mutable state
  let iTimestep = 0,
      iLocationModule = null,
      selectedLocationCell = null,
      selectedInputCell = null,
      selectedObjectCell = null,
      highlightedCellsByLayer = {};

  // Allow a mix of SVG and HTML
  let html$$1 = select(node)
        .append('div')
          .style('margin-left', 'auto')
          .style('margin-right', 'auto')
          .style('position', 'relative')
          .style('width', `${width + margin.left + margin.right}px`),
      svg = html$$1.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

  // Add keyboard navigation
  html$$1
    .attr('tabindex', 0)
    .on('keydown', function() {
      switch (event.keyCode) {
      case 37: // Left
        iTimestep--;
        if (iTimestep < 0) {iTimestep = parsed.timesteps.length - 1;}
        onSelectedTimestepChanged();
        event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        onSelectedTimestepChanged();
        event.preventDefault();
        break;
      }
    });

  // Make the SVG a clickable slideshow
  let slideshow = svg.append('g')
      .on('click', () => {
        iTimestep = (iTimestep + 1) % parsed.timesteps.length;
        onSelectedTimestepChanged();
      });

  slideshow.append('rect')
      .attr('fill', 'transparent')
      .attr('stroke', 'none')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.bottom + margin.top + 10);

  let container = slideshow
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

  // Arrange the boxes
  let box = container.selectAll('.box')
      .data([boxes.location,
             boxes.input,
             boxes.object,
             boxes.feature,
             boxes.motion]);

  box = box.enter().append('g')
    .attr('class', 'layerBox')
    .call(g => {
      g.append('rect')
        .attr('class', 'border')
        .attr('fill', 'none');

      g.append('g')
        .attr('class', 'bits');

      g.append('g')
        .attr('class', 'decodings');
    })
    .merge(box)
    .attr('transform', d => `translate(${d.left}, ${d.top})`);

  box.select('.border')
    .attr('width', d => d.width)
    .attr('height', d => d.height)
    .attr('stroke', d => d.secondary ? 'gray' : 'lightgray')
    .attr('stroke-width', d => d.secondary ? 1 : 3)
    .attr('stroke-dasharray', d => d.secondary ? "5,5" : null);

  let [locationNode,
       inputNode,
       objectNode,
       featureNode,
       _] = box.select('.bits')
        .attr('transform', d => `translate(${d.bitsLeft},${d.bitsTop})`)
        .nodes()
        .map(select);
  let [decodedLocationNode,
       decodedInputNode,
       decodedObjectNode,
       decodedFeatureNode,
       motionNode] = box.select('.decodings')
        .attr('transform', d => `translate(${d.decodingsLeft},${d.decodingsTop})`)
        .nodes()
        .map(select);

  let worldNode = container.append('g')
      .attr('transform', `translate(${boxes.world.left}, ${boxes.world.top})`);
  svg.append('line')
    .attr('stroke', 'gray')
    .attr('stroke-width', 1)
    .attr('x1', boxes.world.left - 5)
    .attr('y1', 10)
    .attr('x2', boxes.world.left - 5)
    .attr('y2', 522);

  let timelineNode = html$$1
      .append('div')
      .style('padding-top', '5px')
      .style('padding-left', '17px') // Because it hangs some text off the side.
      .style('padding-right', '17px')
      .style('text-align', 'center');

  // Label the boxes
  let boxLabel = html$$1.selectAll('.boxLabel')
      .data([boxes.location, boxes.input, boxes.object, boxes.motion,
             boxes.feature, boxes.world]);

  boxLabel.enter()
    .append('div')
      .attr('class', 'boxLabel')
      .style('position', 'absolute')
      .style('text-align', 'left')
      .style('font', '10px Verdana')
      .style('pointer-events', 'none')
    .merge(boxLabel)
      .style('left', d => `${d.left + 7}px`)
      .style('top', d => `${d.top - 9}px`)
      .text(d => d.text);

  // Configure the charts
  let locationModules = locationModulesChart()
        .width(boxes.location.bitsWidth)
        .height(boxes.location.bitsHeight)
        .onCellSelected((iModule, cell) => {
          iLocationModule = iModule;
          selectedLocationCell = cell;
          onLocationCellSelected();
        }),
      decodedLocation = decodedLocationsChart()
        .width(boxes.location.decodingsWidth)
        .height(boxes.location.decodingsHeight)
        .color(parsed.featureColor),
      inputLayer = layerOfCellsChart()
        .width(boxes.input.bitsWidth)
        .height(boxes.input.bitsHeight)
        .columnMajorIndexing(true)
        .onCellSelected(cell => {
          selectedInputCell = cell;
          onInputCellSelected();
        }),
      decodedInput = decodedLocationsChart()
        .width(boxes.input.decodingsWidth)
        .height(boxes.input.decodingsHeight)
        .color(parsed.featureColor),
      objectLayer = layerOfCellsChart()
        .width(boxes.object.bitsWidth)
        .height(boxes.object.bitsHeight)
        .onCellSelected(cell => {
          selectedObjectCell = cell;
          onObjectCellSelected();
        }),
      decodedObject = decodedObjectsChart()
        .width(boxes.object.decodingsWidth)
        .height(boxes.object.decodingsHeight)
        .color(parsed.featureColor),
      featureInput = arrayOfAxonsChart()
        .width(boxes.feature.bitsWidth)
        .height(boxes.feature.bitsHeight),
      decodedFeature = featureChart()
        .color(parsed.featureColor)
        .width(40)
        .height(40),
      motionInput = motionChart(),
      world = worldChart()
        .width(boxes.world.width)
        .height(boxes.world.height)
        .color(parsed.featureColor),
      timeline = timelineChart().onchange(iTimestepNew => {
        iTimestep = iTimestepNew;
        onSelectedTimestepChanged();
      });

  calculateHighlightedCells();
  draw();

  //
  // Lifecycle functions
  //
  function draw(incremental) {
    locationNode.datum({
      modules: parsed.timesteps[iTimestep].locationLayer.modules,
      highlightedCells: highlightedCellsByLayer['locationLayer'] || []
    }).call(locationModules);
    decodedLocationNode.datum({
      decodings: parsed.timesteps[iTimestep].locationLayer.decodings,
      objects: parsed.objects
    }).call(decodedLocation);

    inputNode.datum(
      Object.assign(
        {highlightedCells: highlightedCellsByLayer['inputLayer'] || []},
        parsed.timesteps[iTimestep].inputLayer))
      .call(inputLayer);
    decodedInputNode.datum({
      decodings: parsed.timesteps[iTimestep].inputLayer.decodings,
      objects: parsed.objects
    }).call(decodedInput);

    objectNode.datum(
      Object.assign(
        {highlightedCells: highlightedCellsByLayer['objectLayer'] || []},
        parsed.timesteps[iTimestep].objectLayer))
      .call(objectLayer);
    decodedObjectNode.datum({
      decodings: parsed.timesteps[iTimestep].objectLayer.decodings,
      objects: parsed.objects
    }).call(decodedObject);

    featureNode.datum(parsed.timesteps[iTimestep].featureInput)
      .call(featureInput);
    decodedFeatureNode.datum(
      {name: parsed.timesteps[iTimestep].featureInput.decodings[0]})
      .call(decodedFeature);

    motionNode.datum(parsed.timesteps[iTimestep].deltaLocation)
      .call(motionInput);

    worldNode.datum({
      dims: parsed.worldDims,
      location: parsed.timesteps[iTimestep].worldLocation,
      selectedLocationModule: iLocationModule !== null
        ? parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule]
        : null,
      features: parsed.timesteps[iTimestep].worldFeatures,
      selectedLocationCell
    }).call(world);

    timelineNode.datum({
      timesteps: parsed.timesteps,
      selectedIndex: iTimestep
    }).call(incremental ? timeline.drawSelectedStep : timeline);
  }

  function onSelectedTimestepChanged() {
    calculateHighlightedCells();
    drawHighlightedCells();
    draw(true);
  }

  function onLocationCellSelected() {
    if (iLocationModule != null) {
      let config = parsed.configByModule[iLocationModule],
          module = parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule];

      worldNode.datum(d => {
        d.selectedLocationModule = Object.assign({}, config, module);
        d.selectedLocationCell = selectedLocationCell;
        return d;
      });

      let synapsesByPresynapticLayer =
          module.activeSynapsesByCell[selectedLocationCell];
      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    } else {
      worldNode.datum(d => {
        d.selectedLocationModule = null;
        d.selectedLocationCell = null;
        return d;
      });
    }

    worldNode.call(world.drawFiringFields);

    calculateHighlightedCells();
    drawHighlightedCells();
  }

  function onInputCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();
  }

  function onObjectCellSelected() {
    calculateHighlightedCells();
    drawHighlightedCells();
  }

  function calculateHighlightedCells() {
    highlightedCellsByLayer = {};

    // Selected location cell
    if (iLocationModule != null) {
      let module = parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule];

      let synapsesByPresynapticLayer =
          module.activeSynapsesByCell[selectedLocationCell];
      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected input cell
    if (selectedInputCell != null) {
      let layer = parsed.timesteps[iTimestep].inputLayer,
          synapsesByPresynapticLayer =
            layer.activeSynapsesByCell[selectedInputCell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }

    // Selected object cell
    if (selectedObjectCell != null) {
      let layer = parsed.timesteps[iTimestep].objectLayer,
          synapsesByPresynapticLayer =
            layer.activeSynapsesByCell[selectedObjectCell];

      if (synapsesByPresynapticLayer) {
        highlightedCellsByLayer = synapsesByPresynapticLayer;
      }
    }
  }

  function drawHighlightedCells() {
    locationNode.datum(d => {
      d.highlightedCells = highlightedCellsByLayer['locationLayer'] || [];
      return d;
    }).call(locationModules.drawHighlightedCells);

    inputNode.datum(d => {
      d.highlightedCells = highlightedCellsByLayer['inputLayer'] || [];
      return d;
    }).call(inputLayer.drawHighlightedCells);

    objectNode.datum(d => {
      d.highlightedCells = highlightedCellsByLayer['objectLayer'] || [];
      return d;
    }).call(objectLayer.drawHighlightedCells);
  }
}

function printRecordingFromUrl(node, logUrl) {
  text(logUrl,
          (error, contents) =>
          printRecording(node, contents));
}




var locationModuleInference = Object.freeze({
	printRecording: printRecording,
	printRecordingFromUrl: printRecordingFromUrl
});

/**
 *
 * Example timestep:
 * {
 *   worldLocation: {left: 42.0, top: 12.0},
 *   reset: null,
 *   locationLayer: {
 *     modules: [
 *       {activeCells: [],
 *        activePoints: [],
 *        activeSynapsesByCell: {}},
 *     ]
 *   },
 *   deltaLocationInput: {
 *   },
 * };
 */
function parseData$1(text$$1) {
  let timesteps = [],
      rows = text$$1.split('\n');

  // Row: world dimensions
  let worldDims = JSON.parse(rows[0]);

  let currentTimestep = null,
      didReset = false,
      locationInWorld = null;

  // [{cellDimensions: [5,5], moduleMapDimensions: [20.0, 20.0], orientation: 0.2},
  //  ...]
  let configByModule = JSON.parse(rows[1]).map(d => {
    d.dimensions = {rows: d.cellDimensions[0], cols: d.cellDimensions[1]};
    return d;
  });

  function endTimestep() {
    if (currentTimestep !== null) {
      currentTimestep.worldLocation = locationInWorld;
      timesteps.push(currentTimestep);
    }

    currentTimestep = null;
  }

  function beginNewTimestep(type) {
    endTimestep();

    currentTimestep = {
      worldLocation: locationInWorld,
      type
    };

    if (didReset) {
      currentTimestep.reset = true;
      didReset = false;
    }
  }

  let i = 2;
  while (i < rows.length) {
    switch (rows[i]) {
    case 'reset':
      didReset = true;
      i++;
      break;
    case 'move': {
      beginNewTimestep('move');
      let deltaLocation = JSON.parse(rows[i+1]);

      currentTimestep.deltaLocation = {
        top: deltaLocation[0],
        left: deltaLocation[1]
      };

      i += 2;
      break;
    }
    case 'locationInWorld': {
      let location = JSON.parse(rows[i+1]);

      locationInWorld = {top: location[0], left: location[1]};

      i += 2;
      break;
    }
    case 'shift': {
      let modules = [];
      JSON.parse(rows[i+1]).forEach((activeCells, i) => {
        let cells = activeCells.map(cell => {
          return {
            cell,
            state: 'predicted-active'
          };
        });

        modules.push(Object.assign({cells}, configByModule[i]));
      });

      JSON.parse(rows[i+2]).forEach((activePoints, i) => {
        modules[i].activePoints = activePoints;
      });

      currentTimestep.locationLayer = { modules };

      i += 3;
      break;
    }
    default:
      if (rows[i] == null || rows[i] == '') {
        i++;
      } else {
        throw `Unrecognized: ${rows[i]}`;
      }
    }
  }

  endTimestep();

  return {
    timesteps, worldDims, configByModule
  };
}

let boxes$1 = {
  location: {
    left: 0, top: 20, width: 260, height: 140, text: 'location layer',
    bitsLeft: 0, bitsTop: 0, bitsWidth: 260, bitsHeight: 140,
    decodingsLeft: 0, decodingsTop: 0, decodingsWidth: 0, decodingsHeight: 0
  },
  motion: {
    left: 0, top: 190, width: 260, height: 81, text: 'motion input',
    bitsLeft: 0, bitsTop: 0,
    decodingsLeft: 130, decodingsTop: 36,
    secondary: true
  },
  world: {
    left: 280, top: 34, width: 230, height: 230, text: 'the world'
  }
};

function printRecording$1(node, text$$1) {
  // Constants
  let margin = {top: 5, right: 15, bottom: 15, left: 15},
      width = 510,
      height = 270,
      parsed = parseData$1(text$$1);

  // Mutable state
  let iTimestep = 0,
      iLocationModule = null,
      selectedLocationCell = null;

  // Allow a mix of SVG and HTML
  let html$$1 = select(node)
        .append('div')
          .style('margin-left', 'auto')
          .style('margin-right', 'auto')
          .style('position', 'relative')
          .style('width', `${width + margin.left + margin.right}px`),
      svg = html$$1.append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);

  // Add keyboard navigation
  html$$1
    .attr('tabindex', 0)
    .on('keydown', function() {
      switch (event.keyCode) {
      case 37: // Left
        iTimestep--;
          if (iTimestep < 0) {iTimestep = parsed.timesteps.length - 1;}
        draw(true);
        event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%parsed.timesteps.length;
        draw(true);
        event.preventDefault();
        break;
      }
    });

  // Make the SVG a clickable slideshow
  let slideshow = svg.append('g')
      .on('click', () => {
        iTimestep = (iTimestep + 1) % parsed.timesteps.length;
        draw(true);
      });

  slideshow.append('rect')
      .attr('fill', 'transparent')
      .attr('stroke', 'none')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.bottom + margin.top + 10);

  let container = slideshow
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

  // Arrange the boxes
  let box = container.selectAll('.box')
      .data([boxes$1.location,
             boxes$1.motion]);

  box = box.enter().append('g')
    .attr('class', 'layerBox')
    .call(g => {
      g.append('rect')
        .attr('class', 'border')
        .attr('fill', 'none');

      g.append('g')
        .attr('class', 'bits');

      g.append('g')
        .attr('class', 'decodings');
    })
    .merge(box)
    .attr('transform', d => `translate(${d.left}, ${d.top})`);

  box.select('.border')
    .attr('width', d => d.width)
    .attr('height', d => d.height)
    .attr('stroke', d => d.secondary ? 'gray' : 'lightgray')
    .attr('stroke-width', d => d.secondary ? 1 : 3)
    .attr('stroke-dasharray', d => d.secondary ? "5,5" : null);

  let [locationNode,
       _] = box.select('.bits')
        .attr('transform', d => `translate(${d.bitsLeft},${d.bitsTop})`)
        .nodes()
        .map(select);
  let [decodedLocationNode,
       motionNode] = box.select('.decodings')
        .attr('transform', d => `translate(${d.decodingsLeft},${d.decodingsTop})`)
        .nodes()
        .map(select);

  let worldNode = container.append('g')
      .attr('transform', `translate(${boxes$1.world.left}, ${boxes$1.world.top})`);

  // Label the boxes
  let boxLabel = html$$1.selectAll('.boxLabel')
      .data([boxes$1.location, boxes$1.motion, boxes$1.world]);

  boxLabel.enter()
    .append('div')
      .attr('class', 'boxLabel')
      .style('position', 'absolute')
      .style('text-align', 'left')
      .style('font', '10px Verdana')
      .style('pointer-events', 'none')
    .merge(boxLabel)
      .style('left', d => `${d.left + 17}px`)
      .style('top', d => `${d.top - 9}px`)
      .text(d => d.text);

  // Configure the charts
  let locationModules = locationModulesChart()
        .width(boxes$1.location.bitsWidth)
        .height(boxes$1.location.bitsHeight)
        .onCellSelected((iModule, cell) => {
          iLocationModule = iModule;
          selectedLocationCell = cell;
          onLocationCellSelected();
        }),
      motionInput = motionChart(),
      world = worldChart()
        .width(boxes$1.world.width)
        .height(boxes$1.world.height)
        .color(parsed.featureColor);

  draw();

  //
  // Lifecycle functions
  //
  function draw(incremental) {
    locationNode.datum({
      modules: parsed.timesteps[iTimestep].locationLayer.modules
    }).call(locationModules);

    motionNode.datum(parsed.timesteps[iTimestep].deltaLocation)
      .call(motionInput);

    worldNode.datum({
      dims: parsed.worldDims,
      location: parsed.timesteps[iTimestep].worldLocation,
      selectedLocationModule: iLocationModule !== null
        ? parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule]
        : null,
      features: [],
      selectedLocationCell
    }).call(world);
  }

  function onLocationCellSelected() {
    worldNode.datum(d => {
      d.selectedLocationModule = iLocationModule !== null
        ? Object.assign(
          {},
          parsed.configByModule[iLocationModule],
          parsed.timesteps[iTimestep].locationLayer.modules[iLocationModule])
        : null;
      d.selectedLocationCell = selectedLocationCell;
      return d;
    }).call(world.drawFiringFields);
  }
}

function printRecordingFromUrl$1(node, csvUrl) {
  text(csvUrl,
          (error, contents) =>
          printRecording$1(node, contents));
}




var locationModules = Object.freeze({
	printRecording: printRecording$1,
	printRecordingFromUrl: printRecordingFromUrl$1
});

function Grid2dLayout(nColumns, nRows, left, top, width, height, padding) {
  this.nColumns = nColumns;
  this.nRows = nRows;
  this.left = left;
  this.top = top;
  this.width = width;
  this.height = height;
  this.columnWidth = width / nColumns;
  this.rowHeight = height / nRows;
  this.padding = padding;
}

Grid2dLayout.prototype.getBitTopLeft = function(bitId) {
  var column = Math.floor(bitId / this.nRows),
      row = bitId % this.nRows;
  return [column * this.columnWidth, row * this.rowHeight];
};

Grid2dLayout.prototype.getBitCenter = function(bitId) {
  var ret = this.getBitTopLeft(bitId);
  ret[0] += this.columnWidth/2;
  ret[1] += this.rowHeight/2;
  return ret;
};

Grid2dLayout.prototype.getBitAtPoint = function(x,y) {
  var column = Math.floor(x / this.columnWidth),
      row = Math.floor(y / this.rowHeight),
      bitId = column*this.nRows + row;

  return bitId;
};

function selectedCellPlot() {
  var distalSegmentWidth = 15,
      distalSegmentHeight = 3,
      selectedCellR = 7,
      layout,
      layouts;

  var chart = function chart (selection$$1) {
    selection$$1.each(function(selectedCells) {
      var hoverContainer = select(this);

      var selectedCell = hoverContainer.selectAll('.selectedCell')
          .data(selectedCells);

      selectedCell.exit()
        .remove();

      selectedCell = selectedCell.enter()
        .append('g')
        .attr('class', 'selectedCell')
        .call(enter => {
          enter
            .append('g')
            .attr('class', 'cellIcon')
            .append('polygon')
            .attr('points', '0,-8 7,8 -7,8')
            .attr('fill', 'black');
        })
        .merge(selectedCell);

      selectedCell
        .select('.cellIcon')
        .attr('transform', d => {
          var pos = layout.getBitCenter(d.cellId);
          return `translate(${layout.left + pos[0]},${layout.top + pos[1]})`;
        });

      var synapsesForCell = hoverContainer.selectAll('.synapsesForCell')
          .data(selectedCells);

      synapsesForCell.exit()
        .remove();

      synapsesForCell = synapsesForCell.enter()
        .append('g')
        .attr('class', 'synapsesForCell')
        .merge(synapsesForCell);

      var presynapticCell = synapsesForCell.selectAll('.presynapticCell')
          .data(function(d, i) {
            var synapses = [];
            d.distalSegments.forEach(function(synapsesByState) {
              ['active'].forEach(function(state) {
                synapsesByState[state].forEach(function(synapse) {
                  synapses.push({
                    state: state,
                    presynapticLayer: synapse[0],
                    presynapticBitId: synapse[1],
                    postsynapticCellId: d.cellId
                  });
                });
              });
            });
            return synapses;
          });

      presynapticCell.exit()
        .remove();

      presynapticCell = presynapticCell.enter()
        .append('g')
        .attr('class', 'presynapticCell')
        .call(enter => {
          enter.append('polygon')
            .attr('points', '0,-6 4,3 -4,3')
            .attr('fill', 'crimson')
            .attr('opacity', 0.5);
        })
        .merge(presynapticCell);

      presynapticCell
        .attr('transform', d => {
          var lay = layouts[d.presynapticLayer];
          var position = lay.getBitCenter(d.presynapticBitId);

          return `translate(${lay.left + position[0]},${lay.top + position[1]})`;
        });


    });
  };

  chart.layout = function(_) {
    if (!arguments.length) return layout;
    layout = _;
    return chart;
  };

  chart.layouts = function(_) {
    if (!arguments.length) return layouts;
    layouts = _;
    return chart;
  };

  return chart;
}


function objectPlot() {
  var rowHeight,
      columnWidth,
      color$$1;

  var chart = function chart (selection$$1) {

    var point = selection$$1.selectAll('.point')
        .data(d => d);

    point.exit()
      .remove();

    point = point.enter()
      .append('g')
      .attr('class', 'point')
      .call(enter => {
        enter.append('rect')
          .attr('class', 'featureColor')
          .attr('x', 0)
          .attr('y', 0)
          .attr('width', columnWidth)
          .attr('height', rowHeight)
          .attr('stroke', 'none');

        enter.append('text')
          .attr('class', 'featureText')
          .attr('text-anchor', 'middle')
          .attr('dy', rowHeight * 0.25)
          .attr('x', columnWidth / 2)
          .attr('y', rowHeight / 2)
          .attr('fill', 'white')
          .style('font', `bold ${rowHeight * 0.8}px monospace`);
      })
      .merge(point)
      .attr('transform', (d, i) =>
            `translate(${columnWidth*d[0][1]},${columnWidth*d[0][0]})`);

    point.select('.featureColor')
      .attr('fill', d => {
        if (d) {
          return color$$1(d[1]);
        } else {
          return 'none';
        }
      });

    point.select('.featureText')
      .text(d => d[1]);
  };

  chart.rowHeight = function(_) {
    if (!arguments.length) return rowHeight;
    rowHeight = _;
    return chart;
  };

  chart.columnWidth = function(_) {
    if (!arguments.length) return columnWidth;
    columnWidth = _;
    return chart;
  };

  chart.color = function(_) {
    if (!arguments.length) return color$$1;
    color$$1 = _;
    return chart;
  };

  return chart;
}


function layerOfCellsPlot() {
  var onCellHover;

  var chart = function chart (selection$$1) {
    selection$$1.each(function(d) {
      var hoveredCell = null,
          timestep = d.timestep,
          layout = d.layout,
          layerName = d.layerName;

      var chartContainer = select(this).selectAll('.chartContainer')
          .data([timestep]);

      chartContainer = chartContainer.enter()
        .append('g')
        .attr('class', 'chartContainer')
        .call(function(enter) {
          enter.append('g')
            .attr('class', 'chartBack');
          enter.append('g')
            .attr('class', 'chartMiddle');
          enter.append('g')
            .attr('class', 'chartFront');
          enter.append('g')
            .attr('class', 'chartFront2');
          enter.append('g')
            .attr('class', 'selectedCellContainer');
          enter.append('rect')
            .attr('class', 'mouseCapture')
            .attr('width', layout.width)
            .attr('height', layout.height)
            .attr('fill', 'transparent')
            .attr('stroke', 'none');
        }).merge(chartContainer);

      var back = chartContainer.select('.chartBack'),
          middle = chartContainer.select('.chartMiddle'),
          front = chartContainer.select('.chartFront'),
          front2 = chartContainer.select('.chartFront2');

      chartContainer.select('.mouseCapture')
        .on('mouseout', function() {
          if (hoveredCell != null) {
            hoveredCell = null;
            onCellHover(layerName, null);
          }
        })
        .on('mousemove', function() {
          var box = chartContainer.select('.mouseCapture').node().getBoundingClientRect(),
          x = event.clientX - box.left,
          y = event.clientY - box.top,
          nearestCell = null,
          nearestD2 = null;

          if (timestep) {
            timestep.activeCells.forEach(function(c) {
              var pos = layout.getBitCenter(c.cellId),
              d2 = Math.pow(pos[1] - y, 2) + Math.pow(pos[0] - x, 2);

              if (nearestD2 == null || d2 < nearestD2) {
                nearestD2 = d2;
                nearestCell = c;
              }
            });
          }

          if (hoveredCell != nearestCell) {
            hoveredCell = nearestCell;
            onCellHover(layerName, hoveredCell);
          }
        });

      if (timestep) {

        var activeCell = front.selectAll('.activeCell')
          .data(timestep['activeCells']);

        activeCell.exit()
          .remove();

        activeCell = activeCell.enter()
          .append('g')
          .attr('class', 'activeCell')
          .call(enter => {
            enter.append('polygon')
              .attr('points', '0,-4 2,2 -2,2')
              .attr('stroke', 'none')
              .attr('fill', 'black');
          })
          .merge(activeCell);

        activeCell.attr('transform', d =>
                        `translate(${layout.getBitCenter(d.cellId).join(',')})`);
      }
    });
  };

  chart.onCellHover = function(_) {
    if (!arguments.length) return onCellHover;
    onCellHover = _;
    return chart;
  };

  return chart;
}

function printRecordingFromUrl$2(node, csvUrl) {
  text(csvUrl,
          function (error, contents) {
            return printRecording$2(node, contents);
          });
}

function printRecording$2(node, csv$$1) {

  // Data from the CSV
  var worldDiameter,
      objects,
      timesteps = [],
      featureColor = ordinal();

  // PARSE THE CSV
  (function() {
    var rows = csvParseRows(csv$$1);

    // First row: diameter
    worldDiameter = parseInt(rows[0]);

    // Second row: Features and colors
    //   {'A': 'red',
    //    'B': 'blue',
    //    'C': 'gray'}
    var featureColorMapping = JSON.parse(rows[1]);

    var features = [],
        colors = [];

    for (var feature in featureColorMapping) {
      features.push(feature);
      colors.push(featureColorMapping[feature]);
    }

    featureColor
      .domain(features)
      .range(colors);

    // Third row: Objects
    // {
    //   'Object 1': [
    //     [[0,0], 'A'],
    //     [[0,1], 'B'],
    //     [[1,0], 'A'],
    //   ],
    //   'Object 2': []
    // };
    objects = JSON.parse(rows[2]);

    var currentTimestep = null;
    var didReset = false;
    var objectPlacements = null;

    var i = 3;
    while (i < rows.length) {
      switch (rows[i][0]) {
      case "reset":
        didReset = true;
        i++;
        break;
      case "t":
        if (currentTimestep !== null) {
          currentTimestep.objectPlacements = objectPlacements;

          timesteps.push(currentTimestep);
        }
        currentTimestep = {
          layers: {}
        };

        if (didReset) {
          currentTimestep.reset = true;
          didReset = false;
        }
        i++;
        break;
      case "input":
        let inputName = rows[i][1];
        currentTimestep.layers[inputName] = {
          activeBits: JSON.parse(rows[i+1]),
          decodings: JSON.parse(rows[i+2])
        };

        i += 3;
        break;
      case "layer":
        let layerName = rows[i][1];

        currentTimestep.layers[layerName] = {
          activeCells: JSON.parse(rows[i+1]).map(cellAndSegments => {
            return {
              cellId: cellAndSegments[0],
              distalSegments: cellAndSegments[1].map(segment => {
                var synapses = [];
                segment.forEach(presynapticLayerAndIndices => {
                  var presynapticLayer = presynapticLayerAndIndices[0];
                  presynapticLayerAndIndices[1].forEach(presynapticCell => {
                    synapses.push([presynapticLayer, presynapticCell]);
                  });
                });

                return {active: synapses};
              })
            };
          }),
          decodings: JSON.parse(rows[i+2])
        };

        i += 3;
        break;
      case "objectPlacements":
        let objectPlacementsDict = JSON.parse(rows[i+1]);

        objectPlacements = [];
        for (let k in objectPlacementsDict) {
          objectPlacements.push({
            name: k,
            offset: objectPlacementsDict[k]
          });
        }

        i += 2;
        break;
      case "egocentricLocation":
        currentTimestep.egocentricLocation = JSON.parse(rows[i+1]);
        i += 2;
        break;
      default:
        i++;
        break;
      }
    }

    if (currentTimestep !== null) {
      currentTimestep.objectPlacements = objectPlacements;

      timesteps.push(currentTimestep);
    }
  })();

  //
  // CONSTANTS
  //
  var drawSynapses = false,
      inputs = ['newLocation', 'deltaLocation', 'feature'],
      layers = ['location', 'input', 'object'],
      layouts = {
        world: new Grid2dLayout(
          worldDiameter, worldDiameter,
          630, 120,
          288, 288,
          {left: 0, right: 0, top: 0, bottom: 0}),
        object: new Grid2dLayout(
          256, 16,
          375, 12,
          150, 60,
          {top: 10, right: 10, bottom: 110, left: 10}),
        input: new Grid2dLayout(
          150, 32,
          375, 212,
          150, 60,
          {top: 10, right: 10, bottom: 110, left: 10}),
        location: new Grid2dLayout(
          40, 25,
          130, 212,
          150, 60,
          {top: 10, right: 10, bottom: 110, left: 10}),
        deltaLocation: new Grid2dLayout(
          40, 25,
          12, 212,
          40, 25,
          {top: 10, right: 10, bottom: 60, left: 10}),
        newLocation: new Grid2dLayout(
          40, 25,
          165, 412,
          80, 25,
          {top: 10, right: 45, bottom: 90, left: 45}),
        feature: new Grid2dLayout(
          150, 1,
          375, 412,
          150, 1,
          {top: 10, right: 10, bottom: 70, left: 10})
      };

  //
  // SHARED STATE
  //
  var iTimestep = 0,
      onSelectedTimestepChanged = [], // callbacks
      width = 985,
      height = 630,
      brainLeft = 0,
      brainTop = 5,
  html$$1 = select(node)
    .append('div')
    .attr('tabindex', 0)
    .style('position', 'relative')
    .style('height', height + 'px')
    .style('width', width + 'px')
    .on('keydown', function() {
      switch (event.keyCode) {
      case 37: // Left
        iTimestep--;
        if (iTimestep < 0) {iTimestep = timesteps.length - 1;}
        onSelectedTimestepChanged.forEach(function(f) { f(); });
        event.preventDefault();
        break;
      case 39: // Right
        iTimestep = (iTimestep+1)%timesteps.length;
        onSelectedTimestepChanged.forEach(function(f) { f(); });
        event.preventDefault();
        break;
      }
    }),
  svg = html$$1.append('svg')
      .attr('width', width)
      .attr('height', height)
      .style('max-width', 'none') // jupyter notebook tries to set this
      .style('max-height', 'none');

  svg.append('defs')
    .append('marker')
      .attr('id', 'arrow')
      .attr('markerWidth', 2)
      .attr('markerHeight', 4)
      .attr('refX', 0.1)
      .attr('refY', 2)
      .attr('orient', 'auto')
      .attr('markerUnits', 'strokeWidth')
    .append('path')
    .attr('d', 'M0,0 V4 L2,2 Z');

  var slideshow = svg
      .append('g')
      .on('click', () => {
        iTimestep = (iTimestep+1)%timesteps.length;
        onSelectedTimestepChanged.forEach(function(f) { f(); });
      });

  slideshow.append('rect')
    .attr('fill', 'transparent')
    .attr('stroke', 'none')
    .attr('width', width)
    .attr('height', 600);

  var brain = slideshow
      .append('g')
      .attr('transform', `translate(${brainLeft},${brainTop})`);


  layouts.decodedLocation = new Grid2dLayout(
    worldDiameter, worldDiameter,
    layouts.location.left + 35,
    layouts.location.top + layouts.location.height + 20,
    80, 80);

  layouts.decodedNewLocation = new Grid2dLayout(
    worldDiameter, worldDiameter,
    layouts.newLocation.left + 10,
    layouts.newLocation.top + layouts.newLocation.height + 20,
    60, 60);

  layouts.decodedInput = new Grid2dLayout(
    worldDiameter, worldDiameter,
    layouts.input.left + 35,
    layouts.input.top + layouts.input.height + 20,
    80, 80);


  //
  // timesteps = [
  //   {
  //     layers: {
  //       location: {
  //         activeCells: [{cellId: 42,
  //                        showSynapses: false,
  //                        distalSegments: [{
  //                          active: [["deltaLocation", 13],
  //                                   ["deltaLocation", 14]]
  //                        }]}],
  //         activeColumns: [10]
  //       }
  //     },
  //     senses: {
  //       deltaLocation: {
  //         activeBits: [42, 43]
  //       }
  //     }
  //   }
  // ]



  //
  // LAYERS
  //
  (function() {
    let chart = layerOfCellsPlot()
      .onCellHover(drawHoveredCell);

    function draw() {
      var layer = brain.selectAll('.layer')
          .data(layers.map(layerName => {
            return {
              layerName: layerName,
              layout: layouts[layerName],
              timestep: timesteps[iTimestep].layers[layerName]
            };
          }));

      layer = layer.enter()
        .append('g')
        .attr('class', 'layer')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'node')
            .attr('fill', 'none')
            .attr('stroke', 'lightgray')
            .attr('stroke-width', 3)
            .attr('x', d => d.layout.left - d.layout.padding.left)
            .attr('y', d => d.layout.top - d.layout.padding.top)
            .attr('width', d => d.layout.width + d.layout.padding.left + d.layout.padding.right)
            .attr('height', d => d.layout.height + d.layout.padding.top + d.layout.padding.bottom);

          enter
            .append('g')
              .attr('class', 'layerOfCells')
              .attr('transform', d =>
                  `translate(${d.layout.left},${d.layout.top})`)
            .append('rect')
              .attr('x', -2)
              .attr('y', -2)
              .attr('width', d => d.layout.width + 4)
              .attr('height', d => d.layout.height + 4)
              .attr('fill', 'none')
              .attr('stroke', 'black');
        })
        .merge(layer);

      layer.select('.layerOfCells')
        .call(chart);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();


  //
  // INPUTS
  //
  (function() {

    function draw() {
      var input = brain.selectAll('.input')
          .data(inputs.map(inputName => {
            return {
              layerName: inputName,
              layout: layouts[inputName],
              timestep: timesteps[iTimestep].layers[inputName]
            };
          }));

      input = input.enter()
        .append('g')
        .attr('class', 'input')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'node')
            .attr('fill', 'none')
            .attr('stroke', 'gray')
            .attr('stroke-dasharray', "5, 5")
            .attr('x', d => d.layout.left - d.layout.padding.left)
            .attr('y', d => d.layout.top - d.layout.padding.top)
            .attr('width', d => d.layout.width + d.layout.padding.left + d.layout.padding.right)
            .attr('height', d => d.layout.height + d.layout.padding.top + d.layout.padding.bottom);

          enter
            .append('g')
              .attr('class', 'inputAxons')
              .attr('transform', d =>
                    `translate(${d.layout.left},${d.layout.top})`)
            .append('rect')
              .attr('x', -2)
              .attr('y', -2)
              .attr('width', d => d.layout.width + 4)
              .attr('height', d => d.layout.height + 4)
              .attr('fill', 'none')
              .attr('stroke', 'black');
        })
        .merge(input);

      var activeAxon = input.select('.inputAxons')
          .selectAll('.activeAxon')
          .data(d => d.timestep.activeBits.map(bit => {
            return {
              layout: d.layout, bit: bit
            };
          }));

      activeAxon.exit()
        .remove();

      activeAxon = activeAxon.enter()
        .append('g')
        .attr('class', 'activeAxon')
        .call(enter => {
          enter.append('circle')
            .attr('r', 1.5)
            .attr('stroke', 'none')
            .attr('fill', 'black');
        })
        .merge(activeAxon)
        .attr('transform', d =>
              `translate(${d.layout.getBitCenter(d.bit).join(',')})`);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();


  //
  // DECODED FEATURE
  //
  (function() {

    function draw() {
      var decodedFeatures = brain
          .selectAll('.decodedFeatures')
          .data(['feature'].map(name => {
            return {
              name: name,
              sourceLayout: layouts[name],
              timestep: timesteps[iTimestep].layers[name]
            };
          }));

      decodedFeatures.exit()
        .remove();

      decodedFeatures = decodedFeatures.enter()
        .append('g')
        .attr('class', 'decodedFeatures')
        .merge(decodedFeatures)
        .attr('transform', d =>
              `translate(${d.sourceLayout.left},${d.sourceLayout.top + d.sourceLayout.height + 20})`);

      var decodedFeature = decodedFeatures.selectAll('.decodedFeature')
          .data(d => [d.timestep.decodings[0]]);

      decodedFeature = decodedFeature.enter()
        .append('g')
        .attr('class', 'decodedFeature')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'featureColor')
            .attr('width', d => 40)
            .attr('height', d => 40)
            .attr('fill', 'none')
            .attr('stroke', 'none');

          enter.append('text')
            .attr('class', 'featureText')
            .attr('text-anchor', 'middle')
            .attr('dy', 8)
            .attr('x', 20)
            .attr('y', 20)
            .attr('fill', 'white')
            .style('font', 'bold 26px monospace');
        })
        .merge(decodedFeature)
        .attr('transform', 'translate(55,0)');

      decodedFeature.select('.featureColor')
        .attr('fill', d => featureColor(d));

      decodedFeature.select('.featureText')
          .text(d => d);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();

  //
  // DECODED OBJECT
  //
  (function() {

    function draw() {
      var decodedObjects = brain
          .selectAll('.decodedObjects')
          .data(['object'].map(name => {
            return {
              name: name,
              sourceLayout: layouts[name],
              timestep: timesteps[iTimestep].layers[name]
            };
          }));

      decodedObjects.exit()
        .remove();

      decodedObjects = decodedObjects.enter()
        .append('g')
        .attr('class', 'decodedObjects')
        .merge(decodedObjects)
        .attr('transform', d =>
              `translate(${d.sourceLayout.left + 10},${d.sourceLayout.top + d.sourceLayout.height + 20})`);

      var decodedObjectRow = decodedObjects.selectAll('.decodedObjectRow')
          .data(d => {
            let decodings = d.timestep.decodings;

            var rows = [];
            var i = 0;
            for (; i + 3 < decodings.length; i += 3) {
              rows.push([decodings[0], decodings[1], decodings[2]]);
            }

            if (i < decodings.length) {
              let lastRow = [];
              for (; i < decodings.length; i++) {
                lastRow.push(decodings[i]);
              }
              rows.push(lastRow);
            }

            return rows;
          });

      decodedObjectRow.exit()
        .remove();

      decodedObjectRow = decodedObjectRow.enter()
        .append('g')
        .attr('class', 'decodedObjectRow')
        .merge(decodedObjectRow)
        .attr('transform', (d, i) => {
          var y = (i == 0) ? 0 : 30*i + 10;
          return `translate(0,${y})`;
        });

      var decodedObject = decodedObjectRow.selectAll('.decodedObject')
          .data(d => d);

      decodedObject.exit()
        .remove();

      decodedObject = decodedObject.enter()
        .append('g')
        .attr('class', 'decodedObject')
        .merge(decodedObject)
        .attr('transform', (d, i) => `translate(${i*50},0)`);

      decodedObject
        .datum(d => objects[d])
        .call(objectPlot()
              .rowHeight(10)
              .columnWidth(10)
              .color(featureColor));
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();

  //
  // DECODED DELTA LOCATION
  //
  (function() {

    function draw() {
      var decodedDeltaLocations = brain
          .selectAll('.decodedDeltaLocations')
          .data(['deltaLocation'].map(name => {
            return {
              name: name,
              sourceLayout: layouts[name],
              timestep: timesteps[iTimestep].layers[name]
            };
          }));

      decodedDeltaLocations.exit()
        .remove();

      decodedDeltaLocations = decodedDeltaLocations.enter()
        .append('g')
        .attr('class', 'decodedDeltaLocations')
        .merge(decodedDeltaLocations)
        .attr('transform', d =>
              `translate(${d.sourceLayout.left},${d.sourceLayout.top + d.sourceLayout.height + 20})`);

      var decodedDeltaLocation = decodedDeltaLocations.selectAll('.decodedDeltaLocation')
          .data(d => d.timestep.decodings);

      decodedDeltaLocation.exit()
        .remove();

      decodedDeltaLocation = decodedDeltaLocation.enter()
        .append('g')
        .attr('class', 'decodedDeltaLocation')
        .call(enter => {
          enter.append('g')
            .attr('class', 'arrowTransform')
            .append('line')
            .attr('class', 'deltaLocationArrow')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', 20)
            .attr('y2', 0)
            .attr('stroke', '#000')
            .attr('stroke-width', 5)
            .attr('marker-end', 'url(#arrow)');
        })
        .merge(decodedDeltaLocation)
        .attr('transform', 'translate(12,12)');

      decodedDeltaLocation
        .select('.arrowTransform')
        .attr('transform', d => {
          var radians = Math.atan(-d[0] / d[1]);
          var degrees = radians * 180 / Math.PI;
          if (d[1] < 0) {
            degrees += 180;
          }
          return `rotate(${-degrees} 10 0)`;
        });
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();


  //
  // DECODED LOCATIONS
  //
  (function() {

    function draw() {
      var decodedLocations = brain
        .selectAll('.decodedLocations')
          .data([['location', 'decodedLocation'],
                 ['newLocation', 'decodedNewLocation'],
                 ['input', 'decodedInput']].map(names => {
                   let layerName = names[0];
                   let name = names[1];
                   return {
                     name: name,
                     timestep: timesteps[iTimestep].layers[layerName],
                     layout: layouts[name]
                   };
                 }));

      decodedLocations = decodedLocations.enter()
        .append('g')
        .attr('class', 'decodedLocations')
            .call(enter => {
              enter.append('rect')
                .attr('width', d => d.layout.width)
                .attr('height', d => d.layout.height)
                .attr('fill', 'none')
                .attr('stroke', 'lightgray')
                .attr('stroke-width', 1);

              enter.selectAll('.verticalLine')
                .data(d => sequence(d.layout.nColumns).map(i => {
                  return {
                    layout: d.layout,
                    i: i
                  };
                }))
                .call(function(verticalLine) {
                  verticalLine.enter()
                    .append('line')
                      .attr('class', 'verticalLine')
                      .attr('x1', function(d, i) { return d.i*d.layout.columnWidth; })
                      .attr('y1', 0)
                      .attr('x2', function(d, i) { return d.i*d.layout.columnWidth; })
                      .attr('y2', d => d.layout.height)
                      .attr('stroke', 'lightgray')
                      .attr('stroke-width', 1);

                  verticalLine.exit()
                    .remove();
                });

              enter.selectAll('.horizontalLine')
                .data(d => sequence(d.layout.nRows).map(i => {
                  return {
                    layout: d.layout,
                    i: i
                  };
                }))
                .call(function(horizontalLine) {
                  horizontalLine.enter()
                    .append('line')
                    .attr('class', 'horizontalLine')
                    .attr('x1', 0)
                    .attr('y1', function(d, i) { return d.i*d.layout.rowHeight; })
                    .attr('x2', d => d.layout.width)
                    .attr('y2', function(d, i) { return d.i*d.layout.rowHeight; })
                    .attr('stroke', 'lightgray')
                    .attr('stroke-width', 1);

              horizontalLine.exit()
                .remove();
            });
        })
        .merge(decodedLocations)
        .attr('transform', d =>
              `translate(${d.layout.left},${d.layout.top})`);

      var currentLocation = decodedLocations.selectAll('.currentLocation')
          .data(d => d.timestep.decodings.map(decoding => {
            if (d.name == "decodedInput") {
              return {
                location: decoding[1],
                feature: decoding[0],
                layout: d.layout
              };
            } else {
              return {
                location: decoding,
                layout: d.layout
              };
            }
          }));

      currentLocation.exit()
        .remove();

      currentLocation = currentLocation.enter()
        .append('g')
        .attr('class', 'currentLocation')
        .call(enter => {
          enter.append('rect')
            .attr('class', 'featureColor')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', d => d.layout.columnWidth)
            .attr('height',d => d.layout.rowHeight)
            .attr('stroke', 'none');

          enter.append('text')
            .attr('class', 'featureText')
            .attr('text-anchor', 'middle')
            .attr('dy', d => d.layout.rowHeight * 0.35)
            .attr('x', d => d.layout.columnWidth/2)
            .attr('y', d => d.layout.rowHeight/2)
            .attr('fill', 'white')
            .style('font', 'bold 8px monospace');
        })
        .merge(currentLocation);

      currentLocation.attr('transform', d => {
        var position = [d.layout.columnWidth*d.location[1],
                        d.layout.rowHeight*d.location[0]];
        return `translate(${position[0]}, ${position[1]})`;
      });

      currentLocation.select('.featureColor')
        .attr('fill', d => {
          if (d.feature) {
            return featureColor(d.feature);
          } else {
            return 'black';
          }
        });

      currentLocation.select('.featureText')
        .text(d => d.feature);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();

  var hoveredCellContainer = brain.append('g')
      .attr('class', 'hoveredCellContainer')
      .style('pointer-events', 'none');

  function drawHoveredCell(layerName, hoveredCell) {
    hoveredCellContainer
      .datum(hoveredCell ? [hoveredCell] : [])
      .call(selectedCellPlot().layout(layouts[layerName]).layouts(layouts));
  }

  //
  // LABELS
  //

  [{layout: layouts.object,
    // decodedLayout: layouts.decodedLocation,
    text: 'object layer'},
   {layout: layouts.input,
    text: 'feature-location pair layer'},
   {layout: layouts.location,
    text: 'location layer'},
   {layout: layouts.deltaLocation,
    text: 'motor input'},
   {layout: layouts.world,
    text: 'egocentric space'},
   {layout: layouts.newLocation,
    text: 'egocentric location input'},
   {layout: layouts.feature,
    text: 'feature input'}].forEach(function(d) {
      html$$1.append('div')
        .style('position', 'absolute')
        .style('width', '50px')
        .style('left', `${brainLeft + d.layout.left + d.layout.width + d.layout.padding.right + 6}px`)
        .style('top', `${d.layout.top - d.layout.padding.top}px`)
        .style('text-align', 'left')
        .style('font', '10px Verdana')
        .style('pointer-events', 'none')
        .text(d.text);
    });

  (function() {

    var xOffset = 0;

    timesteps.forEach((timestep, i) => {
      if (i > 0 && timestep.reset) {
        xOffset += 4;
      }

      timestep.xOffset = xOffset;

      xOffset += 12;
    });

    var time = svg.append('g')
        .attr('class', 'time')
        .attr('transform', `translate(${width/2 - xOffset/2}, 600)`);

    function drawTime() {

      var timestepMarker = time.selectAll('.timestepMarker')
          .data(timesteps);

      timestepMarker.exit()
        .remove();

      timestepMarker = timestepMarker.enter()
        .append('g')
        .attr('transform', (d, i) => `translate(${d.xOffset},0)`)
        .attr('class', 'timestepMarker')
        .call(enter => {
          enter.append('circle')
            .attr('class', 'regular')
            .attr('r', 5)
            .attr('cx', 5)
            .attr('cy', 5)
            .attr('stroke', 'none')
            .style('cursor', 'pointer')
            .on('click', function(d, i) {
              iTimestep = i;
              onSelectedTimestepChanged.forEach(function(f) { f(); });
            });
        })
        .merge(timestepMarker)
        .attr('fill', (d, i) => i == iTimestep ? 'black' : 'lightgray');

      let resets = [];
      timesteps.forEach((timestep, i) => {
        if (i > 0 && timestep.reset) {
          resets.push(i);
        }
      });

      var resetMarker = time.selectAll('.resetMarker')
          .data(resets);

      resetMarker = resetMarker.enter()
        .append('rect')
        .attr('class', 'resetMarker')
        .attr('height', 14)
        .attr('width', 2)
        .attr('y', -2)
        .attr('fill', 'gray')
        .merge(resetMarker)
        .attr('x', d => timesteps[d].xOffset - 4);
    }

    onSelectedTimestepChanged.push(drawTime);
      drawTime();

  })();

  //
  // THE WORLD
  //
  (function() {
    let worldLayout = layouts.world;

    svg.append('line')
      .attr('stroke', 'gray')
      .attr('stroke-width', 1)
      .attr('x1', worldLayout.left - 25)
      .attr('y1', 10)
      .attr('x2', worldLayout.left - 25)
      .attr('y2', 550);

    let world = slideshow.append('g')
        .attr('transform', `translate(${worldLayout.left},${worldLayout.top})`);

    world.append('rect')
      .attr('width', d => worldLayout.width)
      .attr('height', d => worldLayout.height)
      .attr('fill', 'none')
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);

    world.selectAll('.verticalLine')
      .data(sequence(worldDiameter))
      .enter()
      .append('line')
      .attr('class', 'verticalLine')
      .attr('x1', i => i * worldLayout.columnWidth)
      .attr('y1', 0)
      .attr('x2', i => i * worldLayout.columnWidth)
      .attr('y2', worldLayout.height)
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);

    world.selectAll('.horizontalLine')
      .data(sequence(worldDiameter))
      .enter()
      .append('line')
      .attr('class', 'horizontalLine')
      .attr('x1', 0)
      .attr('y1', i => i * worldLayout.rowHeight)
      .attr('x2', d => worldLayout.width)
      .attr('y2', i => i * worldLayout.rowHeight)
      .attr('stroke', 'lightgray')
      .attr('stroke-width', 1);


    function draw() {
      var objectPlacements = timesteps[iTimestep].objectPlacements || [];
      var egocentricLocation = timesteps[iTimestep].egocentricLocation;

      var placedObject = world
          .selectAll('.placedObject')
          .data(objectPlacements);

      placedObject = placedObject
        .enter()
        .append('g')
        .attr('class', 'placedObject')
        .merge(placedObject);

      placedObject
        .attr('transform', d => {
          var position = [worldLayout.columnWidth*d.offset[1],
                          worldLayout.rowHeight*d.offset[0]];
          return `translate(${position[0]}, ${position[1]})`;
        })
        .datum(d => objects[d.name])
        .call(objectPlot()
              .rowHeight(worldLayout.rowHeight)
              .columnWidth(worldLayout.columnWidth)
              .color(featureColor));

      var currentLocation = world
          .selectAll('.currentLocation')
          .data([egocentricLocation]);

      currentLocation = currentLocation.enter()
        .append('g')
        .attr('class', 'currentLocation')
        .call(enter => {
          enter.append('rect')
            .attr('stroke', 'gold')
            .attr('stroke-width', 5)
            .attr('fill', 'none')
            .attr('width', worldLayout.columnWidth)
            .attr('height', worldLayout.rowHeight);
        })
        .merge(currentLocation)
        .attr('transform', d => `translate(${worldLayout.columnWidth*d[1]}, ${worldLayout.rowHeight*d[0]})`);
    }

    onSelectedTimestepChanged.push(draw);
    draw();
  })();
}

exports.locationModuleInference = locationModuleInference;
exports.locationModules = locationModules;
exports.printRecording = printRecording$2;
exports.printRecordingFromUrl = printRecordingFromUrl$2;

Object.defineProperty(exports, '__esModule', { value: true });

})));
