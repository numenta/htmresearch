import nodeResolve from 'rollup-plugin-node-resolve';
export default {
  entry: 'js/index.js',
  format: 'umd',
  moduleName: 'htmresearchviz0',
  plugins: [nodeResolve({ jsnext: true, main: true })]
};
