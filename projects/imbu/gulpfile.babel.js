/* -----------------------------------------------------------------------------
 * Copyright © 2015, Numenta, Inc. Unless you have purchased from
 * Numenta, Inc. a separate commercial license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero Public License version 3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for
 * more details.
 *
 * You should have received a copy of the GNU Affero Public License along with
 * this program. If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * -------------------------------------------------------------------------- */
'use strict';


/**
 * Gulp config
 */

import gulp from 'gulp';
import path from 'path';
import util from 'gulp-util';
import webpack from 'webpack';
import webpacker from 'webpack-stream';


/**
 * Gulp task to run WebPack to transpile require/modules/Babel into bundle
 */
gulp.task('webpack', () => {
  let target = util.env.target || 'web';
  let source = path.join(__dirname, 'gui/browser/app.js');
  let destination = path.join(__dirname, 'gui/browser');

  return gulp.src(source)
    .pipe(webpacker({
      bail: true,
      devtool: 'source-map',
      entry: ['babel-polyfill', source],
      module: {
        loaders: [
          // fonts
          {
            test: /\.woff(2)?$/,
            loader: 'url-loader?limit=10000&mimetype=application/font-woff'
          },
          {
            test: /\.(ttf|eot|svg)$/,
            loader: 'file-loader'
          },

          // style
          {
            test: /\.css$/,
            loaders: ['style', 'css']
          },

          // script
          {
            test: /\.(js|jsx)$/,
            loader: 'babel-loader',
            exclude: /node_modules/
          },
          {
            test: /\.json$/,
            loader: 'json'
          }
        ]
      },
      output: {
        filename: 'bundle.js',
        publicPath: destination
      },
      resolve: {
        extensions: [
          '',
          '.css',
          '.eot',
          '.js',
          '.json',
          '.jsx',
          '.svg',
          '.ttf',
          '.woff',
          '.woff2'
        ]
      },
      target,
      verbose: true
    }))
    .pipe(gulp.dest(destination));
});

// Task Compositions
gulp.task('default', []);
