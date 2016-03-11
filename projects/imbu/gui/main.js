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

/**
 * Main Electron code Application entry point, initializes browser app.
 */

// externals

import app from 'app';
import BrowserWindow from 'browser-window';
import crashReporter from 'crash-reporter';
import path from 'path';

// internals

let mainWindow = null; // global reference to keep window object from JS GC
const MAIN_PAGE = `file://${path.join(__dirname, '/browser/index.html')}`;


// MAIN

crashReporter.start({
  product_name: 'Imbu',
  company_name: 'Numenta'
});

app.on('window-all-closed', () => {
  // OS X apps stay active until the user quits explicitly Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Electron finished init and ready to create browser windows
app.on('ready', () => {
  mainWindow = new BrowserWindow({
    width:  1200,
    height: 720
  });
  mainWindow.loadUrl(MAIN_PAGE);
  // mainWindow.openDevTools();
  mainWindow.on('closed', () => {
    mainWindow = null; // dereference single main window object
  });
});
