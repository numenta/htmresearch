/*----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------*/

var util = require('util');
var fs = require('fs');
var async = require('async');

var SensorTag = require('sensortag');

var fileName = 'data/sensortag_data.csv';

SensorTag.discover(function(sensorTag) {

	console.log('==> SensorTag discovered: ' + sensorTag);

	fileName = process.argv[2] || fileName;
	console.log('==> Saving data to CSV file: ' + fileName);
	fileHeader = "timestamp,x,y,z,g,raw_x,raw_y,raw_z\n";
	fs.appendFile(fileName, fileLine, function (err) {
		if (err) throw err;
		console.log('    --> CSV header: ' + fileHeader);
	    });

	sensorTag.on('disconnect', function() {
		console.log('==> SensorTag disconnected!');
		process.exit(0);

	});


	sensorTag.connectAndSetUp(function() {
	
		sensorTag.enableAccelerometer(function() {
			console.log('    --> enableAccelerometer');
			sensorTag.notifyAccelerometer(function(error){
				console.log("    --> notifyAccelerometer");
			});

			sensorTag.on('accelerometerChange', function(x_raw, y_raw, z_raw) {

				raw = [x_raw, y_raw, z_raw];

				x = raw[0]-256*(raw[0]>127);
				y = -(raw[1]-256*(raw[1]>127));
				z = raw[2]-256*(raw[2]>127);

				g = Math.sqrt((x*x)+(y*y)+(z*z));

				timestamp = new Date().getTime();

				fileLine = timestamp + ',' +  x + ',' + y + ',' + z + ',' + g + ',' + raw[0] + ',' + raw[1] + ',' + raw[2] + '\n';
				fs.appendFile(fileName, fileLine, function (err) {
					if (err) throw err;
					console.log(fileLine);
				    });
			});
		});


		sensorTag.enableGyroscope(function() {
			console.log('    --> enableGyroscope');

			sensorTag.notifyGyroscope(function(error){
				console.log("notifyGyroscope");
			});

			sensorTag.on('gyroscopeChange', function(x_raw, y_raw, z_raw) {
				raw = [x_raw, y_raw, z_raw];
				x = raw[2]+256*raw[3]-65536*(raw[3]>127);
				y = raw[0]+256*raw[1]-65536*(raw[1]>127);
				z = raw[4]+256*raw[5]-65536*(raw[5]>127)
	        });

		});

		sensorTag.enableMagnetometer(function() {
			console.log('    --> enableMagnetometer');

			sensorTag.notifyMagnetometer(function(error){
				console.log("notifyMagnetometer");
			});

			sensorTag.on('magnetometerChange', function(x_raw, y_raw, z_raw) {
				raw = [x_raw, y_raw, z_raw];

				x = -(raw[0]+256*raw[1]-65536*(raw[1]>127));
				y = -(raw[2]+256*raw[3]-65536*(raw[3]>127));
				z = raw[4]+256*raw[5]-65536*(raw[5]>127);
          	});

		});

		setTimeout(function() {
			console.log('waiting ...');
		}, 2000);

	});

});	

