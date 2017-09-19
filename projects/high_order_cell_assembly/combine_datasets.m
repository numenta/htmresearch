%!/usr/bin/env python
% ----------------------------------------------------------------------
% Numenta Platform for Intelligent Computing (NuPIC)
% Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
% with Numenta, Inc., for a separate license for this software code, the
% following terms and conditions apply:
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero Public License version 3 as
% published by the Free Software Foundation.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% See the GNU Affero Public License for more details.
%
% You should have received a copy of the GNU Affero Public License
% along with this program.  If not, see http://www.gnu.org/licenses.
%
% http://numenta.org/licenses/
% ----------------------------------------------------------------------

clear all;
close all;
% create a list of data folders

DataFolderList = {};

DataFolderList{end+1} = 'data/2016-10-26_1';
data = load('data/2016-10-26_1/Combo3_AL.mat');
% data = data.data.aldata;
% save('data/2016-10-26_1/Combo3_AL.mat', 'data');
% data = load('data/2016-10-26_1/Combo3_V1.mat');
% data = data.data.v1data;
% save('data/2016-10-26_1/Combo3_V1.mat', 'data');

DataFolderList{end+1} = 'data/2016-07-27_2';
% data = load('data/2016-07-27_2/Combo3_AL.mat');
% data = data.data.aldata;
% save('data/2016-07-27_2/Combo3_AL.mat', 'data');
% data = load('data/2016-07-27_2/Combo3_V1.mat');
% data = data.data.v1data;
% save('data/2016-07-27_2/Combo3_V1.mat', 'data');

% this contains the most recent 5 experiments
load('/Users/ycui/Documents/SpencerData/data/Combo3_V1andAL.mat')
for i=1:size(Combo3_V1andAL, 1)
    date = strtrim(Combo3_V1andAL(i, 1).date);
    DataFolder = strcat('data/', date, '_1');
    DataFolderList{end+1} = DataFolder;
    mkdir(DataFolder);
    data = Combo3_V1andAL(i, 1);    
    save(strcat(DataFolder, '/Combo3_V1.mat'), 'data');
    data = Combo3_V1andAL(i, 2);
    save(strcat(DataFolder, '/Combo3_AL.mat'), 'data');
end



save('./data/DataFolderList.mat', 'DataFolderList');