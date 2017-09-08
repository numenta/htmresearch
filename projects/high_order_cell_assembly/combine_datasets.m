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