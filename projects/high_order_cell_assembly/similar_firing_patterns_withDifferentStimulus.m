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

% this script is to analyze the decorrelated pattern for spatial neuron
% firing pattern among different movie stimulus types
clear;
addpath('./src/');

% load the distance matrix of stimulus movie frames, the loaded data name
% is called "stimulusFramesDistance"
load('./dataAndResult/movie_frames_analysis/stimulusFramesDistance.mat');

% load one sample data with cell spatial locations, which provides data
% like, imgPara(parameters), ROI(cell locations), spiketrain, etc.
load('./dataAndResult/2016-06-21_1/20160727_9_Combo3_V1.mat');

%% part1: store stimulus movie frames pairs between two different stimulus types
% based on 'stimulusFramesDistance' which quatified the distance between
% each frame pairs. 

% stimulus_type = 1 (the grating stimulus), frames range: 81:400
% stimulus_type = 2 (the 1st natural stimuli), frames range: 481~800
framePairList = [];
for i=81:400
    for j=481:800
        framePairList = [framePairList; stimulusFramesDistance(i,j)];
    end
end
% plot the distribution of the distance
histfit(framePairList);

% since we are trying to compare the firing patterns corresponding to
% different stimulus types, we won't assume there will be similarity among
% cell firing patterns


%% part2: dis-similarities among spatial firing pattern corresponding to the 
% similar stimulus frame pairs.
% for each pair of stimulus frames, find out the
% corresponding pair of spatial firing pattern, and directly calculate the
% distance between firing patterns

numNeuron = length(spiketrain);
numFramesPerStim = round(imgPara.stim_time / imgPara.dt);
spikesPerNeuron = zeros(numNeuron, 3);
for i = 1:numNeuron
    for stimType = 1:imgPara.stim_type
        numSpike = 0;
        for rep = 1:imgPara.stimrep
            spikesI = spiketrain(i).st{rep, stimType};
            numSpike = numSpike + length(spikesI);
        end
        spikesPerNeuron(i, stimType) = numSpike; % all the spikes in 20 trials
    end
end

% goodCells store the good cell indices,however, I included them all here
% for simplicity
stimuType1 = 1;
goodCells1 = find(spikesPerNeuron(:,stimuType1)>=0); 
% spikeMat is selected for a specific movie stimulus
spikeMat1 = get_resposne_mat(spiketrain, imgPara, stimuType1, goodCells1, 0);

stimuType2 = 2;
goodCells2 = find(spikesPerNeuron(:,stimuType2)>=0); 
% spikeMat is selected for a specific movie stimulus
spikeMat2 = get_resposne_mat(spiketrain, imgPara, stimuType2, goodCells2, 0);


% from spikeMat to calculate a cell location map for each frame in the
% movie for each trial, stimuType = 1(gratings), =2(natural movie 1), =3(natural movie 2)
stimuA = 1;
stimuB = 2;
if stimuA == 1
    stimuAstart = 81;
    stimuAend = 400;
end
if stimuB ==2
    stimuBstart = 481;
    stimuBend = 800;
end
if stimuB ==3 
    stimuBstart = 881;
    stimuBend = 1200;
end
numFrames = 426;
res = [];
for k = 20:20 % for 20 trials, k is the trial #
    for i=stimuAstart-stimuAstart+1:stimuAend-stimuAstart+1
        for j=stimuBstart-stimuBstart+1:stimuBend-stimuBstart+1
            % store every distance pair, and find its corresponding spatial
            % firing pattern. there are 320 stimulus frames associated with
            % 426 recorded calcium imaging frames
            trialSpikeMat1 = spikeMat1(:,(k-1)*numFrames+1:k*numFrames);
            trialSpikeMat2 = spikeMat2(:,(k-1)*numFrames+1:k*numFrames);
            
            % extract the cell firing pattern for stimulus1
            if(i==1)
                firingPattern1= trialSpikeMat1(:,1)+trialSpikeMat1(:,2);
            else
                left1 = ceil(426*(i-1)/320);
                right1= floor(426*i/320);
                if(left1==right1)
                    firingPattern1 = trialSpikeMat1(:,left1);
                else
                    firingPattern1 = trialSpikeMat1(:,left1)+trialSpikeMat1(:,right1);
                end
                
            end
            
            % extract the cell firing pattern for stimulus2
            if(j==1)
                firingPattern2= trialSpikeMat2(:,1)+trialSpikeMat2(:,2);
            else
                left2 = ceil(426*(j-1)/320);
                right2= floor(426*j/320);
                if(left2==right2)
                    firingPattern2 = trialSpikeMat2(:,left2);
                else
                    firingPattern2 = trialSpikeMat2(:,left2)+trialSpikeMat2(:,right2);
                end
            end
    
            if ((~isempty(find(firingPattern1~=0))) && (~isempty(find(firingPattern2~=0)))) 
                % list the cell indices for both firing patterns
                firingCellIndices1 = goodCells1(find(firingPattern1~=0));
                firingCellIndices2 = goodCells2(find(firingPattern2~=0));
                % count distance from minimum cell number of firing patterns
                cellCount1 = size(firingCellIndices1,1);
                cellCount2 = size(firingCellIndices2,1);
                cellLocations1=zeros(cellCount1,2);
                cellLocations2=zeros(cellCount2,2);
                
                for c=1:cellCount1
                    cellLocations1(c,:) = neuronCoordinates(ROI(:,:,firingCellIndices1(c)));
                end
                for c=1:cellCount2
                    cellLocations2(c,:) = neuronCoordinates(ROI(:,:,firingCellIndices2(c)));
                end
                
                % start to find distances based on the smaller cellcount
                if (cellCount1<=cellCount2)
                    distance = 0;
                    for p=1:cellCount1
                        minDist = 10000000000;
                        for q=1:cellCount2
                            if(norm(cellLocations1(p,:)-cellLocations2(q,:))<minDist)
                                minDist = norm(cellLocations1(p,:)-cellLocations2(q,:));
                            end
                        end
                        distance = distance + minDist;
                    end
                else
                    distance = 0;
                    for p=1:cellCount2
                        minDist = 10000000000;
                        for q=1:cellCount1
                            if(norm(cellLocations2(p,:)-cellLocations1(q,:))<minDist)
                                minDist = norm(cellLocations2(p,:)-cellLocations1(q,:));
                            end
                        end
                        distance = distance + minDist;
                    end
                end
                spatialCellFiringDistance = distance;
                %[i,j], distance can form a tuple 
                res = [res;[stimulusFramesDistance(stimuAstart+i,stimuBstart+j),spatialCellFiringDistance]];
            end
        end
    end
end 

% based on the res, plot the figures by two distances
figure(1)
scatter(res(:,1),res(:,2));

