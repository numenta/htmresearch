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

% this script is to analyze the correlated pattern for spatial neuron
% firing pattern corresponding to different movie stimulus frames

addpath('./src/');

% load the distance matrix of stimulus movie frames, the loaded data name
% is called "stimulusFramesDistance"
load('./dataAndResult/movie_frames_analysis/stimulusFramesDistance.mat');

% load one sample data with cell spatial locations, which provides data
% like, imgPara(parameters), ROI(cell locations), spiketrain, etc.
load('./dataAndResult/2016-06-21_1/20160727_9_Combo3_V1.mat');

%% part1: for a specific movie stimulus, find out the similar frames pairs
% based on 'stimulusFramesDistance' which quatified the distance between
% each frame pairs. 

% stimulut_type = 2 (the 1st natural stimuli), frames range: 481~800
framePairList = [];
for i=481:799
    for j=i+1:800
        framePairList = [framePairList; stimulusFramesDistance(i,j)];
    end
end
% plot the distribution of the distance
% histfit(framePairList)?

% based on the distance distribution, assume the number <=500 indicates the
% similar stimulus movie frames, filter out these pairs, and frames time
% points have to be 20 frames difference
diffFrames = 20; % hyperparamters needed to be tuned
diffDistance = 500; % hyperparametes needed to be tuned
selectedPairs = [];
for i=481:799
    for j=i+1:800
       if (stimulusFramesDistance(i,j)<=diffDistance) && (j-i>=diffFrames)
           selectedPairs = [selectedPairs;[i,j]];
       end
    end
end

%% part2: similarities among spatial firing pattern corresponding to the 
% similar stimulus frame pairs.
% for each pair of stimulus frames (not the selectedPairs), find out the
% corresponding pair of spatial firing pattern, and directly calculate the
% distance between firing patterns

numNeuron = length(spiketrain);
numFramesPerStim = round(imgPara.stim_time / imgPara.dt);
gratingResponse = [];
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

% good cells have to fire more than 3 spikes during 20 trials
stimType = 2; 
% goodCells store the good cell indices
goodCells = find(spikesPerNeuron(:,stimType)>3); 
% spikeMat is selected for a specific movie stimulus
spikeMat = get_resposne_mat(spiketrain, imgPara, stimType, goodCells, 0);

% from spikeMat to calculate a cell location map for each frame in the
% movie for each trial
numFrames = 426;
res = [];
for k = 1:1 % for 20 trials, k is the trial #
    for i=1:319
        for j=i+1:320
            % store every distance pair, and find its corresponding spatial
            % firing pattern. there are 320 stimulus frames associated with
            % 426 recorded calcium imaging frames
            trialSpikeMat = spikeMat(:,(k-1)*numFrames+1:k*numFrames);
            if(i==1)
                firingPattern1= trialSpikeMat(:,1)+trialSpikeMat(:,2);
                
            else
                left1 = ceil(426*(i-1)/320);
                right1= floor(426*i/320);
                if(left1==right1)
                    firingPattern1 = trialSpikeMat(:,left1);
                else
                    firingPattern1 = trialSpikeMat(:,left1)+trialSpikeMat(:,right1);
                end
                
            end
            
            left2 = ceil(426*(j-1)/320);
            right2= floor(426*j/320);
            if(left2==right2)
                firingPattern2 = trialSpikeMat(:,left2);
            else
                firingPattern2 = trialSpikeMat(:,left2)+trialSpikeMat(:,right2);
            end
           
            if ((~isempty(find(firingPattern1==1))) && (~isempty(find(firingPattern2==1))) && (~isempty(find(firingPattern1==2))) && (~isempty(find(firingPattern2==2)))) 
                % list the cell indices for both firing patterns
                firingCellIndices1 = goodCells(find(firingPattern1~=0));
                firingCellIndices2 = goodCells(find(firingPattern2~=0));
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
                res = [res;[stimulusFramesDistance(480+i,480+j),spatialCellFiringDistance]];
            end
        end
    end
end 

% only for selected pairs
resSelect = [];
cellFiringPatternSelectRecord1 = [];
cellFiringPatternSelectRecord2 = [];
stimulusSelectedPairs = [];
for k = 1:1 % for 20 trials, k is the trial #
    for i=1:size(selectedPairs,1)
        
        % store every distance pair, and find its corresponding spatial
        % firing pattern. there are 320 stimulus frames associated with
        % 426 recorded calcium imaging frames
       

        trialSpikeMatSelect = spikeMat(:,(k-1)*numFrames+1:k*numFrames);
        
        firing1NumSelect = round(426*(selectedPairs(i,1)-480)/320);
        firing2NumSelect = round(426*(selectedPairs(i,2)-480)/320);
        
        firingPattern1Select = trialSpikeMat(:,firing1NumSelect);
        firingPattern2Select = trialSpikeMat(:,firing2NumSelect);

        if ((~isempty(find(firingPattern1Select==1))) && (~isempty(find(firingPattern2Select==1)))) 
            % list the cell indices for both firing patterns
            firingCellIndices1Select = goodCells(find(firingPattern1Select==1));
            firingCellIndices2Select = goodCells(find(firingPattern2Select==1));
            % count distance from minimum cell number of firing patterns
            cellCount1Select = size(firingCellIndices1Select,1);
            cellCount2Select = size(firingCellIndices2Select,1);
            cellLocations1Select=zeros(cellCount1Select,2);
            cellLocations2Select=zeros(cellCount2Select,2);
            
            % store the pairs of firing patterns at two different time frames
            cellFiringPattern1 = zeros(512,614);
            cellFiringPattern2 = zeros(512,614);
            
            % store the stimulus movie frame pairs which corresponds to the
            % fring pattern time frames
            stimulusSelectedPairs = [stimulusSelectedPairs;[selectedPairs(i,1),selectedPairs(i,2)]];

            for c=1:cellCount1Select
                cellLocations1Select(c,:) = neuronCoordinates(ROI(:,:,firingCellIndices1Select(c)));
                cellFiringPattern1 = cellFiringPattern1+ROI(:,:,firingCellIndices1Select(c));
            end
            cellFiringPatternSelectRecord1 = [cellFiringPatternSelectRecord1;cellFiringPattern1];
            for c=1:cellCount2Select
                cellLocations2Select(c,:) = neuronCoordinates(ROI(:,:,firingCellIndices2Select(c)));
                cellFiringPattern2 = cellFiringPattern2+ROI(:,:,firingCellIndices2Select(c));
            end
            cellFiringPatternSelectRecord2 = [cellFiringPatternSelectRecord2;cellFiringPattern2];
            % start to find distances based on the smaller cellcount
            if (cellCount1Select<=cellCount2Select)
                distanceSelect = 0;
                for p=1:cellCount1Select
                    minDistSelect = 10000000000;
                    for q=1:cellCount2Select
                        if(norm(cellLocations1Select(p,:)-cellLocations2Select(q,:))<minDistSelect)
                            minDistSelect = norm(cellLocations1Select(p,:)-cellLocations2Select(q,:));
                            
                        end
                    end
                    
                    distanceSelect = distanceSelect + minDistSelect;
                end
            else
                distanceSelect = 0;
                for p=1:cellCount2Select
                    minDistSelect = 10000000000;
                    for q=1:cellCount1Select
                        if(norm(cellLocations2Select(p,:)-cellLocations1Select(q,:))<minDistSelect)
                            minDistSelect = norm(cellLocations2Select(p,:)-cellLocations1Select(q,:));
                        end
                    end
                    distanceSelect = distanceSelect + minDistSelect;
                end
            end
            spatialCellFiringDistanceSelect = distanceSelect;
            %[i,j], distance can form a tuple 
            resSelect = [resSelect;[stimulusFramesDistance(selectedPairs(i,1),selectedPairs(i,2)),spatialCellFiringDistanceSelect]];
        end
    end
end 

% based on the res, plot the figures by two distances
figure(1)
scatter(res(:,1),res(:,2));

figure(2)
scatter(resSelect(:,1),resSelect(:,2));

