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

% this script is to analyze the correlated pattern for spatial cell map
% responding to different time frames

addpath('./src/');

% load('./data/Combo3_V1andAL.mat')
% analyze the 3rd data, and V1 recored spiketrain
% spiketrain = Combo3_V1andAL(3, 1).spiketrain; 
% imgPara = Combo3_V1andAL(3, 1).imgPara;
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
        spikesPerNeuron(i, stimType) = numSpike; %kl: all the spikes in 20 trials
    end
end

fprintf('Number of cells: %d \n', numNeuron); 
%% Population Response to Natural Stimuli
% good cells have to fire more than 3 spikes during 20 trials
stimType = 1; 
% goodCells store the good cell indices
goodCells = find(spikesPerNeuron(:,stimType)>3); 

% spikeMat is selected for a specific movie stimulus
spikeMat = get_resposne_mat(spiketrain, imgPara, stimType, goodCells, 0);

% from spikeMat to calculate a cell location map for each frame in the
% movie for each trial
net = googlenet;
numFrames = size(spikeMat,2)/imgPara.stimrep;
for i = 20:20 % for 20 trials
    trialSpikeMat = spikeMat(:,(i-1)*numFrames+1:i*numFrames);
    histFeatures = zeros(numFrames, 64);%64 is the channel number from CNN
    cellMap = zeros(512*numFrames,614);
    for j=1:numFrames % for all the time frames within one stimulus
        % the cell firing at this specific time frame
        if (~isempty(find(trialSpikeMat(:,j)==1))) 
            firingCellIndices = goodCells(find(trialSpikeMat(:,j)==1));
            % create the firing map from ROI at specific time frame
            firingCellMap = sum(ROI(:,:,firingCellIndices),3);
            cellMap((j-1)*512+1:j*512,:) = firingCellMap;
            % creat broadcasted 3D map
            firingCellMap = [firingCellMap;firingCellMap;firingCellMap];
            firingCellMap = firingCellMap';
            firingCellMap = reshape(firingCellMap, [614,512,3]);
            % apply googlenet to extract features of cellmap pic 
            featuresLayer3 = activations(net, firingCellMap, 3);
            feature = sum(sum(featuresLayer3,1),2);
            feature = reshape(feature,[1,64]);
            histFeatures(j,:) = feature;
        end
    end
    
    % distance 
%     distanceMatrixHist = zeros(numFrames,numFrames);
%     for k=1:numFrames
%         for m=k:numFrames
%             if ((min(histFeatures(k,:))==0)&&(max(histFeatures(k,:))==0)&& ...
%                     (min(histFeatures(m,:))==0)&&(max(histFeatures(m,:))==0))
%                 
%                 distanceMatrixHist(k,m) = 1;
%             else
%                 distanceMatrixHist(k,m) = norm(histFeatures(k,:)-histFeatures(m,:));
%             end
%         end
%     end
    %imagesc(distanceMatrixHist);
end
cellLocation