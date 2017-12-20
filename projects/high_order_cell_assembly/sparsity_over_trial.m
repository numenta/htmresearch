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

% This scritp summarizes the decreasing sparsity across trials for 
% all available experiments for each area
%%
clear all;
close all;
addpath('./src/');
load('./data/DataFolderList.mat')
numActiveNeuron = 0;
numTotalNeuron = 0;

area = 'V1'; % area should be V1 or AL

for stimType = 2
    for exp = 1:length(DataFolderList)
        load(strcat(DataFolderList{exp}, '/Combo3_', area, '.mat'))
        spiketrain = data.spiketrain;
        imgPara = data.imgPara;
        date = DataFolderList{exp}(6:end);
        
        numNeuron = length(spiketrain);
        numFramesPerStim = round(imgPara.stim_time / imgPara.dt);
        gratingResponse = [];
        % spikesPerNeuron records the accumulated spike counts for each
        % neuron during the 20 trials for each stimulus types
        spikesPerNeuron = zeros(numNeuron, 3);
        for i = 1:numNeuron
            for j = 1:imgPara.stim_type
                numSpike = 0;
                for rep = 1:imgPara.stimrep
                    spikesI = spiketrain(i).st{rep, j};
                    numSpike = numSpike + length(spikesI);
                end
                spikesPerNeuron(i, j) = numSpike;
            end
        end
        
        fprintf('Number of cells: %d \n', numNeuron);
        %% Population Response to Natural Stimuli
        goodCells = (1:numNeuron)'; % a vector
        %     goodCells = find(spikesPerNeuron(:,stimType)>3);
        spikeMat = get_resposne_mat(spiketrain, imgPara, stimType,goodCells, 0);
        
        %% show sparsity over trials
        sparsity = calculate_sparsity_over_trials(spikeMat, imgPara);
        numActiveNeuron = numActiveNeuron + sparsity * length(goodCells) * numFramesPerStim;
        numTotalNeuron = numTotalNeuron + length(goodCells);
        h=figure(3); clf;
        subplot(2,2,1);
        plot(sparsity,'k');
        xlabel('trial #');
        ylabel('sparseness');
        %     title(['area ' area ' stim ' num2str(stimType)]);
        print(h,'-dpdf', ['figures/sparseness_over_time_stim_' num2str(stimType) ...
            '_area_' area '_date_' date '.pdf']);
    end
    %% plot the sparsity over trials 
    h=figure(4);clf;
    subplot(2,2,1);
    plot(numActiveNeuron/numTotalNeuron/numFramesPerStim, '-o')
    title(['Area ' area ' Stimulus ' num2str(stimType) ' n=' num2str(numTotalNeuron)] );
    xlabel('Trial #');
    ylabel('Sparseness');
    print(h,'-dpdf', ['figures/sparsity/sparseness_over_time_stim_' num2str(stimType) '_area_' area  '.pdf']);
end