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
        goodCells = (1:numNeuron)';
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
    %%
    h=figure(4);clf;
    subplot(2,2,1);
    plot(numActiveNeuron/numTotalNeuron/numFramesPerStim, '-o')
    title(['Area ' area ' Stimulus ' num2str(stimType) ' n=' num2str(numTotalNeuron)] );
    xlabel('Trial #');
    ylabel('Sparseness');
    print(h,'-dpdf', ['figures/sparsity/sparseness_over_time_stim_' num2str(stimType) '_area_' area  '.pdf']);
end