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
addpath('./src/');
load('data/DataFolderList.mat')
area = 'V1'; % area should be V1 or AL
for exp = 1%:length(DataFolderList)
    % load data
    load(strcat(DataFolderList{exp}, '/Combo3_', area, '.mat'))
    spiketrain = data.spiketrain;
    imgPara = data.imgPara;    
    date = DataFolderList{exp}(6:end);
    expID = date;
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
            spikesPerNeuron(i, stimType) = numSpike;
        end
    end
    
    fprintf('Number of cells: %d \n', numNeuron);
    %% Population Response to Natural Stimuli
    
    stimType = 2;
    goodCells = find(spikesPerNeuron(:,stimType)>3);
    spikeMat = get_resposne_mat(spiketrain, imgPara, stimType,goodCells, 0);
    
    %% show sparsity over trials
    sparsity = calculate_sparsity_over_trials(spikeMat, imgPara);
    h=figure(1); clf;
%     subplot(2,2,1);
    plot(sparsity,'k');
    xlabel('trial #');
    ylabel('sparseness');
    title(['area ' area ' stim ' num2str(stimType)]);
    print(h,'-dpng', ['figures/sparseness_over_time_stim_' num2str(stimType) ...
        '_area_' area expID '.png']);
    
    
    %% perform subset analysis
    subsetIndex = calculate_subset_index(spikeMat, numFramesPerStim);
    
    % generate Poisson spike trains
    subsetIndexShuffleList = [];
    subsetIndexPoissonList=[];
    for rep=1:20
        poissSpikes = generate_poisson_spikes(spikeMat, imgPara);
        shuffledSpikes = shuffle_spikes(spikeMat, imgPara);
        
        subsetIndexShuffle = calculate_subset_index(shuffledSpikes, numFramesPerStim);
        subsetIndexPoisson = calculate_subset_index(poissSpikes, numFramesPerStim);
        subsetIndexPoissonList(end+1) = subsetIndexPoisson;
        subsetIndexShuffleList(end+1) = subsetIndexShuffle;
    end
    %
    h=figure(2); clf;
%     subplot(1,3,1);
    bar([subsetIndex, mean(subsetIndexShuffleList), mean(subsetIndexPoissonList)]);
    ylabel('Subset index');
    set(gca,'Xtick', 1:3, 'XtickLabel', {'Data', 'Shuffled', 'Poisson'});
    title([area]);
    print(h,'-dpng', ['figures/subset_index_stim_' num2str(stimType) ...
        '_area_' area expID '.png']);
    
    %%
    numCoactive = 1;
    [frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive);
    [binEdges, timeJitterDistribution1] = calculate_occurance_time(frequency, numFramesPerStim, spikeMat, binaryWords, numCoactive);
    
    numCoactive = 3;
    [frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive);
    [binEdges, timeJitterDistribution3, timeInMovieDist3] = calculate_occurance_time(...
        frequency, numFramesPerStim, spikeMat, binaryWords, numCoactive);
    
    fakeSpikes = shuffle_spikes(spikeMat, imgPara);
    [frequencyFake, binaryWordsFake] = synchrony_analysis_efficient(fakeSpikes, numCoactive);
    [~, ~, timeInMovieDistFake] = calculate_occurance_time(frequency, numFramesPerStim, fakeSpikes, binaryWords, numCoactive);
    
   
    %% plot example raster
    idx = find(frequency==max(frequency));
    idx = idx(1);
    AssemblyTimes = find(sum(spikeMat(binaryWords(idx,:), :), 1)>=numCoactive);
    AssemblyTimesFake = find(sum(fakeSpikes(binaryWords(idx,:), :), 1)>=numCoactive);
    xl = [0 30];
    % xl = [4.6 5.6];
    smoothK = ones(4, 1)/4;
    figure(4);clf;
    subplot(4,2,1);
    colors = {'r', 'g', 'b'};
    psthSingleNeuron = zeros(numFramesPerStim, 3);
    for i = 1:3
        spikeTimes = find(sum(spikeMat(binaryWords(idx,i), :), 1)>=1);
        timeInMovie = mod(spikeTimes, numFramesPerStim);
        timeInMovie(timeInMovie==0) = numFramesPerStim;
        raster2(spikeTimes, numFramesPerStim, imgPara.dt, colors{i})
        for t=1:length(timeInMovie)
            psthSingleNeuron(timeInMovie(t), i) = psthSingleNeuron(timeInMovie(t)) + 1;
        end
        psthSingleNeuron(:, i) = conv(psthSingleNeuron(:, i), smoothK, 'same');
    end
    psthSingleNeuron = psthSingleNeuron/rep;
    
    axis([0, 30, 0, 20]);
    xlim(xl);
    title('Single Cell Spikes');
    xlabel('Time (s)'); ylabel('Trial');
    times = find(sum(spikeMat(binaryWords(idx,:), :), 1)>=numCoactive);
    
    subplot(4,2,3);
    
    for i = 1:3
        plot((1:numFramesPerStim)*imgPara.dt, psthSingleNeuron(:, i),  colors{i}); hold on;
    end
    xlim(xl); ylabel('Probability'); xlabel('Time (s)');
    
    
    subplot(4,2,2);
    colors = {'r', 'g', 'b'};
    psthSingleNeuronFake = zeros(numFramesPerStim, 3);
    for i = 1:3
        spikeTimes = find(sum(fakeSpikes(binaryWords(idx,i), :), 1)>=1);
        timeInMovie = mod(spikeTimes, numFramesPerStim);
        timeInMovie(timeInMovie==0) = numFramesPerStim;
        raster2(spikeTimes, numFramesPerStim, imgPara.dt, colors{i})
        for t=1:length(timeInMovie)
            psthSingleNeuronFake(timeInMovie(t), i) = psthSingleNeuronFake(timeInMovie(t)) + 1;
        end
        psthSingleNeuronFake(:, i) = conv(psthSingleNeuronFake(:, i), smoothK, 'same');
    end
    psthSingleNeuronFake = psthSingleNeuronFake/rep;
    
    axis([0, 30, 0, 20]);
    xlim(xl);
    title('Shuffled Spikes');
    xlabel('Time (s)'); ylabel('Trial');
    
    subplot(4,2,4);
    for i = 1:3
        plot((1:numFramesPerStim)*imgPara.dt, psthSingleNeuronFake(:, i),  colors{i}); hold on;
    end
    xlim(xl); ylabel('Probability'); xlabel('Time (s)');
    
    
    subplot(4,2,5);
    title('Cell Assembly');
    raster2(AssemblyTimes, numFramesPerStim, imgPara.dt, 'k')
    axis([0, 30, 0, 20]);
    xlabel('Time (s)'); ylabel('Trial');
    xlim(xl);
    
    
    subplot(4,2,6);
    title('Cell Assembly with shuffled spikes');
    raster2(AssemblyTimesFake, numFramesPerStim, imgPara.dt, 'k')
    axis([0, 30, 0, 20]);
    xlabel('Time (s)'); ylabel('Trial');
    xlim(xl);
    
    
    subplot(4,2,7);
    timeInMovie = mod(AssemblyTimes, numFramesPerStim);
    psthAssembly = zeros(numFramesPerStim, 1);
    for i=1:length(timeInMovie)
        psthAssembly(timeInMovie(i)) = psthAssembly(timeInMovie(i)) + 1;
    end
    psthAssembly = psthAssembly/rep;
    psthAssembly = conv(psthAssembly, smoothK, 'same');
    plot((1:numFramesPerStim)*imgPara.dt, psthAssembly, 'k');
    xlim(xl);  ylabel('Probability'); xlabel('Time (s)');
    yl=ylim();
    subplot(4,2,8);
    timeInMovie = mod(AssemblyTimesFake, numFramesPerStim);
    psthAssembly = zeros(numFramesPerStim, 1);
    for i=1:length(timeInMovie)
        psthAssembly(timeInMovie(i)) = psthAssembly(timeInMovie(i)) + 1;
    end
    psthAssembly = psthAssembly/rep;
    psthAssembly = conv(psthAssembly, smoothK, 'same');
    plot((1:numFramesPerStim)*imgPara.dt, psthAssembly, 'k');
    xlim(xl);  ylabel('Probability'); xlabel('Time (s)');
    ylim(yl);
    print(gcf, '-dpng', ['figures/exampleRaster_Stim_' num2str(stimType) '_area_' area '_' expID '.png']);
    %%
    figure(5); clf;
    % subplot(221);
    plot(binEdges*imgPara.dt, timeJitterDistribution3/sum(timeJitterDistribution3));hold on;
    plot(binEdges*imgPara.dt, timeJitterDistribution1/sum(timeJitterDistribution1)); hold on;
    legend('3-order cell assembly', 'single cell', 'location', 'northwestoutside');
    xlim([-1 1])
    plot([0, 0], ylim(), 'k--');
    xlabel('Time jitter (sec)');
    ylabel('Prob. of observing repeated cell assembly');
    axis square;
    print(gcf,'-dpng', ['figures/cellAssembly_jitter_' area '_' expID '.png'])
    
    
    %%
    
    
    %% efficient synchrony analysis
    figure(6);clf;
    rowi=1;
    coli=1;
    tSync = table();
    
    yAll = [];
    yerrAll = [];
    
    for numCoactive = 3:6
        [frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive);
        
        tShuffle = table;
        tPoiss = table;
        for rep=1:3
            fakeSpikes = shuffle_spikes(spikeMat, imgPara);
            [frequencyShuffle, ~] = synchrony_analysis_efficient(fakeSpikes, numCoactive);
            tShuffle = [tShuffle; ...
                table(length(frequencyShuffle), sum(frequencyShuffle>1), max(frequencyShuffle), ...
                'VariableNames', {'NumWords','MeanRpts', 'maxRpts'})];
            
            poissSpikes = generate_poisson_spikes(spikeMat, imgPara);
            frequencyPoiss = synchrony_analysis_efficient(poissSpikes, numCoactive);
            tPoiss = [tPoiss; ...
                table(length(frequencyPoiss), sum(frequencyPoiss>1), max(frequencyPoiss), ...
                'VariableNames', {'NumWords','MeanRpts', 'maxRpts'})];
        end
        tSync = [tSync; table(length(frequency), mean(tShuffle.NumWords), mean(tPoiss.NumWords))];
        disp(mean(frequency))
        disp(mean(frequencyShuffle))
        disp(mean(frequencyPoiss))
        %     continue
        %%
        rowi = 1;
        subplot(2,4,(rowi-1)*4+coli);
        y = [length(frequency), mean(tShuffle.NumWords), mean(tPoiss.NumWords)];
        yerr = [0, std(tShuffle.NumWords), std(tPoiss.NumWords)];
        barwitherr(yerr, y);
        ylabel(['# ' num2str(numCoactive) '-cell assemblies']);
        set(gca,'Xtick', 1:3, 'Xticklabel', {'Data', 'Shuffled', 'Poisson'});
        
        yAll = [yAll; y];
        yerrAll = [yerrAll; yerr];
        
        y = [sum(frequency>1), mean(tShuffle.MeanRpts), mean(tPoiss.MeanRpts)];
        yerr = [0, std(tShuffle.MeanRpts), std(tPoiss.MeanRpts)];
        rowi = 2;
        subplot(2,4,(rowi-1)*4+coli);
        barwitherr(yerr, y);
        ylabel(['# repeatable ' num2str(numCoactive) '-cell assemblies']);
        yl = ylim(); yl(1) = 0;
        ylim(yl);
        % ylim([.95, y(1)+.1]);
        set(gca,'Xtick', 1:3, 'Xticklabel', {'Data', 'Shuffled', 'Poisson'});
        
        coli = coli+1;
    end
    
    %%
    h=figure(7);
%     subplot(221);
    barwitherr(yerrAll, yAll)
    set(gca,'Xtick', 1:4, 'Xticklabel', 3:6);
    xlabel('Cell Assembly Order');
    ylabel('# unique cell assembly');
    legend('Data', 'Shuffled', 'Poisson', 'location', 'northwestoutside')
    print(h,'-dpng',['figures/cellAssembly_Stim_' num2str(stimType) '_area_' area '_' expID '.png']);
end