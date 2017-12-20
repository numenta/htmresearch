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
% area = 'V1'; % AL or V1
% expID = '201610';
% switch area
%     case 'AL'
%         filename = strcat(expID, '/Combo3_AL.mat');
%         load(filename)
% %         load dataToSubutai_102616/Combo3_AL.mat
%     case 'V1'
%         filename = strcat(expID, '/Combo3_V1.mat');
%         load(filename)
% %         load dataToSubutai_102616/Combo3_V1.mat
% end

load('./data/Combo3_V1andAL.mat')
% analyze the 3rd data, and V1 recored spiketrain
spiketrain = Combo3_V1andAL(3, 1).spiketrain; 
imgPara = Combo3_V1andAL(3, 1).imgPara;
area = Combo3_V1andAL(3, 1).area; 
expID = Combo3_V1andAL(3, 1).date;
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

fprintf('Number of cells: %d \n', numNeuron); 
%% Population Response to Natural Stimuli
goodCells = find(spikesPerNeuron(:,stimType)>3); 
% good cells have to fire more than 3 spikes during 20 trials
stimType = 2; 
spikeMat = get_resposne_mat(spiketrain, imgPara, stimType, goodCells, 0);

% frac = 2;
% spikeMat = downsampleSpikeMat(spikeMat, frac);
% numFramesPerStim = numFramesPerStim/frac;
% imgPara.dt = imgPara.dt*frac;

%% show sparsity over trials
% sparsity figure, ppt page24 the left figure
sparsity = calculate_sparsity_over_trials(spikeMat, imgPara);
h=figure(3); clf;
subplot(2,2,1);
plot(sparsity,'k');
xlabel('trial #');
ylabel('sparseness');
title(['area ' area ' stim ' num2str(stimType)]);
print(h,'-dpdf', ['figures/sparseness_over_time_stim_' num2str(stimType) ...
    '_area_' area expID '.pdf']);


%% subset analysis, ppt page24 subset index, an array of numbers trial17 to 20
subsetIndex = calculate_subset_index(spikeMat, numFramesPerStim);

% generate Poisson spike trains
subsetIndexShuffleList = [];
subsetIndexPoissonList=[];
for rep=1:20
    % generate Poisson spike trains 
    poissSpikes = generate_poisson_spikes(spikeMat, imgPara);
    % generate shuffled spike trains
    shuffledSpikes = shuffle_spikes(spikeMat, imgPara);
    
    subsetIndexShuffle = calculate_subset_index(shuffledSpikes, numFramesPerStim);
    subsetIndexPoisson = calculate_subset_index(poissSpikes, numFramesPerStim);
    subsetIndexPoissonList(end+1) = subsetIndexPoisson;
    subsetIndexShuffleList(end+1) = subsetIndexShuffle;
end
% figure in ppt page24 right
h=figure(1);
subplot(1,3,1);
bar([subsetIndex, mean(subsetIndexShuffleList), mean(subsetIndexPoissonList)]);
ylabel('Subset index');
set(gca,'Xtick', 1:3, 'XtickLabel', {'Data', 'Shuffled', 'Poisson'});
title([area]);
print(h,'-dpdf', ['figures/subset_index_stim_' num2str(stimType) ...
    '_area_' area expID '.pdf']);

%% from here, begin to analyze the cell assembly result
numCoactive = 1; 
[frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive);
[binEdges, timeJitterDistribution1] = calculate_occurance_time(frequency, numFramesPerStim, spikeMat, binaryWords, numCoactive);

numCoactive = 3; % 3 cell assembly
[frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive);
[binEdges, timeJitterDistribution3, timeInMovieDist3] = calculate_occurance_time(...
    frequency, numFramesPerStim, spikeMat, binaryWords, numCoactive);

% generate coactive assembly analysis by shuffled spikes
fakeSpikes = shuffle_spikes(spikeMat, imgPara);
[frequencyFake, binaryWordsFake] = synchrony_analysis_efficient(fakeSpikes, numCoactive);
[~, ~, timeInMovieDistFake] = calculate_occurance_time(frequency, numFramesPerStim, fakeSpikes, binaryWords, numCoactive);

%%
figure(4);clf; 
subplot(2,2,1);
% averaged pop rate along all neurons and 20 trials
popR = sum(spikeMat, 1);
popR = mean(reshape(popR, numFramesPerStim, 20), 2)/imgPara.dt;

% averaged pop rate along fake spikes and 20 trials
popR2 = sum(fakeSpikes, 1);
popR2 = mean(reshape(popR2, numFramesPerStim, 20), 2)/imgPara.dt;

cellAssemblyFreq = timeInMovieDist3/20/imgPara.dt;
cellAssemblyFreqFake = timeInMovieDistFake/20/imgPara.dt;
plot((1:numFramesPerStim)*imgPara.dt, popR); hold on;
plot((1:numFramesPerStim)*imgPara.dt, popR2,'r--')
xlabel('time (sec)'); ylabel('Pop Rate (Hz)'); xlim([0, 30]);
legend('data','shuffled')
subplot(2,2,3);
plot((1:numFramesPerStim)*imgPara.dt, cellAssemblyFreq); hold on;
plot((1:numFramesPerStim)*imgPara.dt, cellAssemblyFreqFake,'r'); 
xlabel('time (sec)'); ylabel('Cell Assembly (Hz)'); xlim([0, 30]);
legend('data','shuffled')

subplot(2,2,4);
% plot((1:numFramesPerStim)*imgPara.dt, cellAssemblyFreq); hold on;
plot((1:numFramesPerStim)*imgPara.dt, cellAssemblyFreq-cellAssemblyFreqFake,'r'); 
xlabel('time (sec)'); ylabel('True - Shuffled (Hz)'); xlim([0, 30]);
% legend('data','shuffled')

subplot(2,2,2);
plot(popR, cellAssemblyFreq, '.'); hold on;
plot(popR, cellAssemblyFreqFake, 'r.'); legend('data','shuffled')
xlabel('Pop Rate (Hz)');  ylabel('Cell Assembly (Hz)'); 
%% plot example raster
idx = find(frequency==max(frequency));
idx = idx(1);
AssemblyTimes = find(sum(spikeMat(binaryWords(idx,:), :), 1)>=numCoactive);
AssemblyTimesFake = find(sum(fakeSpikes(binaryWords(idx,:), :), 1)>=numCoactive);
xl = [0 30];
% xl = [4.6 5.6];
smoothK = ones(4, 1)/4;
figure(10);clf;
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
print(gcf, '-dpdf', 'figures/exampleRaster.pdf');
%%
figure(1); clf; 
% subplot(221);
plot(binEdges*imgPara.dt, timeJitterDistribution3/sum(timeJitterDistribution3));hold on;
plot(binEdges*imgPara.dt, timeJitterDistribution1/sum(timeJitterDistribution1)); hold on;
legend('3-order cell assembly', 'single cell', 'location', 'northwestoutside');
xlim([-1 1])
plot([0, 0], ylim(), 'k--');
xlabel('Time jitter (sec)'); 
ylabel('Prob. of observing repeated cell assembly');
axis square;
print(gcf,'-dpdf', ['figures/cellAssembly_jitter_' area '_' expID '.pdf'])


%%


%% efficient synchrony analysis
figure(5);clf;
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
h=figure;
subplot(221);
barwitherr(yerrAll, yAll)
set(gca,'Xtick', 1:4, 'Xticklabel', 3:6);
xlabel('Cell Assembly Order');
ylabel('# unique cell assembly');
legend('Data', 'Shuffled', 'Poisson', 'location', 'northwestoutside')
print(h,'-dpdf',['figures/cellAssembly_Stim_' num2str(stimType) '_area_' area '_' expID '.pdf']);

keyboard;
%% synchrony analysis

numRpts = 1;
numSamples = 10000;
figure(5);clf;
ploti = 1;
for numCoActive = 1:4
    
    observedFreq = zeros(numRpts, 1);
    predictedFreq = zeros(numRpts, 1);
    
    maxObservedFreq = zeros(numRpts, 1);
    maxPredictedFreq = zeros(numRpts, 1);
    for rpt = 1:numRpts
        disp(['# coactive ' num2str(numCoActive) ' rpt ' num2str(rpt)]);
        [freqData, freqFakeData] = synchrony_analysis(spikeMat, fakeSpikes, numSamples, numCoActive);
        
        observedFreq(rpt) = mean(freqData)/numSamples;
        predictedFreq(rpt) = mean(freqFakeData)/numSamples;
        
        maxObservedFreq(rpt) = max(freqData)/numSamples;
        maxPredictedFreq(rpt) = max(freqFakeData)/numSamples;
        
        disp([' observed ' num2str(sum(freqData))]);
        disp([' predicted ' num2str(sum(freqFakeData))]);
    end
    
    subplot(2,2,ploti); ploti=ploti+1;
    bar([observedFreq, predictedFreq]);
%     boxplot([observedFreq, predictedFreq]);
    set(gca,'Xtick', 1:2, 'XtickLabel', {'Data', 'Predicted'});
    ylabel('Mean frequency');
    title(['#spike=' num2str(numCoActive)]);
    ylim([0, max(max([observedFreq, predictedFreq]))]);
end
% subplot(2,2,2);
% boxplot([maxObservedFreq, maxPredictedFreq]);
% set(gca,'Xtick', 1:2, 'XtickLabel', ['Data', 'Predicted']);
% ylabel('Max frequency');
%%
% figure(5); clf;
% loglog(freqData, freqFakeData,'ko'); hold on;
% xlabel('observed frequency '); ylabel('predicted frequency');
% yl = ylim(); xl = xlim();
% axisRange = [min(xl(1), yl(1)), max(xl(2), yl(2))];
% xlim(axisRange); 
% ylim(axisRange);
%%
% disp(' high order correlation analysis ');
% numWords = size(binaryWords, 1);
% numRpts = 1000;
% 
% counts = zeros(numWords, numRpts);
% countsSurrogate = zeros(numWords, numRpts);
% countsSurrogate2 = zeros(numWords, numRpts);
% 
% for rpt = 1:numRpts
%     randNeuronSample = randsample(numNeuron, 10, false);
%     counts(:, rpt) = count_spike_patterns(binaryWords, spikeMat(randNeuronSample, :));
%     countsSurrogate(:, rpt) = count_spike_patterns(binaryWords, fakeSpikes(randNeuronSample, :));
%     countsSurrogate2(:, rpt) = count_spike_patterns(binaryWords, fakeSpikes2(randNeuronSample, :));
%     %     counts = counts+count_spike_patterns(binaryWords, spikeMat(randNeuronSample, :));
%     %     countsSurrogate = countsSurrogate+count_spike_patterns(binaryWords, poissSpikes(randNeuronSample, :));
% end


%%
freqPoiss = [];
freqObserv = [];
for numSpike = 3
    idx = find(numSpikesInWord == numSpike);    
    spikePoiss = countsSurrogate(idx, :);
    spikeObs = counts(idx, :);
    freqPoiss(end+1) = mean(spikePoiss(spikePoiss>0));%/length(countsSurrogate>0);
    freqObserv(end+1) = mean(spikeObs(spikeObs>0));%/length(counts>0);
end

% 
% % semilogy(freqPoiss); hold on;
% % semilogy(freqObserv, 'r');
% %%
% % maxCount = min([max(freqPoiss); max(freqObserv)]);
% % minCount = min([min(freqPoiss); min(freqObserv)]);
% % % binrange = linspace(0, maxCount, 100);
% % binrange = logspace(log10(minCount), log10(maxCount), 50);
% %
% % figure(5);clf;
% % bincountPoiss = histc(freqPoiss, binrange);
% % bincountObserv = histc(freqObserv, binrange);
% % semilogy(binrange, bincountPoiss); hold on;
% % semilogy(binrange, bincountObserv,'r');

repeatabilityObserved = zeros(1, length(idx));
repeatabilityPoiss = zeros(1, length(idx));

for i = 1:length(idx)
    repeatabilityPoiss(i) = mean(countsSurrogate(idx(i), countsSurrogate(idx(i), :)>0));
    repeatabilityObserved(i) = mean(counts(idx(i), counts(idx(i), :)>0));
end

figure(7);clf;
xl = [min([repeatabilityPoiss repeatabilityObserved]),...
      max([repeatabilityPoiss repeatabilityObserved])];

[h,p]= ttest2(repeatabilityPoiss, repeatabilityObserved);
plot(repeatabilityObserved, repeatabilityPoiss,'ko'); hold on;
plot(xl, xl, 'k--');
xlabel(' observed repeatability ');
ylabel(' predicted repeatability ');
title([num2str(numSpike) ' -spike patterns'])
xlim(xl);
ylim(xl);

%%

freqObserved = sum(counts,2)/numRpts;
% freqObserved = sum(countsSurrogate2,2)/numRpts;
freqPoiss = sum(countsSurrogate,2)/numRpts;

colors = {'bo','ro','go','yo'};
numSpike = 4;
idx = find(numSpikesInWord < numSpike);
figure(5); clf;
loglog(freqObserved(idx), freqPoiss(idx),'ko'); hold on;
xlabel('observed frequency '); ylabel('predicted frequency');
axis([10^-3, 10^2, 10^-3, 10^2]);
loglog([0.001, 1000], [0.001, 1000]);
%%
figure(6); clf;hold on;
for numSpike = 1:3
    idx = find(numSpikesInWord == numSpike);
    figure(6);
    loglog(freqObserved(idx), freqPoiss(idx),colors{numSpike}); hold on;
    axis([10^-2, 10^1, 10^-2, 10^1]);
end
%% Example Neuron Response Over Repeated Stimulus Presentation

load dataToSubutai_102616/Combo3_V1.mat
i = 128;
stimType = 1;
spikes_total = [];
for rep = 1:imgPara.stimrep
    spikesI = spiketrain(i).st{rep, stimType};
    spikes_total = [spikes_total spikesI -1];
end

h=figure(1);clf
raster(spikes_total, [], 'k');
xlabel('Frame # ')
ylabel('Trial # ');
print(h,'-dpdf', 'figures/exampleNeuronOverRepeats.pdf');



%% % analyze pairwise correlation over trial
numNegPairs = zeros(imgPara.stimrep, 1);
meanCorr = zeros(imgPara.stimrep, 1);
medianCorr = zeros(imgPara.stimrep, 1);
for rep = 1:imgPara.stimrep
    % extract spikes timing for every trial into spikeTrialI
    spikeTrialI = spikeMat(:, (rep-1)*numFramesPerStim+(1:numFramesPerStim));        
    corrMat = nan(numNeuron, numNeuron);
    for i = 1:numNeuron
        for j = i+1:numNeuron
            % correlation analysis between neurons in each trial
            c = corrcoef(spikeTrialI(i,:), spikeTrialI(j,:)); % corrcoef has numpy library
            corrMat(i,j) = c(1,2);
            corrMat(j,i) = c(1,2);
        end
    end
    % averaged negative correlation in each trial
    numNegPairs(rep) = sum((corrMat(:)<0)) / (numNeuron*(numNeuron-1));
    % averaged correlation in each trial
    meanCorr(rep) = mean(corrMat(~isnan(corrMat)));
    % median number of correlation in each trial
    medianCorr(rep) = median(corrMat(~isnan(corrMat)));
    fprintf('trial %d sparsity %2.5f corr %2.5f numNegPair %2.5f \n', ...
        rep, sparsity(rep), meanCorr(rep), numNegPairs(rep));    
end
subplot(2,2,2);
plot(numNegPairs , 'k');
xlabel('trial #');
ylabel(' # negative corr pairs (%) ');
