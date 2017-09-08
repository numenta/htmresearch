function [freqData, freqFakeData] = synchrony_analysis(spikeTrains, fakeSpikeTrains, numSamples, numCoActive)
%%
numSamples = 50000;
% numCoActive = 3;
[numNeuron, numSteps] = size(spikeTrains);
freqData = zeros(numSamples, 1);
freqFakeData = zeros(numSamples, 1);
for rpt = 1:numSamples
    randNeuronSample = randsample(numNeuron, numCoActive, false);
    freqData(rpt) = sum(sum(spikeTrains(randNeuronSample, :), 1)==numCoActive);
    freqFakeData(rpt) = sum(sum(fakeSpikeTrains(randNeuronSample, :), 1)==numCoActive);
end
