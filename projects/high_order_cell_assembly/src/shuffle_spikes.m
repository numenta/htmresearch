function fakeSpikes = shuffle_spikes(spikeTrains, imgPara)

[numNeuron, NT] = size(spikeTrains);
fakeSpikes = zeros(numNeuron, NT);
numFramesPerStim = round(imgPara.stim_time / imgPara.dt);

for i=1:numNeuron
%     shuffleTime = randperm(NT);
%     fakeSpikes(i, :) = spikeTrains(i, shuffleTime);
    
    spikeR = reshape(spikeTrains(i,:), numFramesPerStim, imgPara.stimrep);
    shuffleSpikes = zeros(numFramesPerStim, imgPara.stimrep);
    for t=1:numFramesPerStim
        shuffleSpikes(t, :) = spikeR(t, randperm(imgPara.stimrep));
    end
    fakeSpikes(i, :) = reshape(shuffleSpikes, 1, NT);
end