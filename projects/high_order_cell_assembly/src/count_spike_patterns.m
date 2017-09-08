function counts = count_spike_patterns(binaryWords, spikeTrain)
%%
numWords = size(binaryWords, 1);
spikesInWord = sum(binaryWords, 2);
timeStep = size(spikeTrain, 2);
counts = zeros(numWords, 1);
totalSpikeOverTime = sum(spikeTrain, 1);

goodTimeStep = find(totalSpikeOverTime > 0);
for t = goodTimeStep
    count = zeros(numWords, 1);
    count(binaryWords * spikeTrain(:, t) == spikesInWord) = 1;
    counts = counts + count;
end
