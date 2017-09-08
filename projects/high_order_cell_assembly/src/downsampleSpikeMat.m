function spikeMatdown = downsampleSpikeMat(spikeMat, frac)
%%
if frac == 1
    spikeMatdown = spikeMat;
else
    [numNeuron, NT] = size(spikeMat);
    spikeMatdown = zeros(numNeuron, NT/frac);
    
    for i=1:size(spikeMatdown, 2);
        %     keyboard
        spikeMatdown(:, i) = sum(spikeMat(:, frac*(i-1)+(1:frac)), 2);
    end
end