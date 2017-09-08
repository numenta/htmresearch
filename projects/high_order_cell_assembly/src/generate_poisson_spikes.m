function poissSpikes = generate_poisson_spikes(spikeMat, imgPara)
% disp(' generating Poisson spike trains ... ');
meanFiringRate = mean(spikeMat, 2);
[numNeuron, NT]= size(spikeMat);


poissSpikes = zeros(numNeuron, NT);
for i=1:numNeuron
    poissSpikes(i,:) = poissrnd(meanFiringRate(i), [1, NT]);
end


% firingRate = zeros(numNeuron, numFramesPerStim);
% poissSpikes = zeros(numNeuron, NT);
% for i=1:numNeuron        
%     spikeR = reshape(spikeMat(i,:), numFramesPerStim, imgPara.stimrep);
%     firingRate(i,:) = mean(spikeR, 2);
%     
%     fakeSpikes = zeros(numFramesPerStim, imgPara.stimrep);
%     for t=1:numFramesPerStim
%         fakeSpikes(t, :) =  poissrnd(firingRate(i,t), [imgPara.stimrep, 1]);
%     end
%     poissSpikes(i,:) = reshape(fakeSpikes, 1, NT);
% end
