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

% this function is to generate poisson spikes by using poissrnd 
function poissSpikes = generate_poisson_spikes(spikeMat, imgPara)
% disp(' generating Poisson spike trains ... ');
meanFiringRate = mean(spikeMat, 2);
[numNeuron, NT]= size(spikeMat); % NT is number of frames through 20 trials


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
