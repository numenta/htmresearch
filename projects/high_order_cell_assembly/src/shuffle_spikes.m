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