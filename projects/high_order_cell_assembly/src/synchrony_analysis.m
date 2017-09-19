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
