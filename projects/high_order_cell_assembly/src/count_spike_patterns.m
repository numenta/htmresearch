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
