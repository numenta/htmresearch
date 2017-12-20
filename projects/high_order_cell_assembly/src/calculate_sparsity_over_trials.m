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

function sparsity = calculate_sparsity_over_trials(spikeMat, imgPara)

sparsity = zeros(imgPara.stimrep, 1);
numFramesPerStim = round(imgPara.stim_time / imgPara.dt);

% the returned sparsity is everaged for all the neurons' firings within
% each specific trial
for rep = 1:imgPara.stimrep        
    spikesCurrentTrial = spikeMat(:, (rep-1)*numFramesPerStim+(1:numFramesPerStim));    
    sparsity(rep) = mean(mean(spikesCurrentTrial)); % average by columns and by rows
end
    