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

function [Nreps,locs] = repeat_detect( spkraster )
%
% Usage: [Nreps,locs] = repeat_detect( spkraster )
%
% locs is a Nreps x 2 matrix with the beginning and end
% of each repeat (automatically excluding -1s)

if spkraster(end) >= 0
  spkraster(end+1) = -1;
end

% Check for empty repeats
negs = find(spkraster < 0);
if length(negs) == 1
  if length(find(diff(spkraster) < 0)) > 1
    % Then not using -1s
    negs = find(diff(spkraster) < 0);
  end
end

Nreps = length(negs);
 
locs = ones(Nreps,2);
startspk = 1;
for i = 1:Nreps
  locs(i,1) = startspk;
  if spkraster(negs(i)) < 0
    locs(i,2) = negs(i)-1;
  else
    locs(i,2) = negs(i);
  end
  startspk = negs(i)+1;
end
