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

function h=raster2(times1, numFramesPerStim, dt, color)

timeInMovie = mod(times1, numFramesPerStim);
trial = floor(times1/numFramesPerStim)+1;

for i = 1:length(times1)
    set(line,'XData',[1 1]*timeInMovie(i)*dt,...
        'YData',[trial(i) trial(i)+0.78],...
        'Color',color, 'LineWidth', .8)
end
