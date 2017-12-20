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

% This file is the code to calculate the distances among recorded neurons
% the locations of the neurons are recorded in ROI file, with the region 
% coverd by '1's. we could calculate the center of each neuron's location
% and further use the center locations to calculate distances among neurons
% that could be stored in a matrix

% calculate the location center for each neuron

numNeuron = size(ROI,3);
xLength = size(ROI,1);
yLength = size(ROI,2);
% center locations for each neuron
centerLocation = zeros(numNeuron, 2);
centroidX = [];
centroidY = [];
for i = 1:numNeuron
    for x = 1:xLength
        for y = 1:yLength
            % calculate the centroid of the image '1's
            % which is defined to be the center location of the neuron
            if (ROI(x,y,i)==1)
                centroidX = [centroidX;x]; 
                centroidY = [centroidY;y]; 
            end
        end
    end
    xLocation = median(centroidX);
    yLocation = median(centroidY);
    centerLocation(i,1) = xLocation;
    centerLocation(i,2) = yLocation;
    centroidX = [];
    centroidY = [];
end

% calculate the euclidean distance between each pair of neurons
% neuronDist ~ neuronDistance
neuronDist = zeros(numNeuron, numNeuron);
for i = 1: numNeuron
    for j=i: numNeuron
        % spatial distance between neurons
        neuronDist(i,j) = norm(centerLocation(i)-centerLocation(j));
    end
end
 
