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

function spikeMat = get_resposne_mat(spiketrain, imgPara, stimType, goodCells, plotRaster)
%%
numNeuron = length(goodCells);
numFramesPerStim = round(imgPara.stim_time / imgPara.dt);

spikeMat = [];
%%
for rep = 1:imgPara.stimrep    
    spikesCurrentTrial = zeros(numNeuron, numFramesPerStim);
    spikesRaster = [];
    cellI = 1;
    for i = goodCells'
        spikesI = spiketrain(i).st{rep, stimType};
        spikesI = round(spikesI(spikesI<=numFramesPerStim));
        spikesI = spikesI(spikesI>0);
                
        spikesCurrentTrial(cellI,spikesI) = 1;
        cellI  = cellI +1;
        spikesRaster = [spikesRaster spikesI*imgPara.dt -1];
    end
    
    spikeMat = [spikeMat spikesCurrentTrial];    
    if(plotRaster>0)
    % plot population response
        h=figure(2);clf;
        raster(spikesRaster, [1, 400]*imgPara.dt, 'k')
%         imagesc(spikesCurrentTrial)
        colormap gray;
        ylabel('Neuron # ');
        xlabel('Time (sec)')
        print(h,'-dpdf', ['figures/populationResponse_rpt_' num2str(rep) '.pdf']);
    end
end