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