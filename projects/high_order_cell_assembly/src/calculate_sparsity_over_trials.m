function sparsity = calculate_sparsity_over_trials(spikeMat, imgPara)

sparsity = zeros(imgPara.stimrep, 1);
numFramesPerStim = round(imgPara.stim_time / imgPara.dt);


for rep = 1:imgPara.stimrep        
    spikesCurrentTrial = spikeMat(:, (rep-1)*numFramesPerStim+(1:numFramesPerStim));    
    sparsity(rep) = mean(mean(spikesCurrentTrial));
end
    