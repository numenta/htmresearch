def get_response_mat(spiketrain, imgPara, stimType, goodCells, plotRaster):
    
    numNeuron = len(goodCells) # the number of neurons
    print numNeuron
    
    # imgPara.stim_time = 32s, imgPara.dt = 0.075103, 
    # numFramesPerStim is the number of the frames within 32s movie stimulus
    
    numFramesPerStim = int(round(imgPara['stim_time'] / imgPara['dt']))
    # print numFramesPerStim
    spikeMat = []
    ## generate the spike timing for all the neurons through all trials
    for rep in range(imgPara['stimrep']):    
        spikesCurrentTrial = np.zeros((numNeuron, numFramesPerStim))
        spikesRaster = []
        cellI = 0
        for i in range(len(goodCells)):
            # spikesI: spiking timing of a specific neuron at a specific trial
            # print i
            spikesI = spiketrain[0,i][0][rep,stimType]
            # print spikesI
            spikesI = np.round(spikesI[np.nonzero(spikesI<=numFramesPerStim)])
            #print spikesI
            spikesI = spikesI[np.nonzero(spikesI>0)];
            spikesI = spikesI.astype(int)
            
            spikesI = spikesI - 1
            # print spikesI
            # along the 426 frames, spike timings was labeled
            spikesCurrentTrial[cellI,spikesI] = 1
            cellI  = cellI +1;
            spikesRaster.append(spikesI*imgPara['dt'] -1)
        

        # return spikeMat as the spiking time for all neurons
        spikeMat.append(spikesCurrentTrial)  
    
    # change spikeMat to be numpy array
    spikeMat = np.array(spikeMat)   
    print spikeMat.shape
    return spikeMat
            
