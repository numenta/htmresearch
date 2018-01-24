import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
import get_response_mat
import calculate_sparsity_over_trials

dataFolderList = ['data/2016-10-26_1','data/2016-07-27_2','data/2016-06-21_1','data/2016-07-07_1'
            ,'data/2016-07-27_1','data/2016-08-09_1','data/2016-08-16_1']
numActiveNeuron = 0;
numTotalNeuron = 0;

area = 'V1'; # area should be V1 or AL

for stimType in range(3):
    for exp in range(len(dataFolderList)):
    #for exp in range(1):
        data=sio.loadmat('./'+dataFolderList[exp]+'/Combo3_'+area+'.mat')
        data_c = data['data']
        # imgPara
        imgPara = data_c[0][0][1]
        print imgPara
        stim_type = imgPara['stim_type']
        stim_time = imgPara['stim_time']
        updatefr = imgPara['updatefr']
        intertime = imgPara['intertime']
        stimrep = imgPara['stimrep']
        dt = imgPara['dt']
        F = imgPara['F']
        F_gray = imgPara['F_gray']
        # spiketrain
        spiketrain = data_c[0][0][2]
        # number of neurons
        numNeuron = spiketrain.shape[1]
        numFramesPerStim = int(round(stim_time / dt))
        
        gratingResponse = [] # never be used
        spikesPerNeuron = np.zeros((numNeuron,3))
        
        for i in range(numNeuron):
            for j in range(stim_type):
                numSpike = 0
                for rep in range(stimrep):
                    spikesI = spiketrain[0,i][0][rep,j]
                    numSpike = numSpike + len(spikesI[0]);
               
                spikesPerNeuron[i, j] = numSpike;
            
        #print spikesPerNeuron
        print "Number of cells: %d" % numNeuron
        # Population Response to Natural Stimuli
         # goodCells = range(numNeuron); # choose all the cells to be goodCells
        goodCells = np.nonzero(spikesPerNeuron[:,stimType]>3)
        print goodCells[0].shape
        spikeMat = get_response_mat(spiketrain, imgPara, stimType,goodCells[0], 0);
        
        # show sparsity over trials
        sparsity = calculate_sparsity_over_trials(spikeMat, imgPara)
        numActiveNeuron = numActiveNeuron + sparsity * len(goodCells) * numFramesPerStim
        numTotalNeuron = numTotalNeuron + len(goodCells)
        """
        # spiketrain data structure example
        # the spiketrain _st of cell 78
        spiketrain_st = spiketrain[0,79][0] # shape=(20,3)
        spiketrain_st_gray = spiketrain[0,79][1] # shape=(20,3)
        # the spiketrain_st of cell 79, trial 13, stim_type = 2
        spiketrain_st_trial_stimType = spiketrain[0,79][0][12,1]
        # the specific spike timing at [0,i]
        print spiketrain_st_trial_stimType[0,1]
        """
        plt.plot(sparsity,'k')
        plt.xlabel('trial #')
        plt.ylabel('sparseness')
        plt.show()

#     
#     h=figure(4);clf;
#     subplot(2,2,1);
#     plot(numActiveNeuron/numTotalNeuron/numFramesPerStim, '-o')
#     title(['Area ' area ' Stimulus ' num2str(stimType) ' n=' num2str(numTotalNeuron)] );
#     xlabel('Trial #');
#     ylabel('Sparseness');
#     print(h,'-dpdf', ['figures/sparsity/sparseness_over_time_stim_' num2str(stimType) '_area_' area  '.pdf']);


