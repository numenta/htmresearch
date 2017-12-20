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

function [frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive)
%%
minRpts = 1;

% numSteps is the total number of frames through all 20 trials
[numNeurons, numSteps] = size(spikeMat);
popRate = sum(spikeMat, 1); % sum all the spike numbers across rows
%%
binaryWords = [];
frequency = [];
for t=1:numSteps % numSteps = 20*426 = 8520
%     disp([' step: ' num2str(t)]);
    if popRate(t) >= numCoactive % match requirement of coactive number of neurons
        activeCells = find(spikeMat(:,t)>0); % return the indices of the neurons firing at time t 
        
        % enumerate all permutations
        c = combnk(activeCells, numCoactive); % return all the possible combinations of fired cell assembly
        if numCoactive > 1
            binaryWords = [binaryWords; c];
        else
            binaryWords = [binaryWords; reshape(c, [], 1)];
        end
        
%         for i = 1:size(c, 1)
%             activeCellsI = c(i,:);
%             % check whether the current word present in the list already
%             presence = 0;
% %             if size(binaryWords, 1) >0
% %                 matches = sum(binaryWords(activeCellsI, :), 1);
% %                 if max(matches) == numCoactive
% %                     presence = 1;                    
% %                     frequency(matches==numCoactive) = frequency(matches==numCoactive)+1;
% %                 end
% %             end
%             if presence == 0
%                 % add current word to list
% %                 disp([' new word: ' num2str(activeCellsI)]);
%                 newBinaryWord = zeros(numNeurons, 1);
%                 newBinaryWord(activeCellsI) = 1;
%                 frequency = [frequency; 1];
%                 binaryWords = [binaryWords newBinaryWord];
%             end
%         end
    end
end

%% frequency returns the counts of firings of cell assembly together through 20 trials
frequency = zeros(size(binaryWords, 1),1); % # of cell assembly by 1 
for i =1:size(binaryWords)
    frequency(i) = sum(sum(spikeMat(binaryWords(i,:),:),1)>=numCoactive);
end

%% merge
idx= find(frequency==1); 
binaryWordsNew = binaryWords(idx, :);
frequencyNew = frequency(idx);

for numRpts=2:max(frequency)
    idx= find(frequency==numRpts);
    spikePatterns = binaryWords(idx,:);

    uniqueSpikes = zeros(length(idx), numCoactive);
    k=1; % the next two for loops is to remove duplicates from spikePatterns
    for i=1:size(spikePatterns, 1)% loop through the number of the cell assembly
%         disp(i);
        presence = 0;
        for j=1:k-1
            if max(abs(spikePatterns(i,:)-uniqueSpikes(j,:))) == 0
                presence=1;
                break;
            end
        end
        if presence == 0
            uniqueSpikes(k,:) = spikePatterns(i,:);
            k=k+1; 
        end
    end
    uniqueSpikes = uniqueSpikes(1:k-1,:);
    
    binaryWordsNew = [binaryWordsNew; uniqueSpikes];
    frequencyNew = [frequencyNew; ones(size(uniqueSpikes,1),1)*numRpts];
end

% returned frequency is the firing frequency for each cell assembly
% returned binaryWords is the corresponding cell indices of cell assembly
frequency = frequencyNew;
binaryWords = binaryWordsNew;
disp([' num binary words: ' num2str(size(binaryWords, 1))]);
disp([' mean frequency : ' num2str(mean(frequency))]);
disp([' max frequency : ' num2str(max(frequency))]);

idx = find(frequency>=minRpts);
frequency = frequency(idx);
binaryWords = binaryWords(idx,:);
