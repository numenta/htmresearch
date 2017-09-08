
function [frequency, binaryWords] = synchrony_analysis_efficient(spikeMat, numCoactive)
%%
minRpts = 1;

[numNeurons, numSteps] = size(spikeMat);
popRate = sum(spikeMat, 1);
%%
binaryWords = [];
frequency = [];
for t=1:numSteps
%     disp([' step: ' num2str(t)]);
    if popRate(t) >= numCoactive
        activeCells = find(spikeMat(:,t)>0);
        
        % enumerate all permutations
        c = combnk(activeCells, numCoactive);
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

%%
frequency = zeros(size(binaryWords, 1),1);
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
    k=1;
    for i=1:size(spikePatterns, 1)
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

frequency = frequencyNew;
binaryWords = binaryWordsNew;
disp([' num binary words: ' num2str(size(binaryWords, 1))]);
disp([' mean frequency : ' num2str(mean(frequency))]);
disp([' max frequency : ' num2str(max(frequency))]);

idx = find(frequency>=minRpts);
frequency = frequency(idx);
binaryWords = binaryWords(idx,:);
