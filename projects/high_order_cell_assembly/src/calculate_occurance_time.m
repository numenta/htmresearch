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

function [binEdges, timeJitterDistribution, timeInMovieDist, occurance_time] = calculate_occurance_time(frequency, numFramesPerStim, spikeMat, binaryWords, numCoactive)

%% check occurence of high-order patterns
idx = find(frequency >=2);
occur_times = cell(length(idx), 1);

timeInMovieSTD = zeros(length(idx), 1);

binEdges = -20:20;
timeJitterDistribution = zeros(1, length(binEdges));

trialDist = [];

interval = [];
timeAll = [];

timeInMovieDist = zeros(numFramesPerStim, 1);
occurance_time = nan(length(frequency), 1);
for i =1:length(idx)
    times = find(sum(spikeMat(binaryWords(idx(i),:),:),1)>=numCoactive);
    timeAll = [timeAll; reshape(times, [], 1)];
%     if min(diff(times))< 10
%         keyboard
%     end
    interval = [interval; reshape(diff(times), [], 1)];
    timeInMovie = mod(times, numFramesPerStim);
    timeInMovie(timeInMovie==0) = numFramesPerStim;
    timeInMovieDist(timeInMovie) = timeInMovieDist(timeInMovie)+1; 
    trial = (times-timeInMovie)/numFramesPerStim + 1;
    occurance_time(idx(i)) = mean(timeInMovie);
    counts = histc(timeInMovie-mean(timeInMovie), binEdges);
    if sum(counts)>1
        counts = counts/sum(counts);
    end
    timeJitterDistribution = timeJitterDistribution + counts;
    timeInMovieSTD(i) = std(timeInMovie);
    
    trialDist = [trialDist trial];
end

timeJitterDistribution = timeJitterDistribution/length(idx);

% %%
% figure(1); clf; 
% subplot(221);
% plot(binEdges, timeJitterDistribution)
% xlabel('Time jitter (frames)'); 
% ylabel('Prob. of observing repeated cell assembly');
% axis square;
% 

% figure(2); clf;
% edges = -0.5:1:20.5;
% trialDistHist = histc(trialDist, edges);
% plot(edges, trialDistHist);
% xlabel('frame '); ylabel('counts');


