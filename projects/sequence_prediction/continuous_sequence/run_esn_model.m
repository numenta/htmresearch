% # ----------------------------------------------------------------------
% # Numenta Platform for Intelligent Computing (NuPIC)
% # Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
% # with Numenta, Inc., for a separate license for this software code, the
% # following terms and conditions apply:
% #
% # This program is free software: you can redistribute it and/or modify
% # it under the terms of the GNU Affero Public License version 3 as
% # published by the Free Software Foundation.
% #
% # This program is distributed in the hope that it will be useful,
% # but WITHOUT ANY WARRANTY; without even the implied warranty of
% # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% # See the GNU Affero Public License for more details.
% #
% # You should have received a copy of the GNU Affero Public License
% # along with this program.  If not, see http://www.gnu.org/licenses.
% #
% # http://numenta.org/licenses/
% # ----------------------------------------------------------------------

% This script requires ESN_Toolbox for the echo-state-network algorithm
% The toolbox is developed by Herbert Jaeger and group members
addpath('ESNToolbox/');
clear all;
close all;

injectPerturbation = 1;
nTrain = 6000;

%%%% load NYC taxi data
if injectPerturbation
    t = readtable('data/nyc_taxi_perturb.csv', 'HeaderLines', 2);
else
    t = readtable('data/nyc_taxi.csv', 'HeaderLines', 2);
end
t.Properties.VariableNames = {...
    'timestamp','passenger_count', 'timeofday', 'dayofweek'};

inputSequence =  [t.passenger_count t.timeofday t.dayofweek];
for i = 1:size(inputSequence, 2)
    inputSequence(:,i) = (inputSequence(:,i)-mean(inputSequence(:,i)))/std(inputSequence(:,i));
end
outputSequence = circshift(t.passenger_count, -5);

% sequenceLength = 600;
% inputSequence = sin((1:sequenceLength)/100*2*pi)';
% outputSequence = circshift(inputSequence, -5);

sequenceLength = length(inputSequence);
predictedData = zeros(sequenceLength, 1);
for i=nTrain+1:sequenceLength
    trainInputSequence = inputSequence(i-nTrain:i-1, :);
    trainOutputSequence = outputSequence(i-nTrain:i-1, :);
    
    %%%% generate an esn
    nInputUnits = size(inputSequence, 2);
    nInternalUnits = 100;
    nOutputUnits = 1;
    
    esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
        'spectralRadius',.1,'inputScaling',.2*ones(nInputUnits,1),'inputShift',zeros(nInputUnits,1), ...
        'teacherScaling',[0.01],'teacherShift',[0],'feedbackScaling', 0, ...
        'outputActivationFunction','identity','inverseOutputActivationFunction','identity',...
        'type', 'plain_esn');
    
    esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;
    
    %%%% train the ESN
    nForgetPoints = 200 ; % discard the first few points to stablize dynamics
    [trainedEsn, stateMatrix] = ...
        train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;
    
    
    % compute the output of the trained ESN on the training and testing data,
    % discarding the first nForgetPoints of each
    nForgetPoints = 200;
    predictedTrainOutput = test_esn(inputSequence(i-nTrain:i, :), trainedEsn, nForgetPoints);    
    
    predictedData(i) = predictedTrainOutput(end);
    i
end
%%
if injectPerturbation
    fileID = fopen('prediction/nyc_taxi_perturb_esn_pred.csv','w');
else
    fileID = fopen('prediction/nyc_taxi_esn_pred.csv','w');
end
fprintf(fileID, '%s, %s, %s\n', 'timestamp', 'passenger_count', 'prediction-5step');
fprintf(fileID, '%s, %s, %s\n', 'datetime', 'float', 'float');
fprintf(fileID, '%s, %s, %s\n', ' ', ' ', ' ');
for i = 1:length(predictedData)
    timestamp =  t.timestamp(i);
    timestamp = timestamp{1};
    fprintf(fileID, '%s, %d, %d\n',timestamp, t.passenger_count(i), predictedData(i));
end
fclose(fileID);

%%
% figure(2),
% nPlotPoints = length(predictedTrainOutput); 
% plot_sequence(trainOutputSequence(nForgetPoints+1:end,:), predictedTrainOutput, nPlotPoints,...
%     'training: teacher sequence (red) vs predicted sequence (blue)');
% figure(3),
% nPlotPoints = length(predictedTrainOutput); 
% plot_sequence(testOutputSequence(nForgetPoints+1:end,:), predictedTestOutput, nPlotPoints, ...
%     'testing: teacher sequence (red) vs predicted sequence (blue)') ; 