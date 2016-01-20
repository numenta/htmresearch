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
addpath('./ESN_Toolbox/');
clear all;
close all;
%%%% load NYC taxi data
t = readtable('../data/nyc_taxi.csv', 'HeaderLines', 2);
t.Properties.VariableNames = {...
    'timestamp','passenger_count', 'timeofday', 'dayofweek'};
% inputSequence =  [t.timeofday t.dayofweek];
inputSequence =  [t.passenger_count t.timeofday t.dayofweek];
for i = 1:size(inputSequence, 2)
    inputSequence(:,i) = (inputSequence(:,i)-mean(inputSequence(:,i)))/std(inputSequence(:,i));
end

outputSequence = circshift(t.passenger_count, -5);
% outputSequence = (outputSequence-mean(outputSequence))/std(outputSequence);

nTrain = 5000;

train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence, testInputSequence] = ...
    split_train_test(inputSequence,train_fraction);
[trainOutputSequence,testOutputSequence] = ...
    split_train_test(outputSequence,train_fraction);
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
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 

%% create input-output plots
%%%% plot the internal states of 4 units
figure(1);
nPoints = 200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1, 'traces of first 4 reservoir units') ; 

figure(2),
nPlotPoints = length(predictedTrainOutput); 
plot_sequence(trainOutputSequence(nForgetPoints+1:end,:), predictedTrainOutput, nPlotPoints,...
    'training: teacher sequence (red) vs predicted sequence (blue)');
figure(3),
nPlotPoints = length(predictedTrainOutput); 
plot_sequence(testOutputSequence(nForgetPoints+1:end,:), predictedTestOutput, nPlotPoints, ...
    'testing: teacher sequence (red) vs predicted sequence (blue)') ; 