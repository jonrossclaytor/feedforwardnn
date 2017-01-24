%%{
% XOR DATA
% read the data from csv
M = csvread('xor.csv');
% define the inputs
inputs = M(:,1:2)';
% define the targets
targets = M(:,3)';
%%}

%{
% IRIS DATA
% read the data from csv
M = csvread('iris.csv');
% define the inputs
inputs = M(:,1:4)';
% define the targets
targets = M(:,5:7)';
%}

%{
% MNIST DATA
% load the data
load('mnistTrn.mat');
% define the inputs
inputs = trn;
% define the targets
targets = trnAns;
%}

% define the stucture of the network
nodelayers = [size(inputs,1) 3 2 size(targets,1)];

% define the hyperparameters
numEpochs = 20;
batchSize = 1;
eta = 0.1;

% execute the function
net(inputs,targets,nodelayers,numEpochs,batchSize,eta);