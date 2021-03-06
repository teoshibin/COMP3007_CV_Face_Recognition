% ALL 

% This script identify 1344 test subject using 2 method 

clear all;
close all;

%% load essentials

addpath(fullfile("Models/TemplateMatching")); % load TemplateMatching
addpath(fullfile("Models/VGGFace"));          % load VGGFace
addpath(genpath("Common"));                   % load common functions

% NOTE: previous implementation was using char array here for template
% matching and trainingsetloading I have change it to char to fit the code
% consistency and quality
trainPath = fullfile("FaceDatabase/Train/");    % path to subject database (change this)
testPath = fullfile("FaceDatabase/Test/");      % path to subject testing set (change this)

% a made up path that may not exist yet
% embedded folder will be created with a postfix of '-embedded' using trainPath

% embeddedTrainPath = fullfile("FaceDatabase/Train-embedded"); % hard coded path
embeddedTrainPath = fullfile(strcat(strip(trainPath,"right",filesep), "-embedded")); % dynamicly generated path

%% Retrive training and testing images

[trainImgSet, trainPersonID] = loadTrainingSet(char(trainPath)); % load training images

%% BASELINE Template Matching

load testLabel;
tic;
outputID = templateMatching(trainImgSet, trainPersonID, char(testPath));
runTime = toc
recAccuracy = matchID(outputID, testLabel)

%% METHOD2 VGGFace with precompute function execluded from tic toc

batchSize = 64;         % decrease this if your pc is getting memory error
environment = "cpu";    % this model is too large for my tiny VRAM

load testLabel;

% use memory to improve runtime speed
VGGFace_ExternalPrecompute(trainPath, embeddedTrainPath, batchSize);

tic;

% only chnage the path if needed
outputID = VGGFace_Identify( ...
                trainPath, ...
                embeddedTrainPath, ...
                testPath, ...
                "precomputeCheck", false, ...
                "batchSize",batchSize, ...
                "similarity_metric","euclidean_l2", ...
                "executionEnvironment",environment ...
           );
       
methodNewTime = toc
recAccuracyNew = matchID(outputID, testLabel)

tic;

% all these name value pairs are optional, it is writen out as a demo
outputID = VGGFace_Identify( ...
                trainPath, ...
                embeddedTrainPath, ...
                testPath, ...
                "precomputeCheck", false, ...
                "batchSize",batchSize, ...
                "similarity_metric","euclidean", ...
                "executionEnvironment",environment ...
           );
       
methodNewTime2 = toc
recAccuracyNew2 = matchID(outputID, testLabel)

tic;

% all these name value pairs are optional, it is writen out as a demo
outputID = VGGFace_Identify( ...
                trainPath, ...
                embeddedTrainPath, ...
                testPath, ...
                "precomputeCheck", false, ...
                "batchSize",batchSize, ...
                "similarity_metric","cosine", ...
                "executionEnvironment",environment ...
           );
       
methodNewTime3 = toc
recAccuracyNew3 = matchID(outputID, testLabel)

%% Compare baseline and method2

Name = ["Template Matching"; "VGGFace euclidean l2"; "VGGFace euclidean"; "VGGFace cosine"];
Time = [runTime; methodNewTime, methodNewTime2, methodNewTime3];
Accuracy = [recAccuracy; recAccuracyNew, recAccuracyNew2, recAccuracyNew3];
tb = table(Name, Time, Accuracy)
