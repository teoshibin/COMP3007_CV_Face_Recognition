clear all;
close all;

%% load essentials

addpath(fullfile("Models","TemplateMatching")); % load TemplateMatching
addpath(fullfile("Models","VGGFace"));          % load VGGFace
addpath(genpath("Common"));                     % load common functions

trainPath = fullfile("FaceDatabase","Train",filesep);
testPath = fullfile("FaceDatabase", "Test",filesep);
embeddedDatabasePath = fullfile("FaceDatabase","Train-embedded");

%% Retrive training and testing images

[trainImgSet, trainPersonID] = loadTrainingSet([char(trainPath) filesep]); % load training images

%% BASELINE Template Matching

load testLabel;
tic;
outputID = templateMatching(trainImgSet, trainPersonID, [char(testPath) filesep]);
runTime = toc
recAccuracy = matchID(outputID, testLabel)

%% METHOD2 VGGFace with cosine similarity

batchSize = 64;
environment = "cpu";

load testLabel;
% using more memory but improve runtime speed
VGGFace_ExternalPrecompute(trainPath, embeddedDatabasePath, batchSize);
tic;

% all these name value pairs are optional, it is writen out as a demo
outputID = VGGFace_Identify( ...
                trainPath, ...
                testPath, ...
                "embeddedDatabase",true, ...
                "embeddedDatabasePath",embeddedDatabasePath, ...
                "batchSize",batchSize, ...
                "similarity_metric","cosine", ...
                "executionEnvironment",environment ...
           );

methodNewTime = toc
recAccuracyNew = matchID(outputID, testLabel)

%% Compare baseline and method2

Name = ["Template Matching"; "VGGFace Cosine"];
Time = [runTime; methodNewTime];
Accuracy = [recAccuracy; recAccuracyNew];
tb = table(Name, Time, Accuracy)
