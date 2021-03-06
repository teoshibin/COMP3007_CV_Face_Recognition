% PLEASE REFER TO "Reproducibility" section of README FOR PARAMETERS 
% DOWNLOAD LINK AND EXECUTION INSTRUCTIONS
% repo and weights: https://github.com/teoshibin/COMP3007_CV_Face_Recognition
% This script identify 1344 test subject using 2 methods

clear all;
close all;

%% load essentials

addpath(fullfile("Models/TemplateMatching")); % load TemplateMatching
addpath(fullfile("Models/VGGFace"));          % load VGGFace
addpath(genpath("Common"));                   % load common functions

trainPath = fullfile("FaceDatabase/Train/");    % path to subject database (change this)
testPath = fullfile("FaceDatabase/Test/");      % path to subject testing set (change this)

% a made up path that may not exist yet (no need to change it unless u prefer hard coded path)
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

% all these name value pairs are optional, it is writen out as a demo
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

%% METHOD2 VGGFace with precompute function included within tic toc

% highlight lines and Ctrl + R / T to toggle commenting

% batchSize = 64;         % decrease this if your pc is getting memory error
% environment = "cpu";    % this model is too large for my tiny VRAM
% 
% load testLabel;
% 
% tic;
% 
% % all these name value pairs are optional, it is writen out as a demo
% % external precompute is not required here as precomputeCheck set to true
% outputID = VGGFace_Identify( ...
%                 trainPath, ...
%                 embeddedTrainPath, ...
%                 testPath, ...
%                 "precomputeCheck", true, ...
%                 "batchSize",batchSize, ...
%                 "similarity_metric","euclidean_l2", ...
%                 "executionEnvironment",environment ...
%            );
%        
% methodNewTime = toc
% recAccuracyNew = matchID(outputID, testLabel)

%% Compare baseline and method2

Name = ["Template Matching"; "VGGFace euclidean l2"];
Time = [runTime; methodNewTime];
Accuracy = [recAccuracy; recAccuracyNew];
tb = table(Name, Time, Accuracy)

close all force;
