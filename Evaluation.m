clear all;
close all;

%% load essentials

% addpath(genpath("Models"));                     % load all models
addpath(fullfile("Models","TemplateMatching")); % load TemplateMatching
addpath(fullfile("Models","VGGFace"));          % load VGGFace
addpath("Common");                              % load common functions

faceDatasetPath = "FaceDatabase";
trainPath = char(fullfile(faceDatasetPath,"Train",filesep));
testPath = char(fullfile(faceDatasetPath, "Test",filesep));

%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

%% BASELINE Template Matching

load testLabel;
tic;
outputID = templateMatching(trainImgSet, trainPersonID, testPath);
runTime = toc
recAccuracy = matchID(outputID, testLabel)

%% METHOD2 VGGFace with cosine similarity

paramsPath = fullfile("Weights","VGGFace","params.mat");

load testLabel;
tic;
outputID = VGGFaceIdentify(trainImgSet, trainPersonID, testPath, paramsPath, "cosine");
methodNewTime = toc
recAccuracyNew = matchID(outputID, testLabel)

%% Compare baseline and method2

Name = ["Template Matching"; "VGGFace Cosine"];
Time = [runTime; methodNewTime];
Accuracy = [recAccuracy; recAccuracyNew];
tb = table(Name, Time, Accuracy)
