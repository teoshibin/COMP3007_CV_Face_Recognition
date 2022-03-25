clear all;
close all;

%% load essentials

% addpath(genpath("Models"));                     % load all models
% addpath(fullfile("Models","TemplateMatching")); % load TemplateMatching
% addpath(fullfile("Models","VGGFace"));          % load VGGFace
addpath("Common");                              % load common functions

% functions paths to different models
% this is setup this way to avoid function name clash
funcTMatchPath = fullfile("Models","TemplateMatching");
funcVGGFacePath = fullfile("Models","VGGFace");

faceDatasetPath = "FaceDatabase";
trainPath = char(fullfile(faceDatasetPath,"Train",filesep));
testPath = char(fullfile(faceDatasetPath, "Test",filesep));

%% Retrive training and testing images

[trainImgSet, trainPersonID]=loadTrainingSet(trainPath); % load training images

%% Template Matching (Baseline Method)

addpath(funcTMatchPath); % load functions

load testLabel;
tic;
outputID = templateMatching(trainImgSet, trainPersonID, testPath);
runTime = toc
recAccuracy = matchID(outputID, testLabel)

rmpath(funcTMatchPath);

%% VGGFace with euclidean distance

addpath(funcVGGFacePath);

paramsPath = fullfile("Weights","VGGFace","params.mat");

load testLabel;
tic;
outputID = VGGFaceIdentify(trainImgSet, trainPersonID, testPath, paramsPath, "euclidean");
methodNewTime = toc
recAccuracyNew = matchID(outputID, testLabel)

%% VGGFace with cosine similarity

paramsPath = fullfile("Weights","VGGFace","params.mat");

load testLabel;
tic;
outputID = VGGFaceIdentify(trainImgSet, trainPersonID, testPath, paramsPath, "cosine");
methodNewTime2 = toc
recAccuracyNew2 = matchID(outputID, testLabel)

%% VGGFace with l2 norm euclidean distance

paramsPath = fullfile("Weights","VGGFace","params.mat");

load testLabel;
tic;
outputID = VGGFaceIdentify(trainImgSet, trainPersonID, testPath, paramsPath, "euclidean_l2");
methodNewTime3 = toc
recAccuracyNew3 = matchID(outputID, testLabel)

rmpath(funcVGGFacePath);

%% Compare all

Name = ["Template Matching"; "VGGFace Euclidean"; "VGGFace Cosine"; "VGGFace Euclidean l2"];
Time = [runTime; methodNewTime; methodNewTime2; methodNewTime3];
Accuracy = [recAccuracy; recAccuracyNew; recAccuracyNew2; recAccuracyNew3];
table(Name, Time, Accuracy)
