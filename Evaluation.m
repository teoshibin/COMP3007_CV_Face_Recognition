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

%% Template Matching (Baseline Method)

load testLabel;
tic;
outputID = templateMatching(trainImgSet, trainPersonID, testPath);
runTime = toc
recAccuracy = matchID(outputID, testLabel)

%% VGGFace with euclidean distance

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

%% Compare baseline and my best

Name = ["Template Matching"; "VGGFace Cosine"];
Time = [runTime; methodNewTime2];
Accuracy = [recAccuracy; recAccuracyNew2];
table(Name, Time, Accuracy)

%% Compare all

Name = ["Template Matching"; "VGGFace Euclidean"; "VGGFace Cosine"; "VGGFace Euclidean l2"];
Time = [runTime; methodNewTime; methodNewTime2; methodNewTime3];
Accuracy = [recAccuracy; recAccuracyNew; recAccuracyNew2; recAccuracyNew3];
table(Name, Time, Accuracy)
