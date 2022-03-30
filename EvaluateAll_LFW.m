clear all;
close all;

%% load essentials

addpath(fullfile("Models","TemplateMatching")); % load TemplateMatching
addpath(fullfile("Models","VGGFace"));          % load VGGFace
addpath(genpath("Common"));                     % load common functions

% NOTE CHAR AND STRING CONCATINATION ARE DIFFERENT AND IM USING STRING HERE
% PREVIOUS IMPLEMENTATION WAS USING CHAR ARRAY

% path to folder containing subfolder of each individual
trainPath = fullfile("FaceDatabase","Train",filesep);
% path to folder containing no subfolder but a bunch of testing images
testPath = fullfile("FaceDatabase", "Test",filesep);
% path to folder containing embedded images from trainPath
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
% use less runtime memory and improve runtime speed
VGGFace_ExternalPrecompute(trainPath, embeddedDatabasePath, batchSize);
tic;

% all these name value pairs are optional, it is writen out as a demo
% additional config are all visible in function doc
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

%% METHOD2 VGGFace with euclidean similarity

batchSize = 64;
environment = "cpu";

load testLabel;
% use less runtime memory and improve runtime speed
VGGFace_ExternalPrecompute(trainPath, embeddedDatabasePath, batchSize);
tic;

% all these name value pairs are optional, it is writen out as a demo
% additional config are all visible in function doc
outputID = VGGFace_Identify( ...
                trainPath, ...
                testPath, ...
                "embeddedDatabase",true, ...
                "embeddedDatabasePath",embeddedDatabasePath, ...
                "batchSize",batchSize, ...
                "similarity_metric","euclidean", ...
                "executionEnvironment",environment ...
           );

t2 = toc
a2 = matchID(outputID, testLabel)

%% METHOD2 VGGFace with euclidean l2 similarity

batchSize = 64;
environment = "cpu";

load testLabel;
% use less runtime memory and improve runtime speed
VGGFace_ExternalPrecompute(trainPath, embeddedDatabasePath, batchSize);
tic;

% all these name value pairs are optional, it is writen out as a demo
% additional config are all visible in function doc
outputID = VGGFace_Identify( ...
                trainPath, ...
                testPath, ...
                "embeddedDatabase",true, ...
                "embeddedDatabasePath",embeddedDatabasePath, ...
                "batchSize",batchSize, ...
                "similarity_metric","euclidean_l2", ...
                "executionEnvironment",environment ...
           );

t3 = toc
a3 = matchID(outputID, testLabel)

%% Compare baseline and method2

Name = ["Template Matching"; "VGGFace Cosine"; "VGGFace Euclidean"; "VGGFace Euclidean L2"];
Time = [runTime; methodNewTime; t2; t3];
Accuracy = [recAccuracy; recAccuracyNew; a2; a3];
tb = table(Name, Time, Accuracy)
