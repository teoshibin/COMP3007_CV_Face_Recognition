function outputID = FaceRecognitionNew_noNorm(trainImgSet, trainPersonID, testPath, paramsPath)

% Progress Bar
f = waitbar(0,'Seting up...','Name','Do not close this tab');
setappdata(f,'canceling',0);

% Constants
executionEnvironment = "cpu";   % this network is too large for my gpu
targetSize = [224 224 3];       % target size for the network


% --- LOAD ---

waitbar(0,f,'Preparing training images...');

% load database (train) images
subject_count = size(trainImgSet, 4);                        % number of people within the training database

% resize loaded train images to target size
trainBatch = zeros([targetSize subject_count], "single");    % init rescaled block of images
for i = 1:subject_count
    trainBatch(:,:,:,i) = imresize(trainImgSet(:,:,:,i), targetSize(1:2));
end
% trainBatch = normalizeBatch(trainBatch);                     % subtract intensity by VGGFace1 dataset mean
X1 = dlarray(trainBatch,"SSCB");                             % convert to dlarray ready for use

waitbar(0,f,'Preparing testing images...');

% load test images
imdsTest = imageDatastore(testPath);    % prepare test images (not loaded yet)

waitbar(0,f,'Loading Model...');

% load model
myModel = loadModel(paramsPath);        % load model along with pretrained model weights


% --- FACE IDENTIFICATION ---

waitbar(0,f,'Precomputing training images...');

% precompute encoding for each subject within training database
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    X1 = gpuArray(X1);
end
Y1 = embedBatch(myModel, X1);

% face identification by doing face verification using emedded subject
outputID = [];                              % output Label ID
testSlice = zeros([targetSize 1], "single");  % a face to match against entire database
Y2array = zeros(size(Y1),"single");         % non-dlarray of Y2

% do face verfication for each test subject
total_test_subject = length(imdsTest.Files);
for i = 1:total_test_subject
    
    waitbar(i / total_test_subject, f, sprintf('Identifying subject ( %d / %d )', i, total_test_subject));
    
    % load & resize image, assign it to a 4d array representing a batch
    testSlice(:,:,:,1) = imresize(single(readimage(imdsTest, i)), targetSize(1:2));
%     testSlice = normalizeBatch(testSlice);
    X2Slice = dlarray(testSlice,"SSCB");   % convert to dlarray
    
    % convert to gpu array if needed
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        X2Slice = gpuArray(X2Slice);
    end
    
    % embed the same subject for only one time
    Y2Slice = extractdata(embedBatch(myModel, X2Slice));
    
    % duplicate the embedding of this subject to match the number of people
    % within the train database
    for k = 1:subject_count
        Y2array(:,:,:,k) = Y2Slice(:,:,:,1);
    end
    Y2 = dlarray(Y2array,"SSCB");   % convert it back to dlarray
        
    % verify Y1 (database) and Y2 (whole batch of a same face)
    result = verifyBatch(myModel, Y1, Y2);
    
    id = find([result.distance] == min([result.distance],[],"all"), 1);
    outputID = [outputID; trainPersonID(id,:)];
   
end

close(f);

end