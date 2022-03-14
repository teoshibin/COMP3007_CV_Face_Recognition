function  outputID=FaceRecognition(trainImgSet, trainPersonID, testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainImgSet: contains all the given training face images
%    trainPersonID: indicate the corresponding ID of all the training images
%    testImgSet: constains all the test face images
%    outputID - predicted ID for all tested images 

%% Extract features from the training images: Here we simply use the gray-scale values as template matching. 
%  You should implement your own feature extraction/description method 

trainTmpSet=zeros(600*600,size(trainImgSet,4)); % use the raw 600x600 intensity values as feature vector 
for i=1:size(trainImgSet,4)
    tmpI= rgb2gray(trainImgSet(:,:,:,i));
    tmpI=double(tmpI(:))/255'; % normalise the intensity to 0-1& store the feature vector
    trainTmpSet(:,i)=(tmpI-mean(tmpI))/std(tmpI); 
    % Use zero-mean normalisation. This is not neccessary if you use other gradient-based feature descriptor
end

%% Face recognition for all the test images

outputID=[];
testImgNames=dir([testPath,'*.jpg']);

for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:).name]);%load one of the test images
    % perform the same pre-processing as the training images
    tmpI= rgb2gray(testImg);
    tmpI=double(tmpI(:))/255';            
    tmpI=(tmpI-mean(tmpI))/std(tmpI); 

    ccValue=trainTmpSet'*tmpI;                % perform dot product (cross correlation with all the training images, and
    labelIndx=find(ccValue==max(ccValue));    % retrieve the label that correspondes to the largest CC similarity value. 
    outputID=[outputID; trainPersonID(labelIndx(1),:)];   % store the ID for each of the test images
end

