function outputID = VGGFace_Identify(trainPath, testPath, config)
% identify test face against train database, identification is done by
% running face verification multiple times.
%
%   STADARD ARGUMENTS
%       
%   trainPath       - can be path to image database or embedding database
%                   
%   testPath        - path to a folder of testing images 
%
%   batchSize       - number of images to be executed at once
%                     this will be shared across multiple different
%                     processes
% 
%   NAME & VALUE PAIR ARGUMENTS
%
%   varargin
%       
%       'similarity' - distance metric to measure similarity between
%                             feature embeddings of images
%                             options = {'euclidean','cosine','euclidean_l2'}
%
%       'enablePrecompute'  - precompute embeddings of the database before
%                             identifying faces
%       

    arguments
        trainPath (1,:) char
        testPath (1,:) char
        config.embeddedDatabase (1,1) logical = false
        config.embeddedDatabasePostfix (1,:) char = "-embedded"
        config.embeddedDatabasePath (1,:) char = defaultEmbeddedDatabasePath()
        config.batchSize (1,1) double {mustBePositive(config.batchSize),mustBeInteger(config.batchSize)} = 32
        config.similarity_metric (1,:) char = 'euclidean'
        config.executionEnvironment (1,:) char {mustBeMember(config.executionEnvironment,["auto","cpu","gpu"])} = "auto"
        config.paramsPath (1,:) char = fullfile("Weights","VGGFace","VGGFaceParams.mat")
    end
    
    % progress bar
%     f = waitbar(0,'Seting up...','Name','Do not close this tab');

    % progress bar
    f = waitbar(0,"Face recognition setting up...","Name","Face Recognition",...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0); 

    %% LOAD MODEL
    model = VGGFace_LoadModel(config.paramsPath);
        
    % target size for the network
    targetSize = model.Layers(1,1).InputSize;      

    %% LOAD DATABASE EMBEDDINGS

    % progress bar
%     waitbar(0,f,'Preparing training images...');

    % load database (train) images
%     subject_count = size(trainImgSet, 4);                        % number of people within the training database
% 
%     % resize loaded train images to target size
%     trainBatch = zeros([targetSize subject_count], "single");    % init rescaled block of images
%     for i = 1:subject_count
%         trainBatch(:,:,:,i) = imresize(trainImgSet(:,:,:,i), targetSize(1:2));
%     end
%     trainBatch = normalizeBatch(trainBatch);                     % subtract intensity by VGGFace1 dataset mean
%     X1 = dlarray(trainBatch,"SSCB");                             % convert to dlarray ready for use

    % if train path is not embedded dataset then compute embeddings
    if ~config.embeddedDatabase
        
        imdsTrain = imageDatastore(trainPath, ...
            "IncludeSubfolders",true, ...
            "LabelSource","foldernames");
        
        %% TODO - add env config for this function
        precomputeDatabase(imdsTrain, config.embeddedDatabasePath, model, @VGGFace_Normalize, config.batchSize);
        
    end

    % access precomputed embeddings
    fdsTrain = fileDatastore( ...
        "FaceDatabase\Train-encoded", ...
        'ReadFcn',@(x) cell2mat(struct2cell(load(x))), ...
        "IncludeSubfolders",true, ...
        "FileExtensions",'.mat' ...
        );

    trainEmbeddings = readall(fdsTrain);
    trainEmbeddings = cat(4, trainEmbeddings{:});
    
    trainImageNum = size(trainEmbeddings,4);
    
    trainPersonID = split(fdsTrain.Files, filesep);
    trainPersonID = cell2mat(trainPersonID(:,end-1));
        
    %% LOAD TESTING IMAGES    

    % prepare test images
    imdsTest = imageDatastore( ...
        testPath, ...
        'ReadSize', config.batchSize ...
        );
    
    testImageNum = length(imdsTest.Files);

    %% FACE IDENTIFICATION

    outputID = [];
    
    batchNum = ceil(testImageNum / config.batchSize);
    lastBatchSize = mod(testImageNum, config.batchSize);
    currentBatchSize = config.batchSize;
    
    % progress bar
    waitbar(0,f,sprintf("Identifying Faces [ 0 / %d ]",testImageNum),...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0);   
    
    % compute dissimilarity distances in batch
    for batchIndex = 1 : batchNum
        
        % change last batch size
        if batchIndex == batchNum && lastBatchSize ~= 0
            currentBatchSize = lastBatchSize;
        end
        
        % embed test images
        imageBatch = read(imdsTest);
        imageBatch = cat(4, imageBatch{:}); % concat images through 4th dim
        imageBatch = imageBatchResize(imageBatch, targetSize(1:2));
        imageBatch = single(imageBatch);
        imageBatch = VGGFace_Normalize(imageBatch);
        imageBatch = dlarray(imageBatch,"SSCB");        
        if (config.executionEnvironment == "auto" && canUseGPU) || config.executionEnvironment == "gpu"
            imageBatch = gpuArray(imageBatch);
        end
        testEmbeddings = predict(model, imageBatch);
        
        % compute similarity between database and each image within batch
        for i = 1 : currentBatchSize
            
            dupTestEmbeddings = testEmbeddings(:,:,:,i);
            dupTestEmbeddings = repmat(dupTestEmbeddings, [1 1 1 trainImageNum]);
            result = VGGFace_Verify(model, trainEmbeddings, dupTestEmbeddings, [1 1], config.similarity_metric);
            
            % find the 1st smallest dissimilarity face index
            id = find([result.distance] == min([result.distance],[],"all"), 1);
            globalIndex = (batchIndex - 1) * config.batchSize + i;
            outputID = [outputID; trainPersonID(id,:)];
            
            % progres bar
            waitbar(globalIndex/testImageNum,f,sprintf("Identifying Faces [ %d / %d ]",globalIndex,testImageNum));
            if getappdata(f,'canceling')
                delete(f);
                error("Execution Cancelled");
            end
            
        end
        
        
        
    end
    
    close(f);
%     % precompute encoding for each subject within training database
%     if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%         X1 = gpuArray(X1);
%     end
%     Y1 = embedBatch(myModel, X1);
% 
%     % face identification by doing face verification using emedded subject
%     outputID = [];                              % output Label ID
%     testSlice = zeros([targetSize 1], "single");  % a face to match against entire database
%     Y2array = zeros(size(Y1),"single");         % non-dlarray of Y2
% 
%     % do face verfication for each test subject
%     total_test_subject = length(imdsTest.Files);
%     for i = 1:total_test_subject
% 
% %         waitbar(i / total_test_subject, f, sprintf('Identifying subject ( %d / %d )', i, total_test_subject));
% 
%         % load & resize image, assign it to a 4d array representing a batch
%         testSlice(:,:,:,1) = imresize(single(readimage(imdsTest, i)), targetSize(1:2));
%         testSlice = normalizeBatch(testSlice);
%         X2Slice = dlarray(testSlice,"SSCB");   % convert to dlarray
% 
%         % convert to gpu array if needed
%         if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%             X2Slice = gpuArray(X2Slice);
%         end
% 
%         % embed the same subject for only one time
%         Y2Slice = extractdata(embedBatch(myModel, X2Slice));
% 
%         % duplicate the embedding of this subject to match the number of people
%         % within the train database
%         for k = 1:subject_count
%             Y2array(:,:,:,k) = Y2Slice(:,:,:,1);
%         end
%         Y2 = dlarray(Y2array,"SSCB");   % convert it back to dlarray
% 
%         % verify between Y1 (database) and Y2 (whole batch of a same face)
%         result = verifyBatch(myModel, Y1, Y2, [1 1], similarity_metric);
% 
%         id = find([result.distance] == min([result.distance],[],"all"), 1);
%         outputID = [outputID; trainPersonID(id,:)];
% 
%     end
%
%     close(f);

end

function out = defaultEmbeddedDatabasePath()
    if config.embeddedDatabase
        out = trainPath;
    else
        temp = split(trainPath,filesep,2);
        temp(end) = strcat(temp(end),config.embeddedDatabasePostfix);
        out = join(temp,filesep);
    end
end