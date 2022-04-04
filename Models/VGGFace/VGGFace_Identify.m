function outputID = VGGFace_Identify(trainPath, embeddedTrainPath, testPath, config)
% Identify test faces against train faces and find the label id
% The identification is done by doing verification multiple times by batch.
% The train dataset will first be embedded using DCNN and verified by
% computing its distance value to the test dataset on runtime. Note: test
% dataset can only be embedded on runtime, but train dataset can be
% pre-embedded before runtime by calling VGGFace_ExternalPrecompute 
% exernally and set this function's 'precomputeCheck' to false.
%
%   trainPath           - path to folder containing folders of individual
%                         subjects
%
%   embeddedTrainPath   - path to target save folder of embedded subjects
%
%   testPath            - path to images, images to be identified
%
%   config              - Name & value Pairs
%
%       'precomputeCheck',true 
%           execute precompute function else skip
%           default to true
%
%       'batchSize'
%           batch size for feeding in images, larger is faster but more 
%           memory intensive. 
%           default to 32
%
%       'similarity_metric'
%           distance metric for verifying faces
%           options: {'euclidean', 'euclidean_l2', 'cosine'}
%           default to 'euclidean'
%
%       'executionEnvironment'
%           hardware selection
%           options: {'gpu', 'cpu', 'auto'}
%           default to auto
%
%       'paramsPath'
%           path to weights location, 
%           default to "Weights\VGGFace\VGGFaceParams.mat"
%           
%   outputID - array of char arrays containing id
%

    arguments
        trainPath (1,:) char
        embeddedTrainPath (1,:) char
        testPath (1,:) char
        config.precomputeCheck (1,1) logical = true
        config.batchSize (1,1) double {mustBePositive(config.batchSize),mustBeInteger(config.batchSize)} = 32
        config.similarity_metric (1,:) char = 'euclidean'
        config.executionEnvironment (1,:) char {mustBeMember(config.executionEnvironment,["auto","cpu","gpu"])} = "auto"
        config.paramsPath (1,:) char = fullfile("Weights/VGGFace/VGGFaceParams.mat")
    end
    
    % ============ SETTING UP =============
    
    % progress bar
    f = waitbar(0,"Setting up...","Name","Face Recognition",...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0); 

    % load model
    model = VGGFace_LoadModel(config.paramsPath);
        
    % target size of the network
    targetSize = model.Layers(1,1).InputSize;      

    % load dataset embeddings
    % if precomputeCheck is enabled then compute embeddings
    if config.precomputeCheck
        
        % prepare train images
        imdsTrain = imageDatastore(trainPath, ...
            "IncludeSubfolders",true, ...
            "LabelSource","foldernames");
        
        % precompute train images (refer to GeneralFunctions)
        precomputeDatabase(imdsTrain, ...
            embeddedTrainPath, ...
            model, ...
            @VGGFace_Normalize, ...
            config.batchSize, ...
            "executionEnvironment", config.executionEnvironment...
            );
        
    end
    
    % if path to embedded train path doesn't exist throw an error
    if ~exist(embeddedTrainPath, "dir")
        error("Invalid Embedded Train Path");
    end
    
    % access precomputed embeddings
    fdsTrain = fileDatastore( ...
        embeddedTrainPath, ...
        'ReadFcn',@(x) cell2mat(struct2cell(load(x))), ...
        "IncludeSubfolders",true, ...
        "FileExtensions",'.mat' ...
        );

    % load precomputed embeddings and turn cell arrays of them into 4
    % dimensional matrix
    trainEmbeddings = readall(fdsTrain);
    trainEmbeddings = cat(4, trainEmbeddings{:});
    
    % total number of training images
    trainImageNum = size(trainEmbeddings,4);
    
    % prepare char array ids
    trainPersonID = split(fdsTrain.Files, filesep);
    trainPersonID = cell2mat(trainPersonID(:,end-1));
    
    % prepare test images
    imdsTest = imageDatastore( ...
        testPath, ...
        'ReadSize', config.batchSize ...
        );
    
    % total number of testing images
    testImageNum = length(imdsTest.Files);

    
    % ========== FACE IDENTIFICATION ==========

    % predicted id
    outputID = [];
    
    % essential variables for partitioning images into batches
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
        imageBatch = imageBatchResize(imageBatch, targetSize(1:2)); % refer to GeneralFunctions
        imageBatch = single(imageBatch); % use single data type
        imageBatch = VGGFace_Normalize(imageBatch); % image preprocessing
        imageBatch = dlarray(imageBatch,"SSCB"); % convert to deep learning array
        if (config.executionEnvironment == "auto" && canUseGPU) || config.executionEnvironment == "gpu"
            imageBatch = gpuArray(imageBatch); % place it into gpu if needed
        end
        testEmbeddings = predict(model, imageBatch); % feed forward
        
        % compute similarity between database for each image within batch
        for i = 1 : currentBatchSize
            
            % duplicate the same test embeddings as same size as database
            % embeddings then verify them using 2 batches
            dupTestEmbeddings = testEmbeddings(:,:,:,i);
            dupTestEmbeddings = repmat(dupTestEmbeddings, [1 1 1 trainImageNum]);
            result = VGGFace_Verify(model, trainEmbeddings, dupTestEmbeddings, [1 1], config.similarity_metric);
            
            % find the 1st smallest distanced face index
            id = find([result.distance] == min([result.distance],[],"all"), 1);
            globalIndex = (batchIndex - 1) * config.batchSize + i;
            outputID = [outputID; trainPersonID(id,:)]; % convert index to string id and append to list
            
            % progres bar
            waitbar(globalIndex/testImageNum,f,sprintf("Identifying Faces [ %d / %d ]",globalIndex,testImageNum));
            if getappdata(f,'canceling')
                close(f);
                delete(f);
                error("Execution Cancelled");
            end
            
        end
        
        
        
    end
    
    delete(f);

end