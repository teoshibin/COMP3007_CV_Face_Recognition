function result = VGGFace_Verify_LFW_Pairs(datasetPath, pairDetailPath, config)

    arguments
        datasetPath (1,:) char
        pairDetailPath (1,:) char
        config.batchSize (1,1) double {mustBePositive(config.batchSize),mustBeInteger(config.batchSize)} = 32
        config.similarity_metric (1,:) char = 'euclidean'
        config.executionEnvironment (1,:) char {mustBeMember(config.executionEnvironment,["auto","cpu","gpu"])} = "auto"
        config.paramsPath (1,:) char = fullfile("Weights/VGGFace/VGGFaceParams.mat")
    end

    %% LOAD MODEL
    model = VGGFace_LoadModel(config.paramsPath);
    
    % target size for the network
    targetSize = model.Layers(1,1).InputSize;  
    
    %% LOAD TESTING IMAGES    

    % prepare test images
    imds = imageDatastore( ...
        datasetPath, ...
        "IncludeSubfolders", true, ...
        "LabelSource","foldernames" ...
        );
    
    testImageNum = length(imds.Files);

    %% FACE IDENTIFICATION
    
    LFWPairBatchLoader
    
    batchNum = ceil(testImageNum / config.batchSize);
    lastBatchSize = mod(testImageNum, config.batchSize);
    currentBatchSize = config.batchSize;
     
    
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
            
        end
        
        
        
    end
    
end