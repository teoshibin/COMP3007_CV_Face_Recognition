function outputID = VGGFace_Identify(trainPath, testPath, config)
% identify test face against train database, identification is done by
% running face verification multiple times.       

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
    f = waitbar(0,"Face recognition setting up...","Name","Face Recognition",...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0); 

    %% LOAD MODEL
    model = VGGFace_LoadModel(config.paramsPath);
        
    % target size for the network
    targetSize = model.Layers(1,1).InputSize;      

    %% LOAD DATABASE EMBEDDINGS

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