function precomputeDatabase(imds, outputPath, dlModel, preprocessFunc, batchSize, skipNum, skipDirCheck, saveOverwrite, varargin)
% compute embeddings for entire image dataset and save them to a new location
% this function assumes only 1 level of subfolder exist within the database
% 
%   imds            - imageDatastore object of intput images
%                     dataset structure : rootFolder > classFolder > images
% 
%   outputPath      - output folder path
% 
%   dlModel         - deep learning model object
% 
%   preprocessFunc  - preprocess function, executed on input data before
%                     feeding into dlModel, the function must be able to
%                     process data in batch
% 
%   batchSize       - feed foward batch size (larger is faster, require more memory)
% 
%   skipNum         - to skip n numbers of images within imds
%                     if batchSize is changed, skipNum will skip according
%                     to the new batchSize, it will redo a small portion of
%                     work
%                     (this is here to save the day when error occurs after 
%                      a good portion of work is done)
% 
%   skipDirCheck    - skip directories generation (to enhance speed if
%                     folders already exist)
% 
%   varargin        - preprocess function input arguments
%
%   Examples:
%               imdsTrain = imageDatastore(datasetPath, ...
%                   "IncludeSubfolders",true, ...
%                   "LabelSource","foldernames");
%               model = loadMyModel(paramsPath);
%
%               precomputeDatabase(imdsTrain, encodedDatasetPath, model, @normalize, 32, 12680, true);
%               
%               This will start encoding images from 12672 (multiple of 32, 
%               this is automatically calculated and used instead of 12680)
%               with the batch size of 32 while ignoring folder generation 
%               at the beginning

    arguments
        imds
        outputPath
        dlModel dlnetwork
        preprocessFunc
        batchSize double {mustBePositive(batchSize)}
        skipNum double = 0
        skipDirCheck logical = false
        saveOverwrite logical = false
    end
    arguments(Repeating)
        varargin
    end

    %% prepare arguments
    argNames = ['skipImages','skipDirCheck','saveOverwrite'];
    
    skipNum = 0;
    skipDirCheck = false;
    saveOverwrite = false;
    for i = 1:2:length(varargin)
        if any(ismember(argNames, varargin{i}),"all")
            config.(varargin{i}) = vararigin{i+1};
        end
    end
        
    %% prepare output file paths and names

    % e.g. ...{'classfolder'}{'filenames.jpg'}
    %      'filenames.jpg'
    parts = split(imds.Files, filesep);
    inFilenames = parts(:, end);

    % e.g. 'filename.mat'
    outFilenames = split(inFilenames, ".");
    outFilenames(:,end) = {'mat'};
    outFilenames = join(outFilenames, ".");

    % e.g. 'classfolder'
    classfolders = parts(:, end-1);
    uniqueClassfolders = unique(classfolders);


    %% progress bar

    f = waitbar(0,sprintf("[1/2] Generating Directories\n \n "),"Name","Precompute Database",...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0);


    if ~skipDirCheck    
        %% generate directories and subdirectories

        if ~exist(outputPath,"dir")
            mkdir(outputPath);
        end
        for i = 1:length(uniqueClassfolders)

            % progress bar

            waitbar(i/length(uniqueClassfolders),f,sprintf("[ 1 / 2 ] Generating Directories [ %d / %d ]\n \n ",i,length(uniqueClassfolders)));
            if getappdata(f,'canceling')
                delete(f);
                error("Execution Cancelled");
            end


            % main code

            classPath = fullfile(outputPath, uniqueClassfolders(i));
            if ~exist(classPath,"dir")
                mkdir(classPath);
            end
        end        
    end
    
    %% generate embeddings and save it
    
    if ~isempty(varargin)
        preprocessFunc = @(x) preprocessFunc(x, varargin{:});
    end
    
    % prepare essential batch information and init payload size
    imageNum = length(imds.Files);
    batchNum = ceil(imageNum / batchSize);
    startBatchIndex = floor(skipNum / batchSize);
    if skipNum == 0
        startBatchIndex = 1;
    end
    lastBatchSize = mod(imageNum, batchSize);  
    currentBatchSize = batchSize;
    imageBatch = zeros([dlModel.Layers(1,1).InputSize batchSize], "single");
    embedingsExist = zeros([1 batchSize], "logical");
    
    % global progress bar variables
    loadedImages = startBatchIndex;
    encodedImages = startBatchIndex;
    savedImages = startBatchIndex;
    
    for batchIndex = startBatchIndex : batchNum

        % if last batch
        if batchIndex == batchNum
            imageBatch = zeros([dlModel.Layers(1,1).InputSize lastBatchSize], "single");
            embedingsExist = zeros([1 lastBatchSize], "single");
            currentBatchSize = lastBatchSize;
        end
        
        
        
        % read batch of images while making sure the image size is correct
        for binIndex = 1 : currentBatchSize
            
            % progress bar
            loadedImages = loadedImages + 1;
            updateBar(savedImages/imageNum, f, 1, loadedImages, encodedImages, savedImages, imageNum)
            
%             % check embeding exist
%             embedingsExist(binIndex) = exist(fullfile(outputPath, classfolders(globalIndex), outFilenames(globalIndex)), 'file');
%             if embedingsExist(binIndex)
%                 continue
%             end
            
            % load image
            globalIndex = (batchIndex - 1 ) * batchSize + binIndex;
            image = readimage(imds, globalIndex);
            imageBatch(:,:,:,binIndex) = single(imresize(image, dlModel.Layers(1,1).InputSize(1:2)));
        end
        
        % preprocess batch image (e.g. normalization)
        imageBatch = preprocessFunc(imageBatch);
        
        % progress bar
        updateBar(savedImages/imageNum, f, 2, loadedImages, encodedImages, savedImages, imageNum)
        
        % predict batch of images
        imageBatch = dlarray(imageBatch,"SSCB");
        embeddingBatch = predict(dlModel, imageBatch);
        
        % progress bar
        encodedImages = loadedImages;
        updateBar(savedImages/imageNum, f, 2, loadedImages, encodedImages, savedImages, imageNum);
        
        % save embeddings of images to individual matrix files
        for binIndex = 1 : currentBatchSize
            
            % progress bar
            savedImages = savedImages + 1;
            updateBar(savedImages/imageNum, f, 3, loadedImages, encodedImages, savedImages, imageNum)
            
            % code
            globalIndex = (batchIndex - 1 ) * batchSize + binIndex;
            targetEmbedding = embeddingBatch(:,:,:,binIndex);
            save(fullfile(outputPath, classfolders(globalIndex), outFilenames(globalIndex)), "targetEmbedding");
        end
        
    end
    
    delete(f);
    
end

function updateBar(progress, f, state, loadedImages, encodedImages, savedImages, imageNum)
    
    if state == 1
        s = ["*"," "," "];
    elseif state == 2
        s = [" ","*"," "];
    else
        s = [" "," ","*"];
    end
    
    waitbar(progress,f, ...
        sprintf("[ 2 / 2 ] Load Images     [ %s / %d ] %s\n" + ...
                "[ 2 / 2 ] Encode Batch    [ %s / %d ] %s\n" + ...
                "[ 2 / 2 ] Save Enbeddings [ %s / %d ] %s", ...
                pad(string(loadedImages),7,'left'),imageNum,s(1), ...
                pad(string(encodedImages),7,'left'),imageNum,s(2), ...
                pad(string(savedImages),7,'left'),imageNum,s(3)));
    
    if getappdata(f,'canceling')
        delete(f);
        error("Execution Cancelled");
    end
    
end