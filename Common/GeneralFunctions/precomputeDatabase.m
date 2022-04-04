function precomputeDatabase(imds, outputPath, dlModel, preprocessFunc, batchSize, config)
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
%   batchSize       - feed foward batch size (larger is faster, require more memory)
%
%   config          - Name & value Pairs
% 
%       'skipDirCheck'  
%           toggle directories generation, default to false (to
%           enhance speed if you know that folder already exist)
%
%       'saveOverwrite' 
%           toggle overwrite, default to false. false will
%           prevent re-embedding of embedded images. true will
%           embed the entire dataset regardless of anything.
%           (when this is false, previous half done job will 
%           simply be continued without starting all over again)
%
%       'executionEnvironment'
%           hardware selection, default to auto
%           options: {'gpu', 'cpu', 'auto'}
%
%   EXAMPLE:
%               imdsTrain = imageDatastore(datasetPath, ...
%                   "IncludeSubfolders",true, ...
%                   "LabelSource","foldernames");
%               model = loadMyModel(paramsPath);
%
%               precomputeDatabase(imdsTrain, encodedDatasetPath, model, @normalizeImage, 4);
%               
%               This will start embedding images that are yet to be
%               embedded with a batch size of 4. If halted due to error or 
%               cancellation, re-running it will simply continue where it 
%               left off.
%

    arguments
        imds
        outputPath
        dlModel dlnetwork
        preprocessFunc
        batchSize double {mustBePositive(batchSize)}
        config.skipDirCheck (1,1) logical = false;
        config.saveOverwrite (1,1) logical = false;
        config.executionEnvironment (1,:) char {mustBeMember(config.executionEnvironment,["auto","cpu","gpu"])} = "auto"
    end
        
    % total number of images to be embedded    
    imageNum = length(imds.Files);
    
    
    % prepare output file paths and names

    % e.g. ..., {'classfolder'}, {'filenames.jpg'}
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

    % e.g. all outputPaths
    cellOutputPaths = cell(imageNum,1);
    cellOutputPaths(:) = {char(outputPath)};
    outputPaths = join([cellOutputPaths,classfolders,outFilenames], filesep);
    

    % progress bar
    f = waitbar(0,sprintf("[1/2] Generating Directories\n \n \n "),"Name","Precompute Database",...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0);

    
    % generate directories and subdirectories
    
    if ~config.skipDirCheck
        if ~exist(outputPath,"dir")
            mkdir(outputPath);
        end
        for i = 1:length(uniqueClassfolders)

            % progress bar

            waitbar(i/length(uniqueClassfolders),f,sprintf("[ 1 / 2 ] Generating Directories [ %d / %d ]\n \n \n ",i,length(uniqueClassfolders)));
            if getappdata(f,'canceling')
                delete(f);
                error("Execution Cancelled");
            end


            % main code

            classPath = fullfile(outputPath, uniqueClassfolders{i});
            if ~exist(classPath,"dir")
                mkdir(classPath);
            end
        end        
    end
    
    % check if embedding exist
    existenceList = false(imageNum,1);
    if ~config.saveOverwrite
        for i = 1:imageNum
            existenceList(i) = exist(outputPaths{i},'file');
        end
    end
    noEmbedGlobalIndex = find(existenceList == false);
    noEmbedImageNum = length(noEmbedGlobalIndex);
    embedImageNum = imageNum - noEmbedImageNum;
    
    
    % generate embeddings and save it
    
    % prepare essential batch information and init payload size
    batchNum = ceil(noEmbedImageNum / batchSize);
    lastBatchSize = mod(noEmbedImageNum, batchSize);  
    currentBatchSize = batchSize;
    imageBatch = zeros([dlModel.Layers(1,1).InputSize batchSize], "single");
    
    % global progress bar variables
    loadedImages = 0;
    encodedImages = 0;
    savedImages = 0;
    
    
    for batchIndex = 1 : batchNum

        % if last batch and last batch size is not zero then make sure the
        % last batch is fed using the last batch size
        if batchIndex == batchNum && lastBatchSize ~= 0
            imageBatch = zeros([dlModel.Layers(1,1).InputSize lastBatchSize], "single");
            currentBatchSize = lastBatchSize;
        end
        
        % read batch of images while making sure the image size is correct
        for binIndex = 1 : currentBatchSize
            
            % progress bar
            loadedImages = loadedImages + 1;
            updateBar(savedImages/noEmbedImageNum, f, 1, embedImageNum, loadedImages, encodedImages, savedImages, noEmbedImageNum)
                        
            % load image
            % this calculates current position in noEmbedGlobalIndex then
            % index it to retrieve the actual global index pointing to the
            % file within imds
            globalIndex = noEmbedGlobalIndex((batchIndex - 1 ) * batchSize + binIndex);
            image = readimage(imds, globalIndex);
            imageBatch(:,:,:,binIndex) = single(imresize(image, dlModel.Layers(1,1).InputSize(1:2)));
        end
        
        % preprocess batch image (e.g. normalization)
        imageBatch = preprocessFunc(imageBatch);
        
        % progress bar
        updateBar(savedImages/noEmbedImageNum, f, 2, embedImageNum, loadedImages, encodedImages, savedImages, noEmbedImageNum)
        
        % predict batch of images
        imageBatch = dlarray(imageBatch,"SSCB");
        if (config.executionEnvironment == "auto" && canUseGPU) || config.executionEnvironment == "gpu"
            imageBatch = gpuArray(imageBatch);
        end
        embeddingBatch = predict(dlModel, imageBatch);
        
        % progress bar
        encodedImages = loadedImages;
        updateBar(savedImages/noEmbedImageNum, f, 2, embedImageNum, loadedImages, encodedImages, savedImages, noEmbedImageNum);
        
        % save embeddings of images to individual matrix files
        for binIndex = 1 : currentBatchSize
            
            % progress bar
            savedImages = savedImages + 1;
            updateBar(savedImages/noEmbedImageNum, f, 3, embedImageNum, loadedImages, encodedImages, savedImages, noEmbedImageNum)
            
            % code
            globalIndex = noEmbedGlobalIndex((batchIndex - 1 ) * batchSize + binIndex);
            targetEmbedding = gather(extractdata(embeddingBatch(:,:,:,binIndex)));
            save(outputPaths{globalIndex}, "targetEmbedding");
        end
        
    end
    
    delete(f); % delete progress bar
    
end

function updateBar(progress, f, state, embedImageNum, loadedImages, encodedImages, savedImages, imageNum)
    
    if state == 1
        s = ["*"," "," "];
    elseif state == 2
        s = [" ","*"," "];
    else
        s = [" "," ","*"];
    end
    
    waitbar(progress,f, ...
        sprintf("          Ignored %d Embedded Images    \n" + ...
                "[ 2 / 2 ] Load Images     [ %s / %d ] %s\n" + ...
                "[ 2 / 2 ] Encode Batch    [ %s / %d ] %s\n" + ...
                "[ 2 / 2 ] Save Enbeddings [ %s / %d ] %s", ...
                embedImageNum, ...
                pad(string(loadedImages),7,'left'),imageNum,s(1), ...
                pad(string(encodedImages),7,'left'),imageNum,s(2), ...
                pad(string(savedImages),7,'left'),imageNum,s(3)));
    
    if getappdata(f,'canceling')
        delete(f);
        error("Execution Cancelled");
    end
    
end