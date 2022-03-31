function [sequenceA, sequenceB, labels, batchNum] = LFWPairBatchIndexer(ds, pairsTable, config)

    arguments
        ds 
        pairsTable (:,5)
        config.batchSize (1,1) double = 32
        config.shuffle (1,1) logical = true
    end

    % rectify and calculate batch size
    if config.batchSize ~= inf
        if mod(config.batchSize, 2) == 1
            config.batchSize = config.batchSize + 1;
        end
    else
        config.batchSize = length(ds.Files);
    end
    
    
%     simLen = size(similarTable, 1);
%     disLen = size(dissimilarTable, 1);
    
%     similarTable.isSimilar = ones(size(similarTable,1),"double");
%     dissimilarTable.isSimilar = zeros(size(dissimilarTable,1),"double");
    
%     wholeTable = [similarTable ; dissimilarTable];
    
    if config.shuffle
        wholeTable = pairsTable(randperm(size(pairsTable,1)), :); % shuffle rows by indexing randomly
    else
        wholeTable = pairsTable;
    end
        
    totalPairs = size(wholeTable, 1);
    batchNum =  ceil(totalPairs / config.batchSize);
    lastBatchSize = mod(totalPairs, config.batchSize);
    
    % get the name of the files
    filenames = split(ds.Files, filesep); % whole path
    filenames = split(filenames(:, end), ".");  % 'NAME_0001.jpg'
    filenames = filenames(:,1);                 % 'NAME_0001'
    
    % construct filenames from pairs and find index refering to imds
    wholeTable.idA = strcat(wholeTable.nameA, "_", arrayfun(@(x) sprintf('%04d',x),wholeTable.a,'un',0));
    wholeTable.indexA = findAll(filenames, wholeTable.idA);
    
    wholeTable.idB = strcat(wholeTable.nameB, "_", arrayfun(@(x) sprintf('%04d',x),wholeTable.b,'un',0));
    wholeTable.indexB = findAll(filenames, wholeTable.idB);
    
    sequenceA = {};
    sequenceB = {};
    labels = {};
    for i = 1:batchNum
        
        % if this is last batch then change list size
        if i == batchNum && lastBatchSize ~= 0
            binSize = lastBatchSize;
        else
            binSize = config.batchSize;
        end
        
        % generate list of index to imds per batch
        idAList = zeros([1, binSize]);
        idBList = zeros([1, binSize]);
        lbList = zeros([1, binSize]);
        for k = 1:binSize
            idAList(k) = wholeTable.indexA((i-1)*config.batchSize + k);
            idBList(k) = wholeTable.indexB((i-1)*config.batchSize + k);
            lbList(k) = wholeTable.isSimilar((i-1)*config.batchSize + k);
        end
        
        sequenceA = [sequenceA idAList];
        sequenceB = [sequenceB idBList];
        labels = [labels lbList];
        
    end
    
end

function out = findAll(A, B)

    out = zeros(size(B,1),1);
    for i = 1:size(B,1)
        out(i) = find(A == B(i), 1);
    end

end