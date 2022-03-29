function [sequenceA, sequenceB, labels, batchNum] = LFWPairBatchIndexer(imds, similarTable, dissimilarTable, batchSize)

    % rectify and calculate batch size
    
    if mod(batchSize, 2) == 1
        batchSize = batchSize + 1;
    end
    
    simLen = size(similarTable, 1);
    disLen = size(dissimilarTable, 1);
    
    similarTable.isSimilar = ones(size(similarTable,1),"double");
    dissimilarTable.isSimilar = zeros(size(dissimilarTable,1),"double");
    
    wholeTable = [similarTable ; dissimilarTable];
    wholeTable = wholeTable(randperm(size(wholeTable,1)), :); % shuffle rows by indexing randomly
    
    totalPairs = simLen + disLen;
    batchNum =  ceil(totalPairs / batchSize);
    lastBatchSize = mod(totalPairs, batchSize);
    
    % get the name of the files
    filenames = split(imds.Files, filesep); % whole path
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
            binSize = batchSize;
        end
        
        % generate list of index to imds per batch
        idAList = zeros([1, binSize]);
        idBList = zeros([1, binSize]);
        lbList = zeros([1, binSize]);
        for k = 1:binSize
            idAList(k) = wholeTable.indexA((i-1)*batchSize + k);
            idBList(k) = wholeTable.indexB((i-1)*batchSize + k);
            lbList(k) = wholeTable.isSimilar((i-1)*batchSize + k);
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