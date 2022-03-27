function [similartable, dissimilartable] = readLFWPairsDevTxt(path)
% read pairsDevTest.txt and pairsDevTrain.txt

    fid = fopen(path,"r");

    % read first line
    setSize = str2double(fgetl(fid));

    % read remaining lines
    similarPairs = strings([setSize 1]);
    dissimilarPairs = strings([setSize 1]);
    for i = 1:setSize
        similarPairs(i,1) = fgetl(fid);    
    end
    for i = 1:setSize
        dissimilarPairs(i,1) = fgetl(fid);    
    end
    
    fclose(fid);

    similarParts = split(similarPairs, char(9));
    similartable = array2table(similarParts,"VariableNames",["nameA", "a", "b"]);
    similartable.a = str2double(similartable.a);
    similartable.b = str2double(similartable.b);
    similartable.nameB = similartable.nameA;
    similartable = [similartable(:,1:2) similartable(:,4) similartable(:,3)];

    dissimilarParts = split(dissimilarPairs, char(9));
    dissimilartable = array2table(dissimilarParts,"VariableNames",["nameA", "a", "nameB", "b"]);
    dissimilartable.a = str2double(dissimilartable.a);
    dissimilartable.b = str2double(dissimilartable.b);

end
