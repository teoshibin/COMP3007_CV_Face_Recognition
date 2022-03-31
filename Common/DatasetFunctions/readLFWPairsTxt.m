function pairsTable = readLFWPairsTxt(path)

    fid = fopen(path,"r");

    % get details from the first line
    partition_details = split(fgetl(fid), char(9));
    partitions = str2double(partition_details(1));
    pairs_per_partition = str2double(partition_details(2));

    % init container
    readValues = strings([(partitions * pairs_per_partition * 2) 4]);

    % get pairs
    lcount = 1;
    for p = 1 : partitions

        % get similar pairs
        for c = 1 : pairs_per_partition
            line = split(fgetl(fid), char(9));
            readValues(lcount, 1) = line(1);
            readValues(lcount, 2) = line(2);
            readValues(lcount, 3) = line(1);
            readValues(lcount, 4) = line(3);
            lcount = lcount + 1;
        end

        % get dissimilar pairs
        for c = 1 : pairs_per_partition
            line = split(fgetl(fid), char(9));
            readValues(lcount, 1) = line(1);
            readValues(lcount, 2) = line(2);
            readValues(lcount, 3) = line(3);
            readValues(lcount, 4) = line(4);
            lcount = lcount + 1;
        end   

    end
    fclose(fid);
    
    table = array2table(readValues,"VariableNames",["nameA", "a", "nameB", "b"]);
    table.a = str2double(table.a);
    table.b = str2double(table.b);
    table.isSimilar = strcmp(table.nameA,table.nameB);

    pairsTable = table;
    
end