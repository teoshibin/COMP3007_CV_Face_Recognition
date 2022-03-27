function table = readLFWPeopleTxt(path)
% This reads a text file regarding details of the subjects within LFW dataset
% This is needed to generate pairs of images for training

    fid = fopen(path,"r");
    data ={};
    folds = fgetl(fid);
    for k = 1:str2double(folds)
        instances = fgetl(fid);
        for i = 1:str2double(instances)
           data = [data; fgetl(fid)]; 
        end   
    end
    fclose(fid);
    data = split(data,char(9));
    table = cell2table(data, "VariableNames", ["name" "instance"]);
    table.instance = str2double(table.instance);
end