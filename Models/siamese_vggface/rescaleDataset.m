function rescaleDataset(datasetPath, outputPath)
% rescale entire dataset

% make dir if output folder doesn't exist
if ~exist(outputPath, "dir")
   mkdir(outputPath) 
end

% load image information
imds = imageDatastore(datasetPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource","none");

% get file paths
files = imds.Files;

% get folder and file paths
parts = split(files,filesep);
filenames = join(parts(:,(end-1):end),filesep);

% 
inName = fullfile(datasetPath,filenames);
outFolder = fullfile(outputPath,parts(:,end-1));
outName = fullfile(outputPath,filenames);

for i = 1:length(files)
    im = readimage(imds,i);
%     im = imread(inName(i));
    im = imresize(im, [150 150]);
    if ~exist(outFolder(i),"dir")
        mkdir(outFolder(i));
    end
    imwrite(im,outName(i));
end

end