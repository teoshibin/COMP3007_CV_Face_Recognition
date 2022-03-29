function [imgSet, personID]=loadTrainingSet(imgPath)
% imgSet stores the training images
% personID is the corresponding ID for each image. 

maxImage=200;  % set a maximum number of images for training.
folderNames=dir(imgPath);
imgSet=zeros(600,600,3,maxImage); % all images are 3 channels with size of 600x600
personID=[]; % the folder names are the labels
k=1;
for i=1:length(folderNames)
    
    % dir sometimes return these dots, added a quick fix here to avoid
    % wrong concatination
    if strcmp(folderNames(i,:).name, ".") || strcmp(folderNames(i,:).name, "..")
       continue 
    end
    
    imgName=dir(strcat(imgPath, folderNames(i,:).name, filesep,'*.jpg'));
    if ~isempty(imgName)
        imgSet(:,:,:,k)= imread(strcat(imgPath, folderNames(i,:).name, filesep, imgName.name));
        personID=[personID; folderNames(i,:).name];  %the folder names are the persons IDs. 
        k=k+1;
    end
end
imgSet=uint8(imgSet(:,:,:,1:k-1));   % Note that it is in unsigned integer 8 format with intensity range of 0 to 255.
%figure,imshow(trainImgSet(:,:,:,1)) % check the first image. 