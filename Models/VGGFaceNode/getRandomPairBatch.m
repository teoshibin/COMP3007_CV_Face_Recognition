
function [X1,X2,pairLabels] = getRandomPairBatch(imds,miniBatchSize)
% this function get random similar or dissimilar pairs
% the label of imageDataStore is required for each image (usually foldername)
% imds imageDataStore object
% miniBatchSize the size of 4th diamension representing a batch of images

pairLabels = zeros(1,miniBatchSize); % init output labels
imgSize = size(readimage(imds,1)); % get image size
X1 = zeros([imgSize miniBatchSize],"single"); % SSCB spatial channel batch
X2 = zeros([imgSize miniBatchSize],"single"); % SSCB

for i = 1:miniBatchSize
    choice = rand(1);

    if choice < 0.5
        [pairIdx1,pairIdx2,pairLabels(i)] = getSimilarPair(imds.Labels);
    else
        [pairIdx1,pairIdx2,pairLabels(i)] = getDissimilarPair(imds.Labels);
    end

    X1(:,:,:,i) = imds.readimage(pairIdx1);
    X2(:,:,:,i) = imds.readimage(pairIdx2);
end

end

function  [pairIdx1,pairIdx2,label] = getDissimilarPair(classLabel)

% Find all unique classes.
classes = unique(classLabel);

% Choose two different classes randomly which will be used to get a
% dissimilar pair.
classesChoice = randperm(numel(classes),2);

% Find the indices of all the observations from the first and second
% classes.
idxs1 = find(classLabel==classes(classesChoice(1)));
idxs2 = find(classLabel==classes(classesChoice(2)));

% Randomly choose one image from each class.
pairIdx1Choice = randi(numel(idxs1));
pairIdx2Choice = randi(numel(idxs2));
pairIdx1 = idxs1(pairIdx1Choice);
pairIdx2 = idxs2(pairIdx2Choice);
label = 0;

end

function [pairIdx1,pairIdx2,pairLabel] = getSimilarPair(classLabel)

% Find all classes that contain greater than 1 images.
[count,classes] = groupcounts(classLabel);
classesGT1 = classes(count > 1);


% Choose a class randomly which will be used to get a similar pair.
classChoice = randi(numel(classesGT1));
% Find the indices of all the observations from the chosen class.
idxs = find(classLabel==classesGT1(classChoice));


% Randomly choose two different images from the chosen class.
pairIdxChoice = randperm(numel(idxs),2);
pairIdx1 = idxs(pairIdxChoice(1));
pairIdx2 = idxs(pairIdxChoice(2));
pairLabel = 1;

end