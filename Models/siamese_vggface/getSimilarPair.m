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