function [pairIdx1,pairIdx2,pairLabel] = getSimilarPair(classLabel)

% Find all unique classes.
classes = unique(classLabel);

% Choose a class randomly which will be used to get a similar pair.
classChoice = randi(numel(classes));

% Find the indices of all the observations from the chosen class.
idxs = find(classLabel==classes(classChoice));

% Randomly choose two different images from the chosen class.
pairIdxChoice = randperm(numel(idxs),2);
pairIdx1 = idxs(pairIdxChoice(1));
pairIdx2 = idxs(pairIdxChoice(2));
pairLabel = 1;

end