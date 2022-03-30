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