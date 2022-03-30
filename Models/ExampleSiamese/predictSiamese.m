function Y = predictSiamese(net,fcParams,X1,X2)
% predictSiamese accepts the network and pair of images, and returns a
% prediction of the probability of the pair being similar (closer to 1) or
% dissimilar (closer to 0). Use predictSiamese during prediction.

% Pass the first image through the twin subnetwork.
Y1 = predict(net,X1);
Y1 = sigmoid(Y1);

% Pass the second image through the twin subnetwork.
Y2 = predict(net,X2);
Y2 = sigmoid(Y2);

% Subtract the feature vectors.
Y = abs(Y1 - Y2);

% Pass result through a fullyconnect operation.
Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);

% Convert to probability between 0 and 1.
Y = sigmoid(Y);

end