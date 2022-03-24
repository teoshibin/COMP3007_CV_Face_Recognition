function Y = encodeSingle(net,X)
% predictSiamese accepts the network and pair of images, and returns a
% prediction of the probability of the pair being similar (closer to 1) or
% dissimilar (closer to 0). Use predictSiamese during prediction.

% Pass the first image through the twin subnetwork.
Y = predict(net,X);
Y = sigmoid(Y);

end