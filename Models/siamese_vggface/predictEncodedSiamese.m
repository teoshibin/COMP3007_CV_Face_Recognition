function Y = predictEncodedSiamese(net,fcParams,Y1,X2)
% This function takes a precomputed encoding and match it with the input
% image
% This function is used along with encodeSingle to improve speed

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