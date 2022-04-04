function out = VGGFace_Normalize(in)
% preprocess images before feeding it into the model
% normalizing it by subtracting the mean of VGGFace's 1 Million images 
% dataset mean that was used for training

    valueR = 129.1863;
    valueG = 104.7624;
    valueB = 93.5940;
    out = imageSubtract(in,valueR, valueG, valueB); % refer to GenrealFunctions
end

