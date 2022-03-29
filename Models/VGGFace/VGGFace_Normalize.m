function out = VGGFace_Normalize(in)
    valueR = 129.1863;
    valueG = 104.7624;
    valueB = 93.5940;
    out = imageSubtract(in,valueR, valueG, valueB);
end

