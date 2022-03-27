function out = VGGFaceNodeNormalize(batch)
% this normalize individual channels by VGGFace1 dataset mean

    shiftR = 129.1863;
    shiftG = 104.7624;
    shiftB = 93.5940;

    out = zeros(size(batch),"single");
    if length(size(batch)) == 4
        for i = 1:size(batch,4)
            out(:,:,1,i) = batch(:,:,1,i) - shiftR;
            out(:,:,2,i) = batch(:,:,2,i) - shiftG;
            out(:,:,3,i) = batch(:,:,3,i) - shiftB;
        end
    elseif length(size(batch)) == 3
        out(:,:,1) = batch(:,:,1) - shiftR;
        out(:,:,2) = batch(:,:,2) - shiftG;
        out(:,:,3) = batch(:,:,3) - shiftB;
    end   

end