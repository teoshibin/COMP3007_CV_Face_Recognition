function out = imageSubtract(batch,valueR, valueG, valueB)
% this normalize individual channels by VGGFace1 dataset mean
 
    out = zeros(size(batch),"single");
    if length(size(batch)) == 4
        
        for i = 1:size(batch,4)
            out(:,:,1,i) = batch(:,:,1,i) - valueR;
            out(:,:,2,i) = batch(:,:,2,i) - valueG;
            out(:,:,3,i) = batch(:,:,3,i) - valueB;
        end
        
    elseif length(size(batch)) == 3
        out(:,:,1) = batch(:,:,1) - valueR;
        out(:,:,2) = batch(:,:,2) - valueG;
        out(:,:,3) = batch(:,:,3) - valueB;
    end 

end