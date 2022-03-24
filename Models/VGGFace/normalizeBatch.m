function out = normalizeBatch(batch)
% this normalize individual channels by VGGFace1 dataset mean

out = zeros(size(batch),"single");
for i = 1:size(batch,4)
    out(:,:,1,i) = batch(:,:,1,i) - 129.1863;
    out(:,:,2,i) = batch(:,:,2,i) - 104.7624;
    out(:,:,3,i) = batch(:,:,3,i) - 93.5940;
end

end