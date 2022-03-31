function out = imageBatchResize(batch,targetWH)

    out = zeros([targetWH size(batch, [3 4])]);
    for i = 1 : size(batch,4)
        out(:,:,:,i) = imresize(batch(:,:,:,i), targetWH);
    end

end