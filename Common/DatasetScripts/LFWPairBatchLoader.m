function [batchA, batchB, label] = LFWPairBatchLoader(imds, sequenceA, sequenceB, labels, iter)

    imageSize = size(readimage(imds,1));
    batchSize = length(sequenceA{iter});
    
    batchA = zeros([imageSize batchSize], "single");
    batchB = zeros([imageSize batchSize], "single");
    
    for i = 1:batchSize
        batchA(:,:,:,i) = readimage(imds, sequenceA{iter}(i));
        batchB(:,:,:,i) = readimage(imds, sequenceB{iter}(i));
    end
    label = labels{iter};
    
end