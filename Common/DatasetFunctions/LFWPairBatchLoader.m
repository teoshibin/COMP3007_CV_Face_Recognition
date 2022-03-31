function [batchA, batchB, label] = LFWPairBatchLoader(imds, sequenceA, sequenceB, labels, iter, config)

    arguments
        imds
        sequenceA
        sequenceB
        labels
        iter
        config.imageSize double = size(readimage(imds,1))
    end

    batchSize = length(sequenceA{iter});
    
    batchA = zeros([config.imageSize batchSize], "single");
    batchB = zeros([config.imageSize batchSize], "single");
    
    for i = 1:batchSize
        batchA(:,:,:,i) = imresize(readimage(imds, sequenceA{iter}(i)), config.imageSize(1:2));
        batchB(:,:,:,i) = imresize(readimage(imds, sequenceB{iter}(i)), config.imageSize(1:2));
    end
    label = labels{iter};
    
end