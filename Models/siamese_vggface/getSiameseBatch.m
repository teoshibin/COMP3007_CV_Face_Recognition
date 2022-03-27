function [X1,X2,pairLabels] = getSiameseBatch(imds,miniBatchSize)

pairLabels = zeros(1,miniBatchSize); % init output labels
imgSize = size(readimage(imds,1)); % get image size
X1 = zeros([imgSize miniBatchSize],"single"); % SSCB spatial channel batch
X2 = zeros([imgSize miniBatchSize],"single"); % SSCB

for i = 1:miniBatchSize
    choice = rand(1);

    if choice < 0.5
        [pairIdx1,pairIdx2,pairLabels(i)] = getSimilarPair(imds.Labels);
    else
        [pairIdx1,pairIdx2,pairLabels(i)] = getDissimilarPair(imds.Labels);
    end

    X1(:,:,:,i) = imds.readimage(pairIdx1);
    X2(:,:,:,i) = imds.readimage(pairIdx2);
end

end