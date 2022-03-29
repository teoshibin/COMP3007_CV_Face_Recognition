function result = VGGFace_Verify(model, batch1, batch2, isEmbedded, similarity_metric)

    arguments
        model dlnetwork
        batch1 dlarray
        batch2 dlarray
        isEmbedded (1,2) logical
        similarity_metric (1,:) char
    end

    if strcmp(similarity_metric, 'euclidean')
        threshold = 0.6;
    elseif strcmp(similarity_metric, 'euclidean_l2')
        threshold = 0.86;
    elseif strcmp(similarity_metric, 'cosine')
        threshold = 0.4;
    end
    
    if ~isEmbedded(1)
        embedding1 = predict(model, batch1);
    else
        embedding1 = batch1;
    end
    if ~isEmbedded(2)
        embedding2 = predict(model, batch2);
    else
        embedding2 = batch2;
    end  
    
    result.distance = inf;
    result.similarity_metric = '';
    result.threshold = -inf;
    result.boolean = false;
    result = repmat(result, [size(batch1, 4) 1]);
    
    for i = 1:size(batch1,4)
        result(i).distance = distance( ...
            gather(extractdata(embedding1(:,:,:,i))), ...
            gather(extractdata(embedding2(:,:,:,i))), ...
            similarity_metric ...
        );
        result(i).similarity_metric = similarity_metric;
        result(i).threshold = threshold;
        result(i).boolean = result(i).distance < threshold;
    end

end