function out = distance(Y1, Y2, type)

    arguments
        Y1 (:,:,:) single
        Y2 (:,:,:) single
        type (1,:) char {mustBeMember(type,{'euclidean'})} = 'euclidean'
    end

    if strcmp(type, "euclidean")
        out = sqrt(sum((double(Y1) - double(Y2)).^2, "all"));
    end
    
end