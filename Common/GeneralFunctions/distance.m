function Y = distance(X1, X2, type)

    arguments
        X1 single
        X2 single
        type (1,:) char {mustBeMember(type,{'euclidean','cosine', 'euclidean_l2'})}
    end
    
    if strcmp(type, "euclidean")
        Y = euclideanDistance(X1, X2);
    elseif strcmp(type, "euclidean_l2")
        Y = euclideanDistance(l2_norm(X1), l2_norm(X2));
    elseif strcmp(type, "cosine")
        Y = 1 - (sum(X1.* X2, [1 2 3]) / (sqrt(sum(X1.^2, [1 2 3])) * sqrt(sum(X2.^2, [1 2 3]))));
    end
    
end

function Y = euclideanDistance(X1, X2)
    Y = sqrt(sum((double(X1) - double(X2)).^2, [1 2 3]));
end

function Y = l2_norm(X)
    Y = X / sqrt(sum(X.^2, [1 2 3]));
end