function VGGFace_ExternalPrecompute(trainPath,embeddedDatabasePath, batchSize)
    
    paramsPath = fullfile("Weights","VGGFace","VGGFaceParams.mat");

    model = VGGFace_LoadModel(paramsPath);

    imdsTrain = imageDatastore(trainPath, ...
            "IncludeSubfolders",true, ...
            "LabelSource","foldernames");
    
    precomputeDatabase(imdsTrain, embeddedDatabasePath, model, @VGGFace_Normalize, batchSize);
    
end

