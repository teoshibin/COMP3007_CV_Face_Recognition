function model = loadModel(weights_path)
% This will load the model in as a dlnetwork

% NOTE:
% The weights that I'm using here uses conv2d as fully Connected for the 
% last few layers

    params = load(weights_path);

    layers = [
        
        imageInputLayer([224 224 3],"Name","input","Normalization","none")
        
        convolution2dLayer([3 3],64,"Name","conv2d_1","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_1.Bias,"Weights",params.conv2d_1.Weights)
        reluLayer("Name","relu_1")
        convolution2dLayer([3 3],64,"Name","conv2d_2","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_2.Bias,"Weights",params.conv2d_2.Weights)
        reluLayer("Name","relu_2")
        maxPooling2dLayer([2 2],"Name","max_pooling2d_1","Stride",[2 2])
        
        convolution2dLayer([3 3],128,"Name","conv2d_3","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_3.Bias,"Weights",params.conv2d_3.Weights)
        reluLayer("Name","relu_3")
        convolution2dLayer([3 3],128,"Name","conv2d_4","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_4.Bias,"Weights",params.conv2d_4.Weights)
        reluLayer("Name","relu_4")
        maxPooling2dLayer([2 2],"Name","max_pooling2d_2","Stride",[2 2])
        
        convolution2dLayer([3 3],256,"Name","conv2d_5","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_5.Bias,"Weights",params.conv2d_5.Weights)
        reluLayer("Name","relu_5")
        convolution2dLayer([3 3],256,"Name","conv2d_6","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_6.Bias,"Weights",params.conv2d_6.Weights)
        reluLayer("Name","relu_6")
        convolution2dLayer([3 3],256,"Name","conv2d_7","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_7.Bias,"Weights",params.conv2d_7.Weights)
        reluLayer("Name","relu_7")
        maxPooling2dLayer([2 2],"Name","max_pooling2d_3","Stride",[2 2])
        
        convolution2dLayer([3 3],512,"Name","conv2d_8","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_8.Bias,"Weights",params.conv2d_8.Weights)
        reluLayer("Name","relu_8")
        convolution2dLayer([3 3],512,"Name","conv2d_9","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_9.Bias,"Weights",params.conv2d_9.Weights)
        reluLayer("Name","relu_9")
        convolution2dLayer([3 3],512,"Name","conv2d_10","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_10.Bias,"Weights",params.conv2d_10.Weights)
        reluLayer("Name","relu_10")
        maxPooling2dLayer([2 2],"Name","max_pooling2d_4","Stride",[2 2])
        
        convolution2dLayer([3 3],512,"Name","conv2d_11","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_11.Bias,"Weights",params.conv2d_11.Weights)
        reluLayer("Name","relu_11")
        convolution2dLayer([3 3],512,"Name","conv2d_12","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_12.Bias,"Weights",params.conv2d_12.Weights)
        reluLayer("Name","relu_12")
        convolution2dLayer([3 3],512,"Name","conv2d_13","Padding",[1 1 1 1],"WeightLearnRateFactor",0,"Bias",params.conv2d_13.Bias,"Weights",params.conv2d_13.Weights)
        reluLayer("Name","relu_13")
        maxPooling2dLayer([2 2],"Name","max_pooling2d_5","Stride",[2 2])
        
        convolution2dLayer([7 7],4096,"Name","conv2d_14","WeightLearnRateFactor",0,"Bias",params.conv2d_14.Bias,"Weights",params.conv2d_14.Weights)
        reluLayer("Name","relu_14")
        dropoutLayer(0.5,"Name","dropout_1")
        convolution2dLayer([1 1],4096,"Name","conv2d_15","WeightLearnRateFactor",0,"Bias",params.conv2d_15.Bias,"Weights",params.conv2d_15.Weights)
        reluLayer("Name","relu_15")
        dropoutLayer(0.5,"Name","dropout_2")
        convolution2dLayer([1 1],2622,"Name","conv2d_16","WeightLearnRateFactor",0,"Bias",params.conv2d_16.Bias,"Weights",params.conv2d_16.Weights)
        
    ];

    lgraph = layerGraph(layers); % compiling the layers, sort of
    model = dlnetwork(lgraph); % to use custom predict function

end