clear all;
%Load the workspace variables
% load('groupL10_fulldataset224.mat')
% load('groupL10_trainingdata224.mat')
% load('groupL10_validationdata224.mat')
% imdsRNTrain = imdsTrain;
% imdsRNVal = imdsValidation;
load('groupL10_fulldataset.mat')
load('groupL10_trainingdata.mat')
load('groupL10_validationdata.mat')
%input layer expects images of size 224×224×3
imageSize = [224 224];
imdsRNTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing','gray2rgb');
imdsRNVal = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing','gray2rgb');

%Randomise the training data to make it better on validation
%X/Y Reflections
%Pixel Translations
imageAugmenter = imageDataAugmenter( ...      
    'RandXReflection',1, ...
    'RandYReflection',1, ...
    'RandRotation', [-180 180],...
    'RandXTranslation',[-20 20], ...
    'RandYTranslation',[-20 20]);
%Random the training data images
imageSize = [config.imageresolution config.imageresolution 1];
imdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%Configure CNN layers (Empty Resnet)
layersRN = resnet50('Weights','imagenet');

layersTransfer = layersRN.Layers(1:end-3);
%Transfer the layers to the new classification task by replacing the last three layers with a fully connected layer, a softmax layer, and a classification output layer. Specify the options of the new fully connected layer according to the new data. Set the fully connected layer to have the same size as the number of classes in the new data. To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer.
numClasses = config.numberofclasses;
layers = [
    layersTransfer
    dropoutLayer(0.5,'Name','drop2')
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc1000')
    softmaxLayer('Name','fc1000_softmax')
    classificationLayer('Name','ClassificationLayer_fc1000')];

%Create new LayerGraph
RNlayerGraph1 = layerGraph(layers);
%Remove the linear connections in the layer graph
RNlayerGraph2 = RNlayerGraph1
for i=1:177
    RNlayerGraph2 = disconnectLayers(RNlayerGraph2,string(RNlayerGraph1.Connections(i,1).Source),string(RNlayerGraph1.Connections(i,2).Destination));
end
%Rebuild the original ResNet connections in the layer graph
RNlayerGraph3 = RNlayerGraph2;
for i=1:189
    RNlayerGraph3 = connectLayers(RNlayerGraph3,string(layersRN.Connections(i,1).Source),string(layersRN.Connections(i,2).Destination));
end
RNlayerGraph3 = connectLayers(RNlayerGraph3,'avg_pool','drop2');
RNlayerGraph3 = connectLayers(RNlayerGraph3,'drop2','fc1000');
RNlayerGraph3 = connectLayers(RNlayerGraph3,'fc1000','fc1000_softmax');
RNlayerGraph3 = connectLayers(RNlayerGraph3,'fc1000_softmax','ClassificationLayer_fc1000');
%Plotting
% figure
% plot(layersRN)
% title('Original ResNet')
% figure
% plot(RNlayerGraph1)
% title('Reconstructed ResNet - Linear')
% figure
% plot(RNlayerGraph3)
% title('Reconstructed ResNet50 - LayerNetwork')

%Configure CNN options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',1, ...
    'LearnRateDropFactor',0.9,...
    'MiniBatchSize',24,...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsRNVal, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...
    'OutputNetwork','best-validation-loss',...
    'Plots','training-progress');

% Run CNN Training
[net, info] = trainNetwork(imdsRNTrain,RNlayerGraph3,options);

%Classify the validation images
[YPred, scores] = classify(net,imdsRNVal);

%Validation Dataset labels
YValidation = imdsValidation.Labels;

%Caluclate CCN classification accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)

%Analyze Network
analyzeNetwork(net)

%Save some variables for the essemble CNN
save('groupL10_cnn_resnet50.mat','YPred','scores','info')