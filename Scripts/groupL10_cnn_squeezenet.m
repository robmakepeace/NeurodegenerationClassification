clear all;
%Load the workspace variables
% load('groupL10_fulldataset227.mat')
% load('groupL10_trainingdata227.mat')
% load('groupL10_validationdata227.mat')
% imdsSNTrain = imdsTrain;
% imdsSNVal = imdsValidation;
load('groupL10_fulldataset.mat')
load('groupL10_trainingdata.mat')
load('groupL10_validationdata.mat')
%input layer expects images of size 227×227×3
imageSize = [227 227];
imdsSNTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing','gray2rgb');
imdsSNVal = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing','gray2rgb');

%Randomise the training data to make it better on validation
%X/Y Reflections
%Pixel Translations
imageAugmenter = imageDataAugmenter( ...      
    'RandXReflection',1, ...
    'RandYReflection',1, ...
    'RandRotation', [-90 90],...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);
%Random the training data images
imageSize = [config.imageresolution config.imageresolution 1];
imdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%Configure CNN layers (Empty Alexnet)
layersSN = squeezenet('Weights','imagenet');

layersTransfer = layersSN.Layers(1:end-2);
%Transfer the layers to the new classification task by replacing the last three layers with a fully connected layer, a softmax layer, and a classification output layer. Specify the options of the new fully connected layer according to the new data. Set the fully connected layer to have the same size as the number of classes in the new data. To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer.
numClasses = config.numberofclasses;
layers = [
    layersTransfer
    dropoutLayer('Name','drop10')
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','fc2')
    softmaxLayer('Name','prob2')
    classificationLayer('Name','ClassificationLayer_predictions')];

%Create new LayerGraph
SNlayerGraph1 = layerGraph(layers);
%Remove the linear connections in the layer graph
SNlayerGraph2 = SNlayerGraph1;
for i=1:69
    SNlayerGraph2 = disconnectLayers(SNlayerGraph2,string(SNlayerGraph1.Connections(i,1).Source),string(SNlayerGraph1.Connections(i,2).Destination));
end
%Rebuild the original Google Net connections in the layer graph
SNlayerGraph3 = SNlayerGraph2;
for i=1:73
    SNlayerGraph3 = connectLayers(SNlayerGraph3,string(layersSN.Connections(i,1).Source),string(layersSN.Connections(i,2).Destination));
end
SNlayerGraph3 = connectLayers(SNlayerGraph3,'pool10','drop10');
SNlayerGraph3 = connectLayers(SNlayerGraph3,'drop10','fc2');
SNlayerGraph3 = connectLayers(SNlayerGraph3,'fc2','prob2');
SNlayerGraph3 = connectLayers(SNlayerGraph3,'prob2','ClassificationLayer_predictions');
%Plotting
% figure
% plot(layersSN)
% title('Original SqueezeNet')
% figure
% plot(SNlayerGraph1)
% title('Reconstructed SqueezeNet - Linear')
% figure
% plot(SNlayerGraph3)
% title('Reconstructed SqueezeNet - LayerNetwork')

%Configure CNN options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.00005, ...
    'LearnRateSchedule','none', ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',0.95,... 
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsSNVal, ...
    'ValidationFrequency',10, ...
    'Verbose',true, ...
    'OutputNetwork','best-validation-loss',...
    'Plots','training-progress');

% Run CNN Training
[net, info] = trainNetwork(imdsSNTrain ,SNlayerGraph3,options);

%Classify the validation images
[YPred, scores] = classify(net,imdsSNVal);

%Validation Dataset labels
YValidation = imdsValidation.Labels;

%Caluclate CCN classification accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)

%Analyze Network
analyzeNetwork(net)

%Save some variables for the essemble CNN
save('groupL10_cnn_squeezenet.mat','YPred','scores','info')