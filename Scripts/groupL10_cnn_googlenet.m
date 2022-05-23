clear all;
%Load the workspace variables
% load('groupL10_fulldataset224.mat')
% load('groupL10_trainingdata224.mat')
% load('groupL10_validationdata224.mat')
% imdsGNTrain = imdsTrain;
% imdsGNVal = imdsValidation;
load('groupL10_fulldataset.mat')
load('groupL10_trainingdata.mat')
load('groupL10_validationdata.mat')
%input layer expects images of size 224×224×3
imageSize = [224 224];
imdsGNTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing','gray2rgb');
imdsGNVal = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing','gray2rgb');

%Randomise the training data to make it better on validation
%X/Y Reflections
%Pixel Translations
imageAugmenter = imageDataAugmenter( ...      
    'RandXReflection',1, ...
    'RandYReflection',1, ...
    'RandRotation', [-180 1800],...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);
%Random the training data images
imageSize = [config.imageresolution config.imageresolution 1];
imdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%Configure CNN layers (Empty Googlenet)
layersGN = googlenet('Weights','imagenet');

%Modify the Googlenet classification layer to two classes only.
layersTransfer = layersGN.Layers(1:end-3);
%Transfer the layers to the new classification task by replacing the last three layers with a fully connected layer, a softmax layer, and a classification output layer. Specify the options of the new fully connected layer according to the new data. Set the fully connected layer to have the same size as the number of classes in the new data. To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer.
numClasses = config.numberofclasses;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,'Name','loss3-classifier')
    softmaxLayer('Name','prob')
    classificationLayer('Name','output')];
%Create new LayerGraph
GNlayerGraph1 = layerGraph(layers);
%Remove the linear connections in the layer graph
GNlayerGraph2 = GNlayerGraph1
for i=1:143
    GNlayerGraph2 = disconnectLayers(GNlayerGraph2,string(GNlayerGraph1.Connections(i,1).Source),string(GNlayerGraph1.Connections(i,2).Destination));
end
%Rebuild the original Google Net connections in the layer graph
GNlayerGraph3 = GNlayerGraph2;
for i=1:170
    GNlayerGraph3 = connectLayers(GNlayerGraph3,string(layersGN.Connections(i,1).Source),string(layersGN.Connections(i,2).Destination));
end
%Plotting
% figure
% plot(layersGN)
% title('Original GoogleNet')
% figure
% plot(GNlayerGraph1)
% title('Reconstructed GoogleNet - Linear')
% figure
% plot(GNlayerGraph3)
% title('Reconstructed GoogleNet - LayerNetwork')

%Configure CNN options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.00001, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',24,...    
    'OutputNetwork','best-validation-loss',...    
    'Shuffle','every-epoch', ...
    'ValidationData',imdsGNVal, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...
    'OutputNetwork','best-validation-loss',...    
    'Plots','training-progress');

% Run CNN Training
[net, info] = trainNetwork(imdsGNTrain,GNlayerGraph3,options);

%Classify the validation images
[YPred, scores] = classify(net,imdsGNVal);
 
%Validation Dataset labels
YValidation = imdsValidation.Labels;
 
%Caluclate CCN classification accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)
 
%Analyze Network
%analyzeNetwork(net)

%Save some variables for the essemble CNN
save('groupL10_cnn_googlenet.mat','YPred','scores','info')