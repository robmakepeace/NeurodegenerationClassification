clear all;
% Load the workspace variables
% load('groupL10_fulldataset227.mat')
% load('groupL10_trainingdata227.mat')
% load('groupL10_validationdata227.mat')
% imdsANTrain = imdsTrain;
% imdsANVal = imdsValidation;
load('groupL10_fulldataset.mat')
load('groupL10_trainingdata.mat')
load('groupL10_validationdata.mat')
%input layer expects images of size 227×227×3
imageSize = [227 227];
imdsANTrain = augmentedImageDatastore(imageSize, imdsTrain, 'ColorPreprocessing','gray2rgb');
imdsANVal = augmentedImageDatastore(imageSize, imdsValidation, 'ColorPreprocessing','gray2rgb');

%Randomise the training data to make it better on validation
%X/Y Reflections
%Pixel Translations
imageAugmenter = imageDataAugmenter( ...      
    'RandXReflection',1, ...
    'RandYReflection',1, ...
    'RandRotation', [0 0],...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);
%Random the training data images
imageSize = [config.imageresolution config.imageresolution 1];
imdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%Configure CNN layers (Empty Alexnet)
layersAN = alexnet('Weights','imagenet');

layersTransfer = layersAN.Layers(1:end-3);
%Transfer the layers to the new classification task by replacing the last three layers with a fully connected layer, a softmax layer, and a classification output layer. Specify the options of the new fully connected layer according to the new data. Set the fully connected layer to have the same size as the number of classes in the new data. To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer.
numClasses = config.numberofclasses;
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%Configure CNN options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.00005, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsANVal, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...
    'OutputNetwork','best-validation-loss',...
    'Plots','training-progress');

% Run CNN Training
[net, info] = trainNetwork(imdsANTrain ,layers,options);

%Classify the validation images
[YPred, scores] = classify(net,imdsANVal);

%Validation Dataset labels
YValidation = imdsValidation.Labels;

%Caluclate CCN classification accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)

%Analyze Network
%analyzeNetwork(net)

%Save some variables for the essemble CNN
save('groupL10_cnn_alexnet.mat','YPred','scores','info')