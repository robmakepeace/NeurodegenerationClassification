%example CNN code from: https://au.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html

%Load the workspace variables
load('groupL10_fulldataset.mat')
load('groupL10_trainingdata.mat')
load('groupL10_validationdata.mat')

%Convolution layer - filter layers
config.convfilterlayers = 32;
%Convolution layer - filter layers increase each layer (i.e.
%24,28,32,36
config.convfilteroffset = 4;
%Convolution layer - filter size
config.convfiltersize = 5;
%Convolution layer - padding 
config.convpadding = 'same';
%Convolution layer - Stride
config.convstride = 1;
%Pooling Layer - Filter size
config.poolfiltersize = 2;
%Pooling Layer - Stride
config.poolstride = 2;
%Initial Learn Rate 
config.InitialLearnRate = 0.001;
%Learn Rate Schedule 
config.LearnRateSchedule = 'piecewise';
%Learn Rate Drop Period
config.LearnRateDropPeriod = 10;
%Learn Rate Drop Factor
config.LearnRateDropFactor = 0.8;
%max Number of Epochs
config.MaxEpochs = 50;

%Randomise the training data to make it better on validation
%X/Y Reflections
% %Pixel Translations
 imageAugmenter = imageDataAugmenter( ...      
     'RandXReflection',1, ...
     'RandYReflection',1, ...
     'RandRotation', [0 0],...
     'RandXTranslation',[-5 5], ...
     'RandYTranslation',[-5 5]);
% %Random the training data images
imageSize = [config.imageresolution config.imageresolution 1];
imdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%figure
%minibatch = preview(imdsTrain);
%imshow(imtile(minibatch.input));

%Generate figure of random MRI images
%figure;
%perm = randperm(config.numberOfImages,64);
%for i = 1:64
%    subplot(8,8,i);
%    imshow(imds.Files{perm(i)});
%end

%Configure CNN layers
layers = [
    imageInputLayer([config.imageresolution config.imageresolution 1])
    
    convolution2dLayer(config.convfiltersize,config.convfilterlayers,'Padding',config.convpadding, 'Stride',config.convstride)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(config.poolfiltersize,'Stride',config.poolstride)
    
    convolution2dLayer(config.convfiltersize,config.convfilterlayers+1*config.convfilteroffset,'Padding',config.convpadding, 'Stride',config.convstride)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(config.poolfiltersize,'Stride',config.poolstride)
    
    convolution2dLayer(config.convfiltersize,config.convfilterlayers+2*config.convfilteroffset,'Padding',config.convpadding, 'Stride',config.convstride)
    batchNormalizationLayer
    reluLayer

    %maxPooling2dLayer(config.poolfiltersize,'Stride',config.poolstride)
    
    %convolution2dLayer(config.convfiltersize,config.convfilterlayers+3*config.convfilteroffset,'Padding',config.convpadding, 'Stride',config.convstride)
    %batchNormalizationLayer
    %reluLayer

    %maxPooling2dLayer(config.poolfiltersize,'Stride',config.poolstride)
    
    %convolution2dLayer(config.convfiltersize,config.convfilterlayers+3*config.convfilteroffset,'Padding',config.convpadding, 'Stride',config.convstride)
    %batchNormalizationLayer
    %reluLayer
    
    %dropoutLayer(0.5)
    %fullyConnectedLayer(128,"BiasLearnRateFactor",1,"WeightLearnRateFactor",1)
    %dropoutLayer(0.5)
    fullyConnectedLayer(config.numberofclasses,"BiasLearnRateFactor",2,"WeightLearnRateFactor",2)
    softmaxLayer
    classificationLayer];

%Configure CNN options
options = trainingOptions('adam', ...
    'InitialLearnRate',config.InitialLearnRate, ...
    'LearnRateSchedule',config.LearnRateSchedule, ...
    'LearnRateDropPeriod',config.LearnRateDropPeriod, ...
    'LearnRateDropFactor',config.LearnRateDropFactor,...
    'MaxEpochs',config.MaxEpochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',true, ...    
    'Plots','training-progress');

% Run CNN Training
[net, info] = trainNetwork(imdsTrain,layers,options);

%Classify the validation images
[YPred,scores] = classify(net,imdsValidation);

%Validation Dataset labels
YValidation = imdsValidation.Labels;

%Caluclate CCN classification accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)

%Analyze Network
analyzeNetwork(net)

%Save some variables for the essemble CNN
save('groupL10_cnn.mat','YPred','scores','info')

% figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
% cm = confusionchart(YValidation,YPred);
% cm.Title = 'Confusion Matrix for Validation Data';
% cm.ColumnSummary = 'column-normalized';
% cm.RowSummary = 'row-normalized';