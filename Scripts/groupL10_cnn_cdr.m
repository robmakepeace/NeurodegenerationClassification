%example CNN code from: https://au.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html

%Load the workspace variables
load('groupL10_fulldataset_cdr.mat')
load('groupL10_trainingdata_cdr.mat')
load('groupL10_validationdata_cdr.mat')

%Generate figure of random MRI images
%figure;
perm = randperm(config.numberOfImages,9);
for i = 1:9
    subplot(3,3,i);
    imshow(imds.Files{perm(i)});
end
imageAugmenter = imageDataAugmenter( ...      
    'RandXReflection',1, ...
    'RandYReflection',1, ...
    'RandRotation', [0 0],...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);
%Random the training data images
imageSize = [config.imageresolution config.imageresolution 1];
imdsTrain = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%Configure CNN layers
layers = [
    imageInputLayer([config.imageresolution config.imageresolution 1])
    
    convolution2dLayer(3,36,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,72,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,36,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(config.numberofclasses)
    softmaxLayer
    classificationLayer];

%Configure CNN options
options = trainingOptions('adam', ...
    'MiniBatchSize',30,...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',2, ...
    'Verbose',true, ...
    'Plots','training-progress');

% Run CNN Training
net = trainNetwork(imdsTrain,layers,options);

%Classify the validation images
YPred = classify(net,imdsValidation);

%Validation Dataset labels
YValidation = imdsValidation.Labels;

%Caluclate CCN classification accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation)

%Analyze Network
analyzeNetwork(net)

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';