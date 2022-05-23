clear all;
close all;

%Load in demographic data
load('groupL10_data.mat')

%config.imageresolution=32;
config.imageresolution=64;
%config.imageresolution=224;
%config.imageresolution=227;

%Hardcoded number of images
config.numberOfImages=373;

%Hardcoded minimum number of category in each datastore
config.numTrainFiles = 117;

%load images as a image datastore
digitDatasetPath = 'C:\Users\robma\Documents\MATLAB\b_'+string(config.imageresolution)+'_'+string(config.imageresolution)+'\';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%Binary Classifcation: demented /non demented - Set Classifcation = 0
config.Classifcation = 0;

%Numerical Classifcation Clincial Demented Rating (CDR): NonDemented /
%Questionable / MildDemented / ModerateDemented - Set Classifcation = 1
%config.Classifcation = 1;

%Label each image with the categories
if (config.Classifcation == 0)
    %Use binary classifcation demented/non demented
    config.numberofclasses=2;
end
if (config.Classifcation == 1)
    %Use Clinical Dementia Rating (CDR) as classification
    %valueset = [0 0.5 1 2];
    %catnames = {'NonDemented'; 'Questionable'; 'MildDemented'; 'ModerateDemented'};
    %imds.Labels=categorical(Data.CDR(1:numberOfImages,:).CDR,valueset,catnames,'Ordinal',true);
    config.numberofclasses=4;
end

%Generate figure of random MRI images
% figure;
% perm = randperm(config.numberOfImages,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
% end

%count number of images
img = readimage(imds,1);
numberofimages = size(img);
labelCount = countEachLabel(imds)

%Divide images into training and validation sets
[imdsTrain,imdsValidation,imdsDitch] = splitEachLabel(imds,config.numTrainFiles,50,'randomize');
labelCountTrain = countEachLabel(imdsTrain)
labelCountValidation = countEachLabel(imdsValidation)
labelCountDitch = countEachLabel(imdsDitch)

%Save the workspace variables
if (config.Classifcation == 0)
    save('groupL10_fulldataset_sigmoid.mat','imds','config')
    save('groupL10_trainingdata_sigmoid.mat','imdsTrain','config')
    save('groupL10_validationdata_sigmoid.mat','imdsValidation','config')
end
if (config.Classifcation == 1)
    save('groupL10_fulldataset_cdr.mat','imds','config')
    save('groupL10_trainingdata_cdr.mat','imdsTrain','config')
    save('groupL10_validationdata_cdr.mat','imdsValidation','config')
end