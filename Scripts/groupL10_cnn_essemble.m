clear all;

threshold = 0.60;
offset = 60;

load('groupL10_validationdata.mat')
YValidation = imdsValidation.Labels;

load('groupL10_cnn.mat');
cnn_YPred = YPred;
cnn_scores = scores;
cnn_info = info;
cnn_train_acc = cnn_info.TrainingAccuracy(length(cnn_info.TrainingAccuracy))
cnn_accuracy = sum(cnn_YPred == YValidation)/numel(YValidation)
if (cnn_accuracy>threshold)
    cnn_weight = 100*cnn_accuracy - offset
else
    cnn_weight = 0
end

load('groupL10_cnn_baseline.mat');
base_YPred = YPred;
base_scores = scores;
base_info = info;
base_train_acc = base_info.TrainingAccuracy(length(base_info.TrainingAccuracy))
base_accuracy = sum(base_YPred == YValidation)/numel(YValidation)
if (cnn_accuracy>threshold)
    base_weight = 100*base_accuracy - offset
else
    base_weight = 0
end


load('groupL10_cnn_squeezenet.mat');
squeeze_YPred = YPred;
squeeze_scores = scores;
squeeze_info = info;
squeeze_train_acc = squeeze_info.TrainingAccuracy(length(squeeze_info.TrainingAccuracy))
squeeze_accuracy = sum(squeeze_YPred == YValidation)/numel(YValidation)
if (squeeze_accuracy>threshold)
    squeeze_weight = 100*squeeze_accuracy - offset
else
    squeeze_weight = 0
end


load('groupL10_cnn_alexnet.mat');
alex_YPred = YPred;
alex_scores = scores;
alex_info = info;
alex_train_acc = alex_info.TrainingAccuracy(length(alex_info.TrainingAccuracy))
alex_accuracy = sum(alex_YPred == YValidation)/numel(YValidation)
if (alex_accuracy>threshold)
    alex_weight = 100*alex_accuracy - offset
else
    alex_weight = 0
end


load('groupL10_cnn_googlenet.mat');
google_YPred = YPred;
google_scores = scores;
google_info = info;
google_train_acc = google_info.TrainingAccuracy(length(google_info.TrainingAccuracy));
google_accuracy = sum(google_YPred == YValidation)/numel(YValidation)
if (google_accuracy>threshold)
    google_weight = 100*google_accuracy - offset
else
    google_weight = 0
end


load('groupL10_cnn_resnet18.mat');
res18_YPred = YPred;
res18_scores = scores;
res18_info = info;
res18_train_acc = res18_info.TrainingAccuracy(length(res18_info.TrainingAccuracy))
res18_accuracy = sum(res18_YPred == YValidation)/numel(YValidation)
if (res18_accuracy>threshold)
    res18_weight = 100*res18_accuracy - offset
else
    res18_weight = 0
end


load('groupL10_cnn_resnet50.mat');
res50_YPred = YPred;
res50_scores = scores;
res50_info = info;
res50_train_acc = res50_info.TrainingAccuracy(length(res50_info.TrainingAccuracy))
res50_accuracy = sum(res50_YPred == YValidation)/numel(YValidation)
if (res50_accuracy>threshold)
    res50_weight = 100*res50_accuracy - offset
else
    res50_weight = 0
end

essemble_scores = (cnn_weight .* cnn_scores + ...
    base_weight .* base_scores + ...
    squeeze_weight .* squeeze_scores + ...
    alex_weight .* alex_scores + ...
    google_weight .* google_scores + ...
    res18_weight .* res18_scores + ...
    res50_weight .* res50_scores) / ...
(cnn_weight + base_weight+ squeeze_weight + alex_weight + google_weight + res18_weight + res50_weight);

for i=1:100
    if essemble_scores(i,1) > essemble_scores(i,2)
        essemble_YPred(i,:) = "Demented";
    else  
        essemble_YPred(i,:) = "Nondemented";
    end
end
essemble_YPred2 = categorical(essemble_YPred);
accuracy = sum(essemble_YPred2 == YValidation)/numel(YValidation)

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(YValidation,essemble_YPred2);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';