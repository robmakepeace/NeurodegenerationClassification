%assignment2_1_image.m
% Ass 2 Question 2.1
%Robert Makepeace, 13886357

%Resizes all images of the dataset from various sizes to a common size.
%Converts colour images to greyscale

clear all;
%Load in demographic data
load('groupL10_data.mat')

Path = 'C:\Users\robma\Documents\MATLAB\b_64_64\';
ExportPath = 'C:\Users\robma\Documents\MATLAB\b_64_64_sigmoid\';
exportfileformat = ".bmp";
imageresolution = 64;

for i=2:373
    for j=1:1
        string(i) + '/373'
        MyString = string(Data.Group(i,:).Group);
        fid1 = string(Path) + string(MyString) + '\';
        fid2 = string(ExportPath) + string(MyString) + '\';
        if (exist(fid1,'dir'))
            Y = imread(fid1 + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_" + imageresolution+"_"+imageresolution + exportfileformat);
            %Rsize to square common image resolution
            %Z = imresize(Y,[imageresolution imageresolution]);
            
            X = double(Y)./255;
        
            %Z = 255 .* (W - min(min(W)) ) ./ (max(max(W)) - min(min(W)) );
            %
            %Weight Vector
            %W =[ -0.6513 -0.5336 0.5105 0.2786 -0.0494; -0.2987 0.1285 -0.3343 -0.3533 -0.0721; 0.8872 0.2361 0.5571 -0.9600 0.8414; -0.9651 -0.7798 0.7455 0.3309 0.0534; -0.3020 0.1279 -0.7926 0.7424 0.3608];
            
            % convolution
            %result = conv2(Y ,W, 'valid'); 
            
            % sigmoid activation function
            s = 1./(1+exp(-1*X)); 
         
%             figure
%             subplot(1,3,1);
%             imshow(Y);
%             title('Image');
%             subplot(1,3,2);
%             imshow(uint8(result));
%             title('Feature after convolution');
%             subplot(1,3,3);
%             imshow(s);
%             title('Feature after sigmoid activation function');

            %Write photo
            imwrite(T,fid2 + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_" + imageresolution+"_"+imageresolution + exportfileformat);
        end
    end
end   
