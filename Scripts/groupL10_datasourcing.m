clear all
%Read in subject data spreadsheet
% T = readtable('oasis_longitudinal_demographics.csv','NumHeaderLines',0);
% %Write columns of data into struct
% Data.Subject_ID = T(:,1);
% Data.MRI_ID = T(:,2);
% Data.Group = T(:,3);
% Data.Visit = T(:,4);
% Data.MR_Delay = T(:,5);
% Data.Sex = T(:,6);
% Data.Handedness = T(:,7);
% Data.Age = T(:,8);
% Data.EDUC = T(:,9);
% Data.SES = T(:,10);
% Data.MMSE = T(:,11);
% Data.CDR = T(:,12);
% Data.eTIV = T(:,13);
% Data.nWBV = T(:,14);
% Data.ASF = T(:,15);
% save('groupL10_data.mat', 'Data')
load('groupL10_data.mat')

%Path = 'C:\Users\robma\Documents\MATLAB\';
Path = 'G:\';
ExportPath = 'C:\Users\robma\Documents\MATLAB\';
%Folder1 = 'guest-20220102_213800\';
Folder1 = 'guest-20220107_223007\';
%Folder2 = 'CENTRAL_OASIS_LONG\';
Folder2 = '';
fid_header  = '.hdr';
fid_image  = '.img';
imageresolution1 = 64;
imageresolution2 = 32;
imageresolution3 = 224;
imageresolution4 = 227;

exportfileformat= ".bmp";

for i=2:373
    for j=1:1
        string(i) + '/373'
        fid1 = string(Path) + string(Folder1) + string(Folder2) + string(Data.Subject_ID(i,:).SubjectID) + '\' + string(Data.MRI_ID(i,:).MRIID) + '\' + string(j) +'\';
        if (exist(fid1,'dir'))
            % Get a list of all files and folders in this folder.
            files = dir(fid1);
            % Get a logical vector that tells which is a directory.
            dirFlags = [files.isdir];
            % Extract only those that are directories.
            subFolders = files(dirFlags); % A structure with extra info.
            % Get only the folder names into a cell array.
            subFolderNames = string({subFolders(3:end).name});
            fid = fid1 + subFolderNames + '\';
            if (exist(fid,'dir'))
                filename = 'mpr-' + string(j) + '.nifti';
                info = analyze75info(fid + filename + fid_header);
                X = analyze75read(fid + filename + fid_image);         
                X = flip(X);
                %Extract a 64x64 bitmap of the brain
                Y = X(110,:,:);
                Y = reshape(Y,[256,128]);
                denom = (ceil(max(max(Y))/500)*500);
                config.Classifcation = 0;
                if (config.Classifcation == 0)
                    MyString = string(Data.Group(i,:).Group);
                end
                if (config.Classifcation == 1)
                    if (Data.CDR(i,:).CDR == 0)
                        MyString = "Nondemented";
                    end
                    if (Data.CDR(i,:).CDR == 0.5)
                        MyString = "Questionable";
                    end
                    if (Data.CDR(i,:).CDR == 1)
                        MyString = "MildDemented";
                    end
                    if (Data.CDR(i,:).CDR == 2)
                        MyString = "ModerateDemented";
                    end
                end

                Z1 = double(Y)/denom;
                imwrite(Z1,string(ExportPath) + 'b_256_128\' + MyString + '\' + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_256_128"+exportfileformat);
                Z2 = imresize(Z1,[imageresolution1 imageresolution1]);
                imwrite(Z2,string(ExportPath) + 'b_64_64\' + MyString + '\' + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_" + imageresolution1+"_"+imageresolution1+exportfileformat)
                Z3 = imresize(Z1,[imageresolution2 imageresolution2]);
                imwrite(Z3,string(ExportPath) + 'b_32_32\' + MyString + '\' + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_" + imageresolution2+"_"+imageresolution2+exportfileformat)
                Z4 = imresize(Z1,[imageresolution3 imageresolution3]);
                W1(:,:,1) = Z4;
                W1(:,:,2) = Z4;
                W1(:,:,3) = Z4;
                imwrite(W1,string(ExportPath) + 'b_224_224\' + MyString + '\' + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_" + imageresolution2+"_"+imageresolution2+exportfileformat)
                Z5 = imresize(Z1,[imageresolution4 imageresolution4]);
                W2(:,:,1) = Z5;
                W2(:,:,2) = Z5;
                W2(:,:,3) = Z5;
                imwrite(W2,string(ExportPath) + 'b_227_227\' + MyString + '\' + string(Data.Subject_ID(i,:).SubjectID) + '_' + string(Data.MRI_ID(i,:).MRIID) + "_" + j + "_" + imageresolution2+"_"+imageresolution2+exportfileformat)
            end
        end
    end
end
% 
% save('L10group_data.mat', 'Data')
% %load('L10group_data.mat')

