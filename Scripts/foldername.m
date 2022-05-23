topLevelFolder = 'G:\guest-20220107_223007\OAS2_0001\OAS2_0001_MR2\1';
% Get a list of all files and folders in this folder.
files = dir(topLevelFolder);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags); % A structure with extra info.
% Get only the folder names into a cell array.
subFolderNames = string({subFolders(3:end).name})
