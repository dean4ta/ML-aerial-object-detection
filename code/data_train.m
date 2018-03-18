function data_train()
%%% data_train.m %%%
% *** run from project root directory, 'code' dir in path ***
% loads data_train.mat and saves out data_train.png
% compatible with common viewers and reliable in quality

clear all; %#ok<CLALL>
addpath('./data');

disp('Loading...');
tmp = load('data_train.mat');
imwrite(tmp.data_train, './data/data_train_matlab.png');
disp('Done!');

end
