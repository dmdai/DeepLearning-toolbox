% function test_example_CNN

clear all;close all;clc

load mnist_uint8;

mu = mean(train_x / 255);
train_x = bsxfun(@minus, double(train_x) / 255, mu);
test_x = bsxfun(@minus, double(test_x) / 255, mu);

train_x = double(reshape(train_x',28,28,60000));
test_x = double(reshape(test_x',28,28,10000));
train_y = double(train_y');
test_y = double(test_y');



% train_x = double(reshape(train_x',28,28,60000))/255;
% test_x = double(reshape(test_x',28,28,10000))/255;
% train_y = double(train_y');
% test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 3) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 5;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');





% batch_x = rand(28,28,5);
% batch_y = rand(10,5);
% cnn.layers = {
%     struct('type', 'i') %input layer
%     struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %sub sampling layer
%     struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %subsampling layer
% };
% cnn = cnnsetup(cnn, batch_x, batch_y);
% 
% cnn = cnnff(cnn, batch_x);
% cnn = cnnbp(cnn, batch_y);
% cnnnumgradcheck(cnn, batch_x, batch_y);
% fprintf('Congratulations!!!\n');
