
clear all;close all;clc

load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% mu = mean(train_x);
% train_x = bsxfun(@minus, train_x, mu);
% test_x = bsxfun(@minus, test_x, mu);

opts.numepochs = 2;
opts.batchsize = 100;

sae = saesetup([784 400 100]);
sae.ae{1}.activation_function = 'sigm';
sae.ae{1}.learningRate = 1;
sae.ae{2}.activation_function = 'sigm';
sae.ae{2}.learningRate = 0.8;


sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:, 2:end));
visualize(sae.ae{2}.W{1}(:, 2:end));


nn = saefoldtonn(sae, 10);
nn.activation_function = 'sigm';
nn.learningRate = 1;
nn.scaling_learningRate = 0.9;
opts.numepochs = 1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);






% rand('state',0)
% sae = saesetup([784 400 100]);
% sae.ae{1}.activation_function       = 'sigm';
% sae.ae{1}.learningRate              = 1;
% sae.ae{1}.inputZeroMaskedFraction   = 0.5;
% opts.numepochs =   1;
% opts.batchsize = 100;
% sae = saetrain(sae, train_x, opts);
% visualize(sae.ae{1}.W{1}(:,2:end)')
% 
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([784 100 10]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 1;
% nn.W{1} = sae.ae{1}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   1;
% opts.batchsize = 100;
% nn = nntrain(nn, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.16, 'Too big error');
