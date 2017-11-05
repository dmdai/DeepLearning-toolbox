function nn = saefoldtonn(sae, outputsize)

if exist('outputsize', 'var')
    sizes = [sae.sizes outputsize];
else
    sizes = sae.sizes;
end

nn = nnsetup(sizes);
for l = 1 : numel(sae.ae)
    nn.W{l} = sae.ae{l}.W{1};
end

end