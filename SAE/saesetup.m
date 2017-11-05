function sae = saesetup(size)
    
    curPath = pwd;
    nnPath = strcat(curPath(1:end-3), 'NN\');
    addpath(nnPath);
    clear curPath nnPath
    
    sae.sizes = size;
    for u = 2 : numel(size)
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)]);
    end
end
