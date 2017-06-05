function [Xs, Ys, Xt, Yt] = PrepareData(src, tgt)

load(['data/' src '_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(fts,1);    clear fts
Ys = labels;           clear labels

load(['data/' tgt '_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(fts,1);     clear fts
Yt = labels;            clear labels