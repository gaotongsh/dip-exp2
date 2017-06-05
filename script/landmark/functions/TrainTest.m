function [pred, acc] = TrainTest(Xs, Ys, Xt, Yt, Gs, inds_landmarks, C, needCross)

addpath SVM-KM/
addpath simplemkl/

idLogic = true(1,length(Ys));
idLogic(inds_landmarks) = false;
id1 = find(idLogic); 

M1 = length(id1);
M2 = length(inds_landmarks);
N = length(Yt);
P = length(Gs);

Kr = zeros(M2, M2, P);      Yr = Ys(inds_landmarks);
Kv = zeros(M1, M2, P);      Yv = Ys(id1);
Kt = zeros(N, M2, P); 
for p = 1 : P
    Kr(:,:,p) = Xs(inds_landmarks,:) * Gs{p} * Xs(inds_landmarks,:)';
    Kv(:,:,p) = Xs(id1,:) * Gs{p} * Xs(inds_landmarks,:)';
    Kt(:,:,p) = Xt * Gs{p} * Xs(inds_landmarks,:)';    
end
    
[~, scale] = AverageKernels(Kr,'');
Kr = NormalizeKernels(Kr,scale);    
Kv = NormalizeKernels(Kv,scale);
Kt = NormalizeKernels(Kt,scale);    

[options verbose] = SetOptionsOfSimpleMKL();

if exist('needCross', 'var') && needCross
    bestC = CrossSimpleMKL(Kr,Yr,Kv,Yv,CRange,length(unique(Ys)),options,verbose);
else
    bestC = C;
end
    
[pred, acc] = RunSimpleMKL(Kr,Yr,bestC,length(unique(Ys)),options,verbose,Kt,Yt);      

    
function [options verbose] = SetOptionsOfSimpleMKL()

verbose = 1;
options.algo='oneagainstall';
options.seuildiffsigma=1e-4;
options.seuildiffconstraint=0.1;
options.seuildualitygap=1e-2;
options.goldensearch_deltmax=1e-1;
options.numericalprecision=1e-8;
options.stopvariation=1;
options.stopKKT=1;
options.stopdualitygap=0;
options.firstbasevariable='first';
options.nbitermax=100;
options.seuil=0.;
options.seuilitermax=10;
options.lambdareg = 1e-6;
options.miniter=0;
options.verbosesvm=0;
options.efficientkernel=1;

    
function bestC = CrossSimpleMKL(K,Y,Kv,Yv,CRange,NClass,options,verbose)

bestAccy = 0;
for C = CRange    
    [beta,w,w0,pos,nbsv,SigmaH,obj] = mklmulticlass(K,Y,C,NClass,options,verbose);
    Kt = CombineMorKernel(Kv(:,pos,:), 1, beta);
    kernel='numerical';
    kerneloption.matrix=Kt;
    switch options.algo
    case 'oneagainstall'
        [ypred,maxi] = svmmultival([],[],w,w0,nbsv,kernel,kerneloption);
    case 'oneagainstone'
        [ypred,vote]=svmmultivaloneagainstone([],[],w,w0,nbsv,kernel,kerneloption);
    end
    accy = mean(ypred==Yv);
    if bestAccy < accy
        bestAccy = accy;
        bestC = C;
    end
end


function [pred, accy, beta w w0 pos nbsv SigmaH obj] = RunSimpleMKL(Kr,Yr,C,NClass,options,verbose, Kt,Yt)

[beta,w,w0,pos,nbsv,SigmaH,obj] = mklmulticlass(Kr,Yr,C,NClass,options,verbose);
Kt = CombineMorKernel(Kt(:,pos,:), 1, beta);
kernel='numerical';
kerneloption.matrix=Kt;

switch options.algo
case 'oneagainstall'
    [pred,maxi] = svmmultival([],[],w,w0,nbsv,kernel,kerneloption);
case 'oneagainstone'
    [pred,vote] = svmmultivaloneagainstone([],[],w,w0,nbsv,kernel,kerneloption);
end

accy = mean(pred==Yt);


function [A scale] = AverageKernels(M,type)

A = zeros(size(M(:,:,1)));
scale = zeros(size(M,3),1);
for k = 1 : length(scale)
    if strcmp(type,'median')
        tmp = M(:,:,k);
        scale(k) = median(tmp(:));
    else
        scale(k) = trace(M(:,:,k));
    end
    if abs(scale(k)) < 1e-20
        scale(k) = 1e-20;
    end
    scale(k) = 1/scale(k);
    A = A + M(:,:,k) * scale(k);
end
A = A/size(M,3);


function K = NormalizeKernels(M,scale)
K = zeros(size(M));
for k = 1 : size(M,3)
    K(:,:,k) =  M(:,:,k) * scale(k);
end


function A = CombineMorKernel(M, scale, weight)

if length(weight)==1
    weight = ones(length(M),1) * weight;
end

if length(scale)==1
    scale = ones(length(M),1) * scale;
end

A = zeros(size(M(:,:,1)));
for k = 1 : size(M,3)
    A = A + M(:,:,k) * scale(k) * weight(k);
end
