function Ex_Landmark(src, tgt, d) 

% This demonstrates the landmark-based approach for domain adaptation. It consists
% of three components,
% 1) to identify the landmarks from the source domain,
% 2) to construct auxiliary domain adaptation tasks and solve them by GFK [2], and
% 3) training with landmarks and the kernels of 2), to obtain the target
% classifier

% --------- Four domains: { Caltech10, amazon, webcam, dslr } ------------
d = 20; % subspace dimension, the following dims are used in the paper:
% webcam-dslr: 10
% dslr-amazon: 20
% webcam-amazon: 10
% caltech-webcam: 20
% caltech-dslr: 10
% caltech-amazon: 20
% Note the dim from X to Y is the same as that from Y to X.

% Ref:
% [1] Connecting the Dots with Landmarks: Discriminatively Learning Domain-
% Invariant Features for Unsupervised Domain Adaptation. B. Gong, K. Grauman, and F. Sha.
% Proceedings of the International Conference on Machine Learning (ICML), Atlanta, GA, June 2013.
%
% [2] Geodesic Flow Kernel for Unsupervised Domain Adaptation.  
% B. Gong, Y. Shi, F. Sha, and K. Grauman.  
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

% Contact: Boqing Gong (boqinggo@usc.edu)


addpath functions

% --------- parameters ---------------
sigmas = 2.^(-6:6);     % scales at each of which to select landmarks
threshold = 1e-8;       % threshold for converting solved real-valued beta to binary alpha
C = 2^10;               % regularization cost in multiple kernel learning

%------ prepare data -----------------------
[Xs, Ys, Xt, Yt] = PrepareData(src, tgt);
load('cat_train.mat');
load('cat_test.mat');
Xss = cat_train(:,1:4096);
Xtt = cat_test(:,1:4096);

Xss = Xss ./ repmat(sum(Xss,2),1,size(Xss,2)); 
Xs = zscore(Xss,1);
Xtt = Xtt ./ repmat(sum(Xtt,2),1,size(Xtt,2)); 
Xt = zscore(Xtt,1);

Ys = cat_train(:,4097) + 1;
%Yt = cat_validation(:,4097) + 1;
Yt = Xt(:,1);

% -----------1) Select landmarks------
fprintf('identifying landmarks...\n')
Beta = [];
fn = ['Landmarks/landmark_' src '_' tgt '.mat'];
if ~exist(fn, 'file')
	Beta = SelectLandmarks(Xs, Ys, Xt, d, sigmas);
	save(fn, 'Beta', 'sigmas');
else
	load(fn, 'Beta', 'sigmas');
end

[Alpha inds_landmarks] = Beta2Alpha(Beta, Ys, threshold);    clear Beta

% ---- 2) Construct auxiliary tasks and solve them by GFK ---------
fprintf('solving auxiliary tasks...\n')
Gs = SolveAuxiliaryTasks(Xs, Xt, d, Alpha);

% -- 3) Learn classifier using landmarks as proxy to target discriminativess --
% Here linear kernels (GFKs of 2)) are used. If one want to fairly contrast 
% the landmark approach to some nonlinear classifiers, s/he can easily
% extend the GFKs to nonlinear kernels like RBFs. -------------------------
fprintf('training and testing...\n')
[pred, acc] = TrainTest(Xs, Ys, Xt, Yt, Gs, inds_landmarks, C);
pred = pred - 1;

% disp(acc)
% fid = fopen('results.txt', 'a');
% fprintf(fid, '%s %s-->%s: %g\n', datestr(now), src, tgt, acc);
% fclose(fid);

save(['cat_pred.mat'], 'pred');
