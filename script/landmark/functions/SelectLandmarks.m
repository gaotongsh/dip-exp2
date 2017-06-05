function Beta = SelectLandmarks(Xs, Ys, Xt, D, ratio)

M = length(Ys);
N = size(Xt,1);

Pt = princomp(Xt);
G = GFK(princomp(Xs), Pt(:,1:D));

% ----------- discover landmarks at each scale --------
K = [Xs;Xt]*G*[Xs;Xt]';
D2 = repmat(diag(K),1,size(K,1)) + repmat(diag(K)',size(K,1),1) - 2*K;
dm2 = median(D2(:));

% options for QP solver of quadprog. Depending on your configuration and
% platform, you may need re-configurate the options yourself
% options = optimset('Display','iter', 'MaxIter', 1000, 'Algorithm', 'active-set');
options = optimset('Display','final', 'MaxIter', 1500, 'Algorithm', 'interior-point-convex','TolFun',1e-15);      
Beta = zeros(length(ratio), M);
for k = 1 : length(ratio)
    r = ratio(k);
    K = exp(-r*D2/dm2);
    H = 2*K(1:M,1:M);   f = -2/N*sum(K(1:M,M+1:end),2);
    A = -OneOfKEncoding(Ys)';   b = sum(A,2)/M/2;   
    Aeq = ones(1,M);            beq = 1;    
    lb = zeros(M,1);            ub = [];
    
    x = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
    
    Beta(k,:) = x;  %k = k+1;
end
