function Gs = SolveAuxiliaryTasks(Xs, Xt, D, Alpha)

Gs = cell(size(Alpha,1),1);
for p = 1 : size(Alpha,1)
    idp1 = ~Alpha(p,:);
    idp2 = Alpha(p,:);
    
    Ps = princomp(Xs(idp1,:));
%    Pt = princomp([Xs(idp2,:);Xt]);
    [~,Pt] = nnmf([Xs(idp2,:);Xt], D);
    Gs{p} = GFK(Ps, Pt');
end