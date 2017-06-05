function [Alpha inds_landmarks] = Beta2Alpha(Beta, Ys, threshold)
% [Alpha inds_landmarks] = Beta2Alpha(Beta, Ys, threshold)
% Threshold Beta to recover the indicator variables Alpha.
% After thresholding, the class balance constraint could be violated! We
% correct this issue by randomly flipping the indicators of some landmarks.
% For details please see the program. 

% Output: Alpha and landmark indices

Alpha = false(size(Beta));

inds_landmarks = [];
for i = 1 : size(Beta,1)
    a = (Beta(i,:)>threshold);
    if sum(a) > 2/3*length(Ys)
        continue;
    else
        Alpha(i,:) = a;
    end
    inds_landmarks = [inds_landmarks, find(a)];
end

Alpha = Alpha(any(Alpha,2),:);
inds_landmarks = tideIDX(inds_landmarks,Ys);

    
function saveID2 = tideIDX(id2,Y)
idLogic = true(1,length(Y));
idLogic(id2) = false;
id1 = find(idLogic);   % non-landmarks

saveID2 = id2;         % initial landmarks
id2 = unique(id2);
Y2 = Y(id2);
C = length(unique(Y));
for c = 1 : C
    num = sum(Y==c);
    num2 = sum(Y2==c);
    if num2 < 1/3*num
        id = intersect(find(Y'==c),id1);    
        moveNum = ceil(1/3*num) - num2;
        saveID2 = [saveID2, id(1:moveNum)];
        for j = 1 : moveNum
            id1(id1==id(j)) = id(moveNum+1);
        end
    end
end