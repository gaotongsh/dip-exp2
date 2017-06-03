%%%%%%%%%%%%%%%%%%%%%%
%    DIP-Exp2
%    Naive
%%%%%%%%%%%%%%%%%%%%%%

load('../data/bird_train.mat')
load('../data/bird_validation_new.mat')

X = bird_train(:,1:4096);
Y = bird_train(:,4097);
Mdl = fitcensemble(X,Y);

test = bird_validation(:,1:4096);
result = Mdl.predict(test);
s = 0;
for i=1:690
    if result(i) == bird_validation(i,4097)
        s = s + 1;
    end
end

fprintf('Accuracy: %f\n', s/690);
