clear all
close all

% part 1 - plot the false positive true positive
Probabilities = [ 0.967, 0.448, 0.568, 0.879, 0.015, 0.780, 0.978, 0.004];
Classifications = [1, 0, 1, 0, 1, 0, 1, 0];
[X, Y, thre, AUC]=perfcurve(Classifications, Probabilities,1);
figure(1);
plot(X,Y);
title('TruePositive rate vs. FalsePositive rate');
xlabel('FalsePositive rate');
ylabel('TruePositive rate');

% part 2 - task 1 - plot values & get data
load dataSet_1.mat
factors=glmfit(predictor, response, 'binomial');
prob=glmval(factors, predictor, 'logit');
[X, Y, thre, total_auc]=perfcurve(response, prob,1);
figure(2);
plot(X,Y);
title('ROC curve');
[total_auc]

% part 2 - task 2 - data split
training_predictor = predictor(1:3000);
training_response = response(1:3000);
validation_predictor = predictor(3001:4000);
validation_response = response(3001:4000);

factors=glmfit(training_predictor, training_response, 'binomial');
prob=glmval(factors, training_predictor, 'logit');
[X, Y, thre, training_auc]=perfcurve(training_response, prob,1);
figure(3);
plot(X,Y);
title('training ROC curve');
[training_auc]

factors=glmfit(validation_predictor, validation_response, 'binomial');
prob=glmval(factors, validation_predictor, 'logit');
[X, Y, thre, validation_auc]=perfcurve(validation_response, prob,1);
figure(4);
plot(X,Y);
title('validation ROC curve');
[validation_auc]

[training_auc - validation_auc]
