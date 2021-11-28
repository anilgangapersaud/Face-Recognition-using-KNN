%% Part 2. Face Detection
% Student Name: Anil Gangapersaud
% Student Number: 215178411

%% 1. Generate cropped not_face images
run('generate_cropped_notfaces.m');

%% 3. Generate HOG features for all training and validation images
run('get_features.m');

%% 2. Split Data into Training and Testing sets
disp("Splitting datasets into training and test sets...");
trainingData = zeros(10786,1116);
validationData = zeros(2700,1116);
trainingLabels = zeros(10786, 1);
validationLabels = zeros(2700,1);
j=1;
for i=1:5393
    trainingData(j,:) = pos_feats(i,:);
    trainingLabels(j) = 1;
    j=j+1;
end
for i=1:5393
    trainingData(j,:) = neg_feats(i,:);
    trainingLabels(j) = -1;
    j=j+1;
end
j=1;
for i=5394:6743
    validationData(j,:) = pos_feats(i,:);
    validationLabels(j) = 1;
    j=j+1;
end
for i=5394:6743
    validationData(j,:) = neg_feats(i,:);
    validationLabels(j) = -1;
    j=j+1;
end
disp("Finished splitting data!");
%% 4. Train an SVM on the features from training set.
lambda = 0.0001;
[w,b] = vl_svmtrain(trainingData',trainingLabels',lambda);
%% 5. Test SVM on the validation set features
fprintf('Classifier performance on train data:\n');
confidences = validationData*w + b;
save('my_svm.mat','w','b');
%% 6. Report a brief summary of the approach in recog_summary.m
run('recog_summary.m');