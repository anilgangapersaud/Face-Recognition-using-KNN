%% Part 1. K-Nearest Neighbour Classifier
% Student Name: Anil Gangapersaud
% Student Number: 215178411
run('../vlfeat-0.9.21/toolbox/vl_setup');
%% Load Dataset
face_data = dir('cropped_training_images_faces/*.jpg') ;
not_face_data = dir('cropped_training_images_notfaces/*.jpg');
%% Split Data into Groups
trainingData = {10786};
testData = {2700};
trainingLabels = zeros(10786,1);
testLabels = zeros(2700,1);
j=1;
for i=1:5393
    image = im2single(imread(strcat('cropped_training_images_faces/', face_data(i).name)));
    trainingData{j} = image;
    trainingLabels(j) = 1;
    j=j+1;
end
for i=1:5393
    image = im2single(imread(strcat('cropped_training_images_notfaces/', not_face_data(i).name)));
    trainingData{j} = image;
    trainingLabels(j) = 0;
    j=j+1;
end
j=1;
for i=5394:6743
    image = im2single(imread(strcat('cropped_training_images_faces/', face_data(i).name)));
    testData{j} = image;
    testLabels(j) = 1;
    j=j+1;
end
for i=5394:6743
    image = im2single(imread(strcat('cropped_training_images_notfaces/', not_face_data(i).name)));
    testData{j} = image;
    testLabels(j) = 0;
    j=j+1;
end
%% Consider each fold as validation data (Cross Validation Step)
disp("Performing KNN on Fold 1");
validationData = trainingData(1:2157);
trainingSubset = trainingData(2158:10786);
validationLabels = trainingLabels(1:2157);
trainingLabels1 = trainingLabels(2158:10786);
accuracy1 = KNearestHelperGrayscale(trainingSubset, validationData, trainingLabels1, validationLabels);

disp("Performing KNN on Fold 2");
validationData = trainingData(2158:4375);
trainingSubset = [trainingData(1:2157) trainingData(4376:10786)];
validationLabels2 = trainingLabels(2158:4375);
trainingLabels2 = [trainingLabels(1:2157); trainingLabels(4376:10786)];
accuracy2 = KNearestHelperGrayscale(trainingSubset, validationData, trainingLabels2, validationLabels2);

disp("Performing KNN on Fold 3");
validationData = trainingData(4376:6533);
trainingSubset = [trainingData(1:4375) trainingData(6534:10786)];
validationLabels3 = trainingLabels(4376:6533);
trainingLabels3 = [trainingLabels(1:4375); trainingLabels(6534:10786)];
accuracy3 = KNearestHelperGrayscale(trainingSubset, validationData, trainingLabels3, validationLabels3);

disp("Performing KNN on Fold 4");
validationData = trainingData(6534:8691);
trainingSubset = [trainingData(1:6533) trainingData(8692:10786)];
validationLabels4 = trainingLabels(6534:8691);
trainingLabels4 = [trainingLabels(1:6533); trainingLabels(8692:10786)];
accuracy4 = KNearestHelperGrayscale(trainingSubset, validationData, trainingLabels4, validationLabels4);

disp("Performing KNN on Fold 5");
validationData = trainingData(8692:10786);
trainingSubset = trainingData(1:8691);
validationLabels5 = trainingLabels(8692:10786);
trainingLabels5 = trainingLabels(1:8691);
accuracy5 = KNearestHelperGrayscale(trainingSubset, validationData, trainingLabels5, validationLabels5);

% Get the optimal value of K
acc_matrix = [accuracy1 accuracy2 accuracy3 accuracy4 accuracy5];
acc_mean_grayscale = mean(acc_matrix, 2);
[~,optimal_k_grayscale] = max(acc_mean_grayscale);
%% Step One: Classifying Gray Images
disp("Classifying Raw Grayscale Images...");
% init counters
countMatches = 0;
countIterations = 0;
% perform classification on each test image
for i=1:size(testData,2)
    testImage = testData{i};
    countIterations = countIterations + 1;
    % store calculated distances between test image and training images
    distances = zeros(size(trainingData,2),1);
    % calculate euclidean distances
    for j=1:size(trainingData,2)
        distances(j) = sqrt(sum((testImage(:) - trainingData{j}(:)).^2));
    end
    % sort the distance vector and take the first k values
    [distancesSorted, index] = sort(distances);
    kNearestIndices = index(1:optimal_k_grayscale);
    kNearestLabels = trainingLabels(kNearestIndices,:);
    % check if the majority vote matches the label
    if (mode(kNearestLabels) == testLabels(i))
        countMatches = countMatches+1;
    end
end
grayscale_accuracy = countMatches/countIterations;
save('knn_grayscale.mat','grayscale_accuracy','acc_mean_grayscale','optimal_k_grayscale');
%% Load HOG Features
run('get_features.m');
%% Split Data into Groups
trainingData = zeros(10786,1116);
testData = zeros(2700,1116);
trainingLabels = zeros(10786, 1);
testLabels = zeros(2700,1);
j=1;
for i=1:5393
    trainingData(j,:) = pos_feats(i,:);
    trainingLabels(j) = 1;
    j=j+1;
end
for i=1:5393
    trainingData(j,:) = neg_feats(i,:);
    trainingLabels(j) = 0;
    j=j+1;
end
j=1;
for i=5394:6743
    testData(j,:) = pos_feats(i,:);
    testLabels(j) = 1;
    j=j+1;
end
for i=5394:6743
    testData(j,:) = neg_feats(i,:);
    testLabels(j) = 0;
    j=j+1;
end
%% Consider each fold as validation data (Cross Validation Step)
disp("Performing KNN on Fold 1");
validationData = trainingData(1:2157,:);
trainingSubset = trainingData(2158:10786,:);
validationLabels = trainingLabels(1:2157);
trainingLabels1 = trainingLabels(2158:10786);
accuracy1 = KNearestHelperHOG(trainingSubset, validationData, trainingLabels1, validationLabels);

disp("Performing KNN on Fold 2");
validationData = trainingData(2158:4375,:);
trainingSubset = [trainingData((1:2157),:); trainingData((4376:10786),:)];
validationLabels2 = trainingLabels(2158:4375);
trainingLabels2 = [trainingLabels(1:2157); trainingLabels(4376:10786)];
accuracy2 = KNearestHelperHOG(trainingSubset, validationData, trainingLabels2, validationLabels2);

disp("Performing KNN on Fold 3");
validationData = trainingData(4376:6533,:);
trainingSubset = [trainingData((1:4375),:); trainingData((6534:10786),:)];
validationLabels3 = trainingLabels(4376:6533);
trainingLabels3 = [trainingLabels(1:4375); trainingLabels(6534:10786)];
accuracy3 = KNearestHelperHOG(trainingSubset, validationData, trainingLabels3, validationLabels3);

disp("Performing KNN on Fold 4");
validationData = trainingData(6534:8691,:);
trainingSubset = [trainingData((1:6533),:); trainingData((8692:10786),:)];
validationLabels4 = trainingLabels(6534:8691);
trainingLabels4 = [trainingLabels(1:6533); trainingLabels(8692:10786)];
accuracy4 = KNearestHelperHOG(trainingSubset, validationData, trainingLabels4, validationLabels4);

disp("Performing KNN on Fold 5");
validationData = trainingData(8692:10786,:);
trainingSubset = trainingData(1:8691,:);
validationLabels5 = trainingLabels(8692:10786);
trainingLabels5 = trainingLabels(1:8691);
accuracy5 = KNearestHelperHOG(trainingSubset, validationData, trainingLabels5, validationLabels5);

% Get the optimal value of K
acc_matrix = [accuracy1 accuracy2 accuracy3 accuracy4 accuracy5];
acc_mean_hog = mean(acc_matrix, 2);
[~,optimal_k_hog] = max(acc_mean_hog);
%% Classify HOG Test Images
disp("Classifying HOG Images...");
% init counters
countMatches = 0;
countIterations = 0;
% perform classification on each test image
for i=1:size(testData,2)
    testImage = testData(i,:);
    countIterations = countIterations + 1;
    % store calculated distances between test image and training images
    distances = zeros(size(trainingData,2),1);
    % calculate euclidean distances
    for j=1:size(trainingData,2)
        trainingImage = trainingData(j,:);
        distances(j) = sqrt(sum((testImage(:) - trainingImage(:)).^2));
    end
    % sort the distance vector and take the first k values
    [distancesSorted, index] = sort(distances);
    kNearestIndices = index(1:optimal_k_hog);
    kNearestLabels = trainingLabels(kNearestIndices,:);
    % check if the majority vote matches the label
    if (mode(kNearestLabels) == testLabels(i))
        countMatches = countMatches+1;
    end
end
hog_accuracy = countMatches/countIterations;
%% Report Results
load('knn_grayscale.mat');
fprintf("--------------------------\n");
fprintf("KNN Classification Summary\n");
fprintf("--------------------------\n");
fprintf("KNN Accuracy on Grayscale Images: %d. Optimal K Value: %d\n", grayscale_accuracy, optimal_k_grayscale);
fprintf("KNN Accuracy on HOG Images: %d. Optimal K Value: %d\n", hog_accuracy, optimal_k_hog);
figure;
% plot the graph of average k's for 
P1 = subplot(1,2,1);
plot(acc_mean_grayscale, '-');
ylabel(P1, "Average Accuracy");
xlabel(P1, "K");
title(P1, "Grayscale");
P2 = subplot(1,2,2);
plot(acc_mean_hog,'-');
ylabel(P2,"Average Accuracy");
xlabel(P2,"K");
title(P2,"HOG");