%% 1. K-Nearest Neighbour Classifier
% Student Name: Anil Gangapersaud
% Student Number: 215178411
run('vlfeat-0.9.21/toolbox/vl_setup');
%% Load Datasets
disp("Loading Datasets...");
face_data = dir('cropped_training_images_faces/*.jpg') ;
not_face_data = dir('cropped_training_images_notfaces/*.jpg');
disp("Loaded Datasets!");
%% Split Data into Training and Testing sets
disp("Splitting datasets into training and test sets...");
trainingData = {10786};
testData = {2700};
trainingLabels = zeros(10786,1);
testLabels = zeros(2700,1);
j=1;
% store 5393 face images in training data
for i=1:5393
    image = im2single(imread(strcat('cropped_training_images_faces/', face_data(i).name)));
    trainingData{j} = image;
    trainingLabels(j) = 1;
    j=j+1;
end
% store 5393 non-face images in training data
for i=1:5393
    image = im2single(imread(strcat('cropped_training_images_notfaces/', not_face_data(i).name)));
    trainingData{j} = image;
    trainingLabels(j) = 0;
    j=j+1;
end
j=1;
% store 6743-5394 face images in test data
for i=5394:6743
    image = im2single(imread(strcat('cropped_training_images_faces/', face_data(i).name)));
    testData{j} = image;
    testLabels(j) = 1;
    j=j+1;
end
% store 6743-5394 non-face images in test data
for i=5394:6743
    image = im2single(imread(strcat('cropped_training_images_notfaces/', not_face_data(i).name)));
    testData{j} = image;
    testLabels(j) = 0;
    j=j+1;
end
disp("Finished splitting data!");
%% Classify Raw Greyscale Test Images
disp("Classifying raw grayscale images...");
tic
% define range of k values to run the classifier on
kmin = 1;
kmax = 1;

% store the accuracy for each k in a vector
accuracy = zeros(kmax,1);

% perform classification on each k value
for k=kmin:kmax
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
        kNearestIndices = index(1:k);
        kNearestLabels = trainingLabels(kNearestIndices,:);
        % check if the majority vote matches the label
        if (mode(kNearestLabels) == testLabels(i))
            countMatches = countMatches+1;
        end
    end
    accuracy(k) = countMatches/countIterations;
end

% Plot accuracy points on a graph
plot(accuracy,'-');
disp("Finished classifying raw grayscale images!");
toc
disp("Press any key to continue");
pause;
%% Classify HOG Test Images
disp("Classifying HOG images...");
tic
% define range of k values to run the classifier on
kmin = 1;
kmax = 20;

% store the accuracy for each k in a vector
accuracy = zeros(kmax,1);

% perform classification on each k value
for k=kmin:kmax
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
            trainingImage = trainingData{j};
            I1 = vl_hog(testImage, 1);
            I2 = vl_hog(trainingImage, 1);
            distances(j) = sqrt(sum((I1(:) - I2(:)).^2));
        end
        % sort the distance vector and take the first k values
        [distancesSorted, index] = sort(distances);
        kNearestIndices = index(1:k);
        kNearestLabels = trainingLabels(kNearestIndices,:);
        % check if the majority vote matches the label
        if (mode(kNearestLabels) == testLabels(i))
            countMatches = countMatches+1;
        end
    end
    accuracy(k) = countMatches/countIterations;
end
% Draw graph
plot(accuracy,'-');
toc
disp("Finished classifying HOG images!");
pause;