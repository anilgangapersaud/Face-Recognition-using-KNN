%% Assignment 3 - Master Script
% Student Name: Anil Gangapersaud
% Student Number: 2151781411

%% README
% The K-Nearest Neighbour Classifier (Part 1) takes a considerable amount of time to
% run since I am cross validation across 5 folds. 

%% Part 1 - K-Nearest Neighbour Classifier
run('kNearestNeighbourClassifier.m');
disp("Finished Part 1.");
disp("Press any key to continue.");
pause;

%% Part 2 - Face Detection
run('faceDetection.m');
disp("Finished Part 2.");
disp("Press any key to continue.");
pause;

%% Part 3 - Multi-Scale Face Detection
run('detect.m');
disp("Finished Part 3a.");
disp("Press any key to continue.");
pause;
% Part 3 - Run the Detector on class.jpg
run('detect_class_faces.m');
disp("Finished Part 3b.");
disp("Press any key to exit.");
pause;