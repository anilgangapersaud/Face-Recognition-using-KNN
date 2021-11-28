function accuracy = KNearestHelperHOG(trainingData,testData,trainingLabels, testLabels)
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
            [~, index] = sort(distances);
            kNearestIndices = index(1:k);
            kNearestLabels = trainingLabels(kNearestIndices,:);
            % check if the majority vote matches the label
            if (mode(kNearestLabels) == testLabels(i))
                countMatches = countMatches+1;
            end
        end
        accuracy(k) = countMatches/countIterations;
    end
end
