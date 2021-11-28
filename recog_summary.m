%% Report of SVM Classifier
% Student Name: Anil Gangapersaud
% Student Number: 215178411

fprintf("--------------------------------\n");
fprintf("SVM Classier Performance Report\n");
fprintf("--------------------------------\n");
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, validationLabels);
fprintf("--------------------------------\n");
fprintf("SVM Classier Parameters\n");
fprintf("--------------------------------\n");
fprintf("Dataset Size = %d\n", 13786);
fprintf("Positive Data Size = %d\n", 6743);
fprintf("Negative Data Size = %d\n", 6743);
fprintf("Training Data Size = %d\n", 10786);
fprintf("Validation Data Size = %d\n", 2700);
fprintf("Lambda = %d\n", lambda);
fprintf("Cell Size for HOG = %d\n", cellSize);
fprintf("--------------------------------\n");
fprintf("Analysis\n");
fprintf("--------------------------------\n");
fprintf("I find that using a smaller lambda yields a higher accuracy on the validation set.\n");


