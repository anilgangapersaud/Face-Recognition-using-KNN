run('vlfeat-0.9.21/toolbox/vl_setup')
imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

cellSize = 6;
dim = 36;
for i=1:nImages
    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    imshow(im);
    hold on;
    
    filtered_bboxes = zeros(0,4);
    local_confidences = zeros(0,1);
    image_name = {imageList(i).name};
    plots = [];
    % make predictions at multiple scales
    for scale=1:-0.05:0.05
        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        im_scaled = imresize(im,scale);
        
        feats = vl_hog(im_scaled,cellSize);

        % concatenate the features into 6x6 bins, and classify them (as if they
        % represent 36x36-pixel faces)
        [rows,cols,~] = size(feats);    
        confs = zeros(rows,cols);
        for r=1:rows-5
            for c=1:cols-5
                % create feature vector for the current window and classify it using the SVM model, 
                x = feats(r:r + 5, c:c + 5, :);
                % take dot product between feature vector and w and add b,
                confidence = dot(x(:),w) + b;
                % store the result in the matrix of confidence scores confs(r,c)
                confs(r,c) = confidence;
            end
        end

        % get the most confident predictions 
        [~,inds] = sort(confs(:),'descend');
        if (inds > 50) 
            inds = inds(1:50); % (use a bigger number for better recall)
        end

        for n=1:numel(inds)        
            [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
            conf = confs(row,col);
            if (conf > 1) 
                bbox = [ col*cellSize*(1/scale) ...
                         row*cellSize*(1/scale) ...
                        (col+cellSize-1)*cellSize*(1/scale) ...
                        (row+cellSize-1)*cellSize*(1/scale)];
                dontPlot = 0;
                bbox_length = size(filtered_bboxes,1);
                if (bbox_length > 0)
                    % Non-Max Suppression
                    for y=1:bbox_length
                        bbox2 = filtered_bboxes(y,:);
                        bi=[max(bbox(1),bbox2(1)) ; max(bbox(2),bbox2(2)) ; min(bbox(3),bbox2(3)) ; min(bbox(4),bbox2(4))];
                        iw=bi(3)-bi(1)+1;
                        ih=bi(4)-bi(2)+1;
                        if iw>0 && ih>0        
                            ua=(bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1)+...
                            (bbox2(3)-bbox2(1)+1)*(bbox2(4)-bbox2(2)+1)-...
                            iw*ih;
                            ov=iw*ih/ua;
                            if (ov > 0)
                                 score1 = conf;
                                 score2 = local_confidences(y);
                                 if (score1 > score2)
                                    % delete the old box, this score is better
                                    delete(plots(y));
                                    plots(y) = [];
                                    local_confidences(y) = [];
                                    filtered_bboxes(y,:) = [];
                                    break;
                                 else 
                                    % keep the old box, this box is not better
                                    dontPlot = 1;
                                    break;
                                 end
                            end
                        end
                    end
                end
                % plot
                if (dontPlot == 0)
                    plot_rectangle = [bbox(1), bbox(2); ...
                       bbox(1), bbox(4); ...
                       bbox(3), bbox(4); ...
                       bbox(3), bbox(2); ...
                       bbox(1), bbox(2)];
                    plots = [plots; plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-')];
                    local_confidences = [local_confidences; conf];
                    filtered_bboxes = [filtered_bboxes; bbox];
                    image_names = [image_names; image_name];
                end
            end
        end
    end
    bboxes = [bboxes; filtered_bboxes];
    confidences = [confidences; local_confidences];
    fprintf('got preds for image %d/%d\n',i,nImages);
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);