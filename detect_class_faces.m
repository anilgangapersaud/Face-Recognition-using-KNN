% run the detector on class.jpg
class = im2single(imread('class.jpg'));
imshow(class);
hold on;
bboxes = zeros(0,4);
confidences = zeros(0,1);
plots = [];
for scale=1:-0.05:0.05
   im_scaled = imresize(class, scale);
   feats = vl_hog(im_scaled,cellSize);
   [rows,cols,~] = size(feats);
   confs = zeros(rows,cols);
   for r=1:rows-5
       for c=1:cols-5
           x = feats(r:r + 5, c:c + 5, :);
           confidence = dot(x(:),w) + b;
           confs(r,c) = confidence;
       end
   end
    [~,inds] = sort(confs(:),'descend');
    if (inds > 30)
        inds = inds(1:30);
    end
    for n=1:numel(inds)
        [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
        conf = confs(row,col);
        if (conf > 1.5)
            bbox = [col*cellSize*(1/scale)...
                row*cellSize*(1/scale)...
                (col+cellSize-1)*cellSize*(1/scale)...
                (row+cellSize-1)*cellSize*(1/scale)];
            dontPlot = 0;
            bbox_length = size(bboxes,1);
            deleteIndices = [];
            if (bbox_length > 0)
                % Non-Max Suppression
                for y=1:bbox_length
                    bbox2 = bboxes(y,:);
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
                             score2 = confidences(y);
                             if (score1 > score2)
                                % delete the old box, this score is better
                                delete(plots(y));
                                plots(y) = [];
                                confidences(y) = [];
                                bboxes(y,:) = [];
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
                confidences = [confidences; conf];
                bboxes = [bboxes; bbox];
            end
        end
    end
end
