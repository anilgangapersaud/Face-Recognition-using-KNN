% you might want to have as many negative examples as positive examples
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
n_have = 0;
if ~exist(new_imageDir, 'dir')
    mkdir(new_imageDir);
else 
    n_have = length(dir(sprintf('%s/*.jpg',new_imageDir)));
end

dim = 36;

while n_have < n_want
    % choose a random image from the directory
    r = randsample(nImages, 1);
    imageName = strcat(imageDir, '/',imageList(r,:).name);
    image = imread(imageName);    
    % generate random 36x36 crops from the non-face images
    [h,w] = size(rgb2gray(image));
    rect = [randsample(w-36,1) randsample(h-36,1) 35 35]; 
    crop = imcrop(image, rect);
    imwrite(rgb2gray(crop), strcat(new_imageDir, '/',strcat('notface_crop_', string(n_have),'.jpg')));
    n_have = n_have + 1;
end