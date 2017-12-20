% this script is to generate features of stimulus movie frames to compute
% the similarity or distance among them

images = imageDatastore('SpencerMovie_m4v',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% [trainingImages,testImages] = splitEachLabel(images,1,'randomized');
 
imagefiles = dir('./SpencerMovie_m4v/*.png');      
nfiles = length(imagefiles);  % Number of files found

numImages = nfiles;

net = googlenet;

featuresLayer3 = zeros(139316,192,64);
featuresLayer2 = zeros(139316,192,64);
% extract features for each video scene image from the pretrained googlenet
% layer 3
for i=1:numImages
    img = readimage(images, i);
    img = imresize(img,1.2);
    featuresLayer3((i-1)*116+1:i*116,:,:) = activations(net, img, 3);
end

% histogram build up for each image
histFeatures = zeros(numImages, 128);
for i=1:numImages
    img = readimage(images, i);
    img = imresize(img,1.2);
    % extract features by passing image into pretrained googlenet and get 
    % the features from the 138 layer
    feature = sum(sum(activations(net, img, 138),1),2);
    feature = reshape(feature,[1,128]);
    histFeatures(i,:) = feature;
end


% calcuate the euclidean distance of extracted features between images from
% exact features extracted from CNN
distanceMatrix = zeros(1201,1201);
for i=1:1201
    for j=i:1201
        distanceMatrix(i,j) = sum(sum(sum((featuresLayer3((i-1)*116+1:i*116,:,:)-featuresLayer3((j-1)*116+1:j*116,:,:)).^2)))^0.5;
    end
end

% calculate the euclidean distance of summarized features from histogram 
% I used this histogram to build up distances between stimulus movie frames
distanceMatrixHist = zeros(numImages,numImages);
for i=1:numImages
    for j=i:numImages
        distanceMatrixHist(i,j) = norm(histFeatures(i,:)-histFeatures(j,:));
    end
end

