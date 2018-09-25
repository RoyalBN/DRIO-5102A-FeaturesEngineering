
function [r] = evalBVF(nv)

% Load Image Set 
rootFolder='/tmp/101_ObjectCategories'

categories = {'garfield', 'buddha', 'cougar_body','brontosaurus','accordion','airplanes','bass','beaver','cannon','car_side','chair','cellphone','crab','cup','dolphin','electric_guitar'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames')

tbl = countEachLabel(imds)

% Prepare Training and Validation Image Sets
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

% Find the first instance of an image for each category
garfield = find(trainingSet.Labels == 'garfield', 1);
buddha = find(trainingSet.Labels == 'buddha', 1);
cougar_body = find(trainingSet.Labels == 'cougar_body', 1);
brontosaurus = find(trainingSet.Labels == 'brontosaurus', 1);

% figure

subplot(1,4,1);
imshow(readimage(trainingSet,garfield))
subplot(1,4,2);
imshow(readimage(trainingSet,buddha))
subplot(1,4,3);
imshow(readimage(trainingSet,cougar_body))
subplot(1,4,4);
imshow(readimage(trainingSet,brontosaurus))

% Create a Visual Vocabulary and Train an Image Category Classifier
bag = bagOfFeatures(trainingSet, 'VocabularySize', nv);

img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

% Evaluate Classifier Performance
confMatrix = evaluate(categoryClassifier, trainingSet);
confMatrix = evaluate(categoryClassifier, validationSet);




% Compute average accuracy
mean(diag(confMatrix));


% Try the Newly Trained Classifier on Test Images
img = imread(fullfile(rootFolder, 'garfield', 'image_0001.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
categoryClassifier.Labels(labelIdx)


end
