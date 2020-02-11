# Distortion Classfication

## Data Generation
The data was generated using `java/DistortImage/src/image/DistortImage.java`.

![Image 1]("sampleImages/00000007.0.organism.jpg")

## Basline Model
We trained a softmax regression model as a baseline using `softmaxRegression/softmax.py`.

## Transfer Learning Model
We fine tuned VGG16 model using `transferLearning/vgg16.py`.

## Saved Models
Saved models are in `models/`--only the ones under the GitHub 100MB file size limit.