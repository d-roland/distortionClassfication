# Distortion Classfication

## Data Generation
The data was generated using `java/DistortImage/src/image/DistortImage.java`.  This process downloaded images from ImageNet, sized down the image to 224x224 pixels, and applied a series of six distortions on each image.

Original Image: 

![Image 1](sampleImages/00000007.0.organism.jpg)

Gaussian (Smooth) Blur:

![Image 1](sampleImages/00000007.1.organism.jpg)

Motion Blur:

![Image 1](sampleImages/00000007.2.organism.jpg)

Non-Monochrome Gaussian Noise:

![Image 1](sampleImages/00000007.3.organism.jpg)

Monochrome Gaussian Noise:

![Image 1](sampleImages/00000007.4.organism.jpg)

Marble:

![Image 1](sampleImages/00000007.5.organism.jpg)

Twirl:

![Image 1](sampleImages/00000007.7.organism.jpg)

## Basline Model
We trained a softmax regression model as a baseline using `softmaxRegression/softmax.py`.

## Transfer Learning Model
We fine tuned VGG16 model using `transferLearning/vgg16.py`.

## Saved Models
Saved models are in `models/`--only the ones under the GitHub 100MB file size limit.