# Transfer learning from VGG16

Source : VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. 
```
@inproceedings {Simonyan2015,
	year = {2015},
	booktitle = {International Conference on Learning Representations (ICLR)},
	title = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
	author = {Karen Simonyan and Andrew Zisserman}
}
```

## Transfer learning
The fine tuning of VGG base model is controlled via transfer_vgg16.py (version 1 with 8 classes) or transfer_vgg16-2.py (version 2 with 3 classes)

## Evaluation of transfered models
The evaluation can be run via evaluate_vgg16.py (for model version 1) and evaluate_vgg16-2.py (for model version 2)

## Error analysis on transfered models
The data necessary to perform error analysis can be obtained via error_analysis_vgg16.py (for model version 1) and error_analysis_vgg16-2.py (for model version 2)
