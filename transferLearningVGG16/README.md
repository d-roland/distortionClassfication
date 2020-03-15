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
The fine tuning of VGG base model is controlled via transfer_vgg16.py (with 7 classes or  with 3 classes)\
Usage:\
python3 transfer_vgg16.py --data_dir <DATASET_DIR> --label_scheme <0 for 7 classes or 1 for 3 classes>

## Evaluation of transfered models
The evaluation can be run via evaluate_vgg16.py (both for model with 7 classes or model with 3 classes)\
Usage:\
python3 evaluate_vgg16.py --data_dir <DATASET_DIR> --label_scheme <0 for 7 classes or 1 for 3 classes>

## Error analysis on transfered models
The data necessary to perform error analysis can be obtained via error_analysis_vgg16.py (both for model with 7 classes or model with 3 classes)\
Usage:\
python3 error_analysis_vgg16.py --data_dir <DATASET_DIR> --label_scheme <0 for 7 classes or 1 for 3 classes>
