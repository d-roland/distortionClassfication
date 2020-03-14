# Folder containing all scripts and modules required to customize datasets

## Generating input data and labels
Two label schemes are currently used by our models, via 2 modules:\
environment-dir.py allows the generation of input data and labels according to 2 label schemes: 7 or 3 classes (for 7 or 3 distortion types).\

## Resizing and labelling datasets
Since our models need input images of resolution 224x224x3 with specific filenames (which include the label), it may be necessary to resize and label external datasets.\
To that extent, 2 scripts are provided, corresponding to our 2 label schemes:\
First label scheme (7 classes): resize_and_label_dataset.py reshapes images and adapts their filename to <image_name>.<label_number>.<image_extension>\
Second label scheme (3 classes): resize_and_label_dataset-2.py reshapes images and adapts their filename to <image_name>.blur.<blur_radius>.<image_extension> or <image_name>.noise.<noise_amount>.<noise_density>.<image_extension>\
For this second script, the blur_radius, noise_amount and noise_density have to be provided via a dedicated txt file.

Usage:\
python resize_and_label_dataset.py -d <INPUT_IMAGE_DIR> -o <OUTPOUT_IMAGE_DIR> -l label -s WIDTH HEIGHT\
python resize_and_label_dataset-2.py -d <INPUT_IMAGE_DIR> -o <OUTPOUT_IMAGE_DIR> -l label -f <TXT_FILE> -s WIDTH HEIGHT

