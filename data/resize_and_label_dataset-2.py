import numpy as np
import cv2
import os
import sys
import argparse
from PIL import Image

""" Resize any image dataset to specified dimensions and affect label as part of filename.
    Consistent with environment2.py where images have the following naming scheme:        
    00000001-blur-3.3907833—organism.jpg
    00000001-noise-8-0.12586156-organism.jpg
    Usage: python resize_and_label_dataset-2.py -d <INPUT_IMAGE_DIR> -o <OUTPOUT_IMAGE_DIR> -l label -f <TXT_FILE> -s WIDTH HEIGHT
"""

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Directory to look up for images")
parser.add_argument("-o", help="Output directory")
parser.add_argument("-l", help="Label number: 0. Original / 1. Blur / 2. Noise")
parser.add_argument("-f", help="Help file containing distortion levels")
parser.add_argument("-s", nargs=2, type=int, help="Output size")
args = parser.parse_args()

input_dir = os.path.normpath(args.d) if args.d else os.getcwd()
output_dir = os.path.normpath(args.o) if args.o else os.path.join(os.getcwd(), 'resized')
output_size = tuple(args.s) if args.s else (224,224)
output_label = int(args.l) if args.l else 0

if output_label == 1:
    distortion_type = "blur"
elif output_label == 2:
    distortion_type = "noise"
else: 
    distortion_type = "normal"
    
file_name, img_name = {}, {}
if args.f:
    with open(args.f, "r") as f_read:
        for line in f_read:
            line = line.strip() 
            if line:
                filename, imgname, distortion_level = [elt for elt in line.split(" ")]
                file_name[filename] = distortion_level
                img_name[imgname] = distortion_level
    
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file in os.listdir(input_dir):
    infile = os.path.join(input_dir, file)
    if file in file_name:
        outfile = str(os.path.splitext(os.path.basename(infile))[0])+"-"+str(distortion_type)+"-"+str(file_name[file])+"-0"
    elif file in img_name:
        outfile = str(os.path.splitext(os.path.basename(infile))[0])+"-"+str(distortion_type)+"-"+str(img_name[file])+"-0"
    else:
        outfile = str(os.path.splitext(os.path.basename(infile))[0])+"-"+str(distortion_type)+"-0"+"-0"
    extension = os.path.splitext(infile)[1]
    if extension in ('.jpeg', '.jpg', '.bmp'):
        img = cv2.imread(infile , cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        c = None if len(img.shape) < 3 else img.shape[2]
        if h == w: 
            img2 = cv2.resize(img, (output_size[0],output_size[1]))
        else:
            dif = h if h > w else w
            interpolation = cv2.INTER_AREA if dif > (output_size[0]+output_size[1])//2 else cv2.INTER_CUBIC
            x_pos = (dif - w)//2
            y_pos = (dif - h)//2
            if len(img.shape) == 2:
                mask = np.zeros((dif, dif), dtype=img.dtype)
                mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
            else:
                mask = np.zeros((dif, dif, c), dtype=img.dtype)
                mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
            img2 = cv2.resize(mask, (output_size[0],output_size[1]), interpolation)
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        img2.save(os.path.join(output_dir, outfile+extension))
