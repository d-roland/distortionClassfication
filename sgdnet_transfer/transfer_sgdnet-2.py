# Import all relevant packages and modules for transfer learning
import cv2,keras
import keras.backend as K
import tensorflow as tf
from keras import applications, optimizers
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential
from modelfinal import TVdist,SGDNet
import sys,os
import numpy as np
from utilitiestrain import preprocess_imagesandsaliencyforiqa,preprocess_label
import time
import h5py, yaml
import math
from math import ceil
from argparse import ArgumentParser
from scipy import stats

""" Script controlling the fine tuning of SGDNet model
    Key inputs are dataset folder and label scheme (8 or 3 classes)
    Key setups include model fine tuning and hyperparameters
"""

# The following 6 imports can be skipped if not running from a Jupiter notebook
# Instead, simply use the following:
# from data.environment import Environment
import importlib.util
spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()

# We kept the command line arguments of original SGDNet, but not required to set them
# Key 2 command lines are data_dir and label_scheme (number of classes as per environment2.py)

parser = ArgumentParser(description='PyTorch saliency guided CNNIQA')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='config file path (default: config.yaml)')
parser.add_argument('--database', default='LIVEc', type=str,
                    help='database name (default: LIVEc)')
parser.add_argument('--phase', default='train', type=str,
                    help='phase (default: train')
parser.add_argument('--fixed', action='store_true',
                    help='use fixed backbone (default: False')   
parser.add_argument('--out2dim', type=int, default=1024,
                    help='number of epochs to train (default: 1024)')  
parser.add_argument('--basemodel', default='resnet', type=str,
                    help='resnet or vgg (default: resnet)')      
parser.add_argument('--saliency', default='output', type=str,
                    help='saliency information as input or output or none (default: output)')          
parser.add_argument('--CA', action='store_false',
                    help='use CA? (default: true')   
parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur", type=str,
                    help='directory containing input images and labels')
parser.add_argument('--label_scheme', default=2, type=int,
                    help='number and type of labels')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)
print('phase: ' + args.phase)
print('database: ' + args.database)
print('base_model: ' + args.basemodel)
print('saliency information: ' + args.saliency)
print('CA: ' + str(args.CA))
config.update(config[args.database])

# Generate train, dev and test sets via the generate_train_dev_test_lists generator
train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, .85, .075, .075, label_scheme = args.label_scheme)

# Set number of labels depending on chosen label scheme, to specify Softmax size
if args.label_scheme == 0:
    num_classes = 8
elif args.label_scheme == 1:
    num_classes = 3
else:
    Print("Wrong label_scheme, currently 0 or 1 possible only")

# Keep original setups of SGDNet and load corresponding model
crop_h =  config['shape_r']  
crop_w =  config['shape_c']   

sgdnet_model = SGDNet(basemodel = args.basemodel, saliency = args.saliency, CA = args.CA,fixed =args.fixed, img_cols=crop_w, img_rows=crop_h, out2dim=args.out2dim )

arg = 'saliencyoutput-alpha0.25-ss-Koniq10k-1024-EXP0-lr=0.0001-bs=19.33-0.1721-0.0817-0.1637-0.2054.pkl'
weight_file = './checkpoint/'+ arg
sgdnet_model.load_weights(weight_file)

layer_dict = dict([(layer.name, layer) for layer in sgdnet_model.layers])

# Getting output tensor of the last SGDNet layer that we want to include
x = layer_dict['fc2'].output

# Stacking a new simple convolutional network on top of it
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(4096, activation='relu')(x)
# x = Dense(4096, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.
custom_model = Model(input=sgdnet_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:182]:
    layer.trainable = False

# Print model architecture
custom_model.summary()

# Select an optimizer and other hyperparameters

#optimizer='adam'
#optimizerObj = optimizers.Adam(learning_rate=.05)

# optimizer='sgd'
optimizer='RMSProp'
optimizerObj=optimizers.RMSprop(decay=2e-5)
#optimizerObj=optimizers.RMSprop()

# Do not forget to compile the new model
custom_model.compile(loss='categorical_crossentropy',
                     optimizer=optimizerObj,
                     metrics=['accuracy'])
custom_model.summary()

# Set batch size, steps and epoch numbers
batch_size = 128 # runs out of memory at 512 batch_size
train_steps = ceil(len(train_list)+len(dev_list)+len(test_list))*0.85/batch_size # dataset_size * train_share / batch_size 
val_steps = ceil(len(train_list)+len(dev_list)+len(test_list))*0.075/batch_size
test_steps = ceil(len(train_list)+len(dev_list)+len(test_list))*0.075/batch_size
nb_epoch = 10

# Set checkpoints
filepath="/home/jupyter/Env/keras_ve/transfer-learning/SGDNet/sgdnet50-d"+str(args.label_scheme)+"-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
filepath2="/home/jupyter/Env/keras_ve/transfer-learning/SGDNet/sgdnet50weights-d"+str(args.label_scheme)+"-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=5)
checkpoint2 = ModelCheckpoint(filepath2, verbose=1, save_best_only=False, save_weights_only=True, period=5)
callbacks_list = [checkpoint, checkpoint2]

# custom_model.save_weights("C:/models/vgg16weights-"+optimizer+"-00-"+str(batch_size)+".h5")
history = custom_model.fit_generator(env.single_distortion_data_generator(train_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="train", steps=train_steps, label_scheme = args.label_scheme),
                              steps_per_epoch=train_steps,#len(train_list)/batch_size,
                              epochs=nb_epoch,
                              verbose=1,
                              validation_data=env.single_distortion_data_generator(dev_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="dev", steps=val_steps, label_scheme = args.label_scheme),
                              validation_steps=val_steps,#len(dev_list)/batch_size,
                              callbacks=callbacks_list)

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps, label_scheme = args.label_scheme),
                                        steps=test_steps#len(test_list)/batch_size
                                        )
# Ultimately, print score and accuracy
print('Test score:', score[0])
print('Test accuracy:', score[1])
