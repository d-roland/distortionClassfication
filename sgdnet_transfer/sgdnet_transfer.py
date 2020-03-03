from __future__ import division

import importlib.util
spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

#from data.environment import Environment

import cv2,keras
from keras import applications, optimizers
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential
import sys,os
import numpy as np

from utilitiestrain import preprocess_imagesandsaliencyforiqa,preprocess_label
import time

from modelfinal import TVdist,SGDNet
import keras.backend as K
import tensorflow as tf
import h5py, yaml
import math
from argparse import ArgumentParser
from scipy import stats
import json


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

env = foo.Environment()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)


# https://riptutorial.com/keras/example/32608/transfer-learning-using-keras-and-vgg

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

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)
print('phase: ' + args.phase)
print('database: ' + args.database)
print('base_model: ' + args.basemodel)
print('saliency information: ' + args.saliency)
print('CA: ' + str(args.CA))
config.update(config[args.database])

crop_h =  config['shape_r']  
# print (crop_h) #384
crop_w =  config['shape_c']   

sgdnet_model = SGDNet(basemodel = args.basemodel, saliency = args.saliency, CA = args.CA,fixed =args.fixed, img_cols=crop_w, img_rows=crop_h, out2dim=args.out2dim )

arg = 'saliencyoutput-alpha0.25-ss-Koniq10k-1024-EXP0-lr=0.0001-bs=19.33-0.1721-0.0817-0.1637-0.2054.pkl'
weight_file = './checkpoint/'+ arg
sgdnet_model.load_weights(weight_file)

layer_dict = dict([(layer.name, layer) for layer in sgdnet_model.layers])

# Getting output tensor of the last SGDNet layer that we want to include
x = layer_dict['global_average_pooling2d_1'].output

# Stacking a new simple convolutional network on top of it
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(4096, activation='relu')(x)
# x = Dense(4096, activation='relu')(x)
# x = Dropout(0.5)(x)
x = Dense(7, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.
custom_model = Model(input=sgdnet_model.input, output=x)
#custom_model = Model(outputs=x, inputs=sgdnet_model.input)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:177]:
    layer.trainable = False

custom_model.summary()

# extracting model architecture in json format 
sgdnet_custom_json = custom_model.to_json()
with open("sgdnet_custom_json.json", "w") as json_file:
    json.dump(sgdnet_custom_json, json_file)

# Do not forget to compile it
optimizer='adam'
optimizerObj = optimizers.Adam(learning_rate=.5)

# optimizer='sgd'
# optimizer='RMSProp'
# optimizerObj=optimizers.RMSprop(decay=2e-5)

custom_model.compile(loss='categorical_crossentropy',
                     optimizer=optimizerObj,
                     metrics=['accuracy'])
custom_model.summary()

batch_size = 256 # runs out of memory at 512 batch_size
train_steps = 500
val_steps = 70
test_steps = 70
nb_epoch = 10

# checkpoint
filepath="/home/jupyter/Env/keras_ve/transfer-learning/SGDNet/sgdnet50-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
filepath2="/home/jupyter/Env/keras_ve/transfer-learning/SGDNet/sgdnet50weights-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=5)
checkpoint2 = ModelCheckpoint(filepath2, verbose=1, save_best_only=False, save_weights_only=True, period=1)
callbacks_list = [checkpoint, checkpoint2]

# custom_model.save_weights("C:/models/vgg16weights-"+optimizer+"-00-"+str(batch_size)+".h5")
history = custom_model.fit_generator(env.single_distortion_data_generator(train_list, batch_size=batch_size, flatten=False, batch_name="train", steps=train_steps),
                              steps_per_epoch=train_steps,#len(train_list)/batch_size,
                              epochs=nb_epoch,
                              verbose=1,
                              validation_data=env.single_distortion_data_generator(dev_list, batch_size=batch_size, flatten=False, batch_name="dev", steps=val_steps),
                              validation_steps=val_steps,#len(dev_list)/batch_size,
                              callbacks=callbacks_list)

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps),
                                        steps=test_steps#len(test_list)/batch_size
                                        )
print('Test score:', score[0])
print('Test accuracy:', score[1])
