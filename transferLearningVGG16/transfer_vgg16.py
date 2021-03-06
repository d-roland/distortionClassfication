from keras import applications, optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import os
from argparse import ArgumentParser
from math import ceil

""" Script controlling the fine tuning of VGG model
    Key inputs are dataset folder and label scheme
    Key setups include Environment module (input data and label generation), model fine tuning and hyperparameters
    Usage: python transfer_vgg16.py --data_dir <DATASET_DIR> --label_scheme <0 for 7 classes or 1 for 3 classes>
"""

# The following 5 imports can be skipped if not running from a Jupiter notebook
# Instead, simply use the following:
# from data.environment import Environment
import importlib.util
spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()

# Key 2 command lines are data_dir and label_scheme (number of classes as per environment2.py)
parser = ArgumentParser()
parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur", type=str,
                    help='directory containing input images and labels')
parser.add_argument('--label_scheme', default=1, type=int,
                    help='number and type of labels')
args = parser.parse_args()

# Generate train, dev and test sets via the generate_train_dev_test_lists generator
train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, .85, .075, .075, label_scheme = args.label_scheme)

# Set number of labels depending on chosen label scheme, to specify Softmax size
if args.label_scheme == 0:
    num_classes = 7
elif args.label_scheme == 1:
    num_classes = 3
else:
    Print("Wrong label_scheme, currently 0 or 1")

# Following implementation of transfer learning follows the guidelines of
# https://riptutorial.com/keras/example/32608/transfer-learning-using-keras-and-vgg
vgg_model = applications.VGG16(weights='imagenet', include_top=True)
vgg_model.summary()

# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 224, 224, 3)       0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

# Getting output tensor of the last VGG layer that we want to include
x = layer_dict['block4_pool'].output

# Stacking a new simple convolutional network on top of it
x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
# x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.
custom_model = Model(input=vgg_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in custom_model.layers[:15]:
    layer.trainable = False

# Select an optimizer and other hyperparameters:
# optimizer='adam'
# optimizerObj = optimizers.Adam(learning_rate=.5)

# optimizer='sgd'
optimizer='RMSProp'
optimizerObj=optimizers.RMSprop(decay=2e-5)

# Do not forget to compile it
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
filepath="/home/jupyter/Env/keras_ve/transfer-learning/VGG/vgg16-d"+str(args.label_scheme)+"-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
filepath2="/home/jupyter/Env/keras_ve/transfer-learning/VGG/vgg16weights-d"+str(args.label_scheme)+"-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=5)
checkpoint2 = ModelCheckpoint(filepath2, verbose=1, save_best_only=False, save_weights_only=True, period=1)
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
