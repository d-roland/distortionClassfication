from data.environment import Environment
from keras import applications, optimizers
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

env = Environment()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)

# Creating new model. Please note that this is NOT a Sequential() model.
# custom_model = load_model("C:/models/vgg16-RMSProp-05-256-0.99-0.25.c.h5")
# custom_model = load_model("C:/models/vgg16-adam-05-128-0.9892-0.00.d.h5")
# flatten = False
custom_model = load_model("models/softmax-20-1024-0.24.h5")
flatten = True

# # Do not forget to compile it
# # optimizer='adam'
# # optimizerObj = optimizers.Adam(learning_rate=.5)
#
# # optimizer='sgd'
# optimizer='RMSProp'
# optimizerObj=optimizers.RMSprop(decay=2e-5)
#
# custom_model.compile(loss='categorical_crossentropy',
#                      optimizer=optimizerObj,
#                      metrics=['accuracy'])
# custom_model.summary()

batch_size = 256 # runs out of memory at 512 batch_size
train_steps = 630
val_steps = 70
test_steps = 70
nb_epoch = 20

# checkpoint
# filepath="C:/models/vgg16-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
# filepath2="C:/models/vgg16weights-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.4f}-{val_loss:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=5)
# checkpoint2 = ModelCheckpoint(filepath2, verbose=1, save_best_only=False, save_weights_only=True, period=1)
# callbacks_list = [checkpoint, checkpoint2]

# custom_model.save_weights("C:/models/vgg16weights-"+optimizer+"-00-"+str(batch_size)+".h5")
# history = custom_model.fit_generator(env.single_distortion_data_generator(train_list, batch_size=batch_size, flatten=False, batch_name="train", steps=train_steps),
#                               steps_per_epoch=train_steps,#len(train_list)/batch_size,
#                               epochs=nb_epoch,
#                               verbose=1,
#                               validation_data=env.single_distortion_data_generator(dev_list, batch_size=batch_size, flatten=False, batch_name="dev", steps=val_steps),
#                               validation_steps=val_steps,#len(dev_list)/batch_size,
#                               callbacks=callbacks_list)

score = custom_model.evaluate_generator(env.single_distortion_data_generator(dev_list, batch_size=batch_size, flatten=flatten, batch_name="dev", steps=val_steps),
                                        steps=val_steps#len(test_list)/batch_size
                                        )
print('Validation score:', score[0])
print('Validation accuracy:', score[1])

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, batch_size=batch_size, flatten=flatten, batch_name="test", steps=test_steps),
                                        steps=test_steps#len(test_list)/batch_size
                                        )
print('Test score:', score[0])
print('Test accuracy:', score[1])