from data.environment import Environment
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

""" Script controlling the training of softmax model
    ensure the label_scheme and data_dir are set correctly below
"""
label_scheme = 1
data_dir = 'C:/out2'

env = Environment()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(data_dir,.95, .025, .025, label_scheme=label_scheme)

# for (images, labels) in env.single_distortion_data_generator(train_list, flatten=True):
#     print(images.shape)
print(len(train_list))
print(len(dev_list))
print(len(test_list))

# https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970
if label_scheme == 0:
    num_classes = 7
elif label_scheme == 1:
    num_classes = 3
else:
    print("Wrong label_scheme, currently 0 or 1")
input_dim = 224*224*3
output_dim = num_classes
batch_size = 256
train_steps = 630
val_steps = 58
test_steps = 58
nb_epoch = 10

model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

optimizer="RMSProp"
# optimizer="sgd"
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# checkpoint
filepath="..\models\softmax-"+optimizer+"-{epoch:02d}-"+str(batch_size)+"-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=5)
callbacks_list = [checkpoint]

history = model.fit_generator(env.single_distortion_data_generator(train_list, data_dir, batch_size=batch_size, flatten=True, label_scheme=label_scheme),
                              steps_per_epoch=train_steps,
                              epochs=nb_epoch,
                              verbose=1,
                              validation_data=env.single_distortion_data_generator(dev_list, data_dir, batch_size=batch_size, flatten=True, label_scheme=label_scheme),
                              validation_steps=val_steps,
                              callbacks=callbacks_list)
# model.save("models\softmax."+str(nb_epoch)+"."+str(batch_size)+".h5")
score = model.evaluate_generator(env.single_distortion_data_generator(test_list, data_dir, batch_size=batch_size, flatten=True, label_scheme=label_scheme), steps=test_steps)
print('Test score:', score[0])
print('Test accuracy:', score[1])