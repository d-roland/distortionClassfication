from data.environment import Environment
from keras.models import Sequential
from keras.layers import Dense, Activation
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

env = Environment()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)

# for (images, labels) in env.single_distortion_data_generator(train_list, flatten=True):
#     print(images.shape)
print(len(train_list))
print(len(dev_list))
print(len(test_list))

input_dim = 224*224*3
output_dim = nb_classes = 7
model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
batch_size = 1024
nb_epoch = 5

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(env.single_distortion_data_generator(train_list, batch_size=batch_size, flatten=True),
                              steps_per_epoch=len(train_list)/batch_size,
                              epochs=nb_epoch,
                              verbose=1,
                              validation_data=env.single_distortion_data_generator(dev_list, batch_size=batch_size, flatten=True),
                              validation_steps=len(dev_list)/batch_size)
model.save("models\softmax."+nb_epoch+"."+batch_size+".h5")
score = model.evaluate_generator(env.single_distortion_data_generator(test_list, batch_size=batch_size, flatten=True), steps=len(test_list)/batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])