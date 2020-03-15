from data.environment import Environment
from keras.models import load_model
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

""" Script controlling the evaluation of softmax model
    ensure the label_scheme, data_dir, and model_path are set correctly below
"""
label_scheme = 1
data_dir = 'C:/out2'
model_path = 'C:\models\softmax-d2-l1-RMSProp-07-256-0.4499-9997.60.l.h5'

env = Environment()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(data_dir,.95, .025, .025, label_scheme=label_scheme)
model = load_model(model_path)

batch_size = 256
train_steps = 630
val_steps = 58
test_steps = 58
nb_epoch = 10

score = model.evaluate_generator(env.single_distortion_data_generator(test_list, data_dir, batch_size=batch_size, flatten=True, batch_name="test", steps=test_steps, label_scheme=label_scheme),
                                        steps=test_steps#len(test_list)/batch_size
                                        )
print('Test score:', score[0])
print('Test accuracy:', score[1])