from data.environment import Environment
from keras.models import load_model
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

""" Script controlling the evaluation of VGG16 scratch model
    ensure the label_scheme, data_dir, and model_path are set correctly below
"""
label_scheme = 1
data_dir = 'C:/out2'
model_path = 'C:/models/vgg16scratch-d3-l1-RMSProp-04-128-0.8410-0.34.k.h5'

env = Environment()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(data_dir, .95, .025, .025, label_scheme=label_scheme)

model = load_model(model_path)

batch_size = 128  # runs out of memory at 512 batch_size
#train_steps =  int(sample_size/batch_size)
train_steps = 630*2
#val_steps = int(train_steps*0.025)
val_steps = 58*2
test_steps = val_steps
nb_epoch = 10
label_scheme = 1


score = model.evaluate_generator(env.single_distortion_data_generator(test_list, data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps, label_scheme=label_scheme),
                                        steps=test_steps#len(test_list)/batch_size
                                        )
print('Test score:', score[0])
print('Test accuracy:', score[1])