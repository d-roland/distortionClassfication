from data.environment2 import Environment2
from keras.models import load_model
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # turn off gpu training

env = Environment2()
train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)

model = load_model('C:\models\softmax-d2-l1-RMSProp-07-256-0.4499-9997.60.l.h5')

batch_size = 256 # runs out of memory at 512 batch_size
train_steps = 630
val_steps = 58
test_steps = 58
nb_epoch = 10
label_scheme = 1


score = model.evaluate_generator(env.single_distortion_data_generator(test_list, batch_size=batch_size, flatten=True, batch_name="test", steps=test_steps, label_scheme=label_scheme),
                                        steps=test_steps#len(test_list)/batch_size
                                        )
print('Test score:', score[0])
print('Test accuracy:', score[1])