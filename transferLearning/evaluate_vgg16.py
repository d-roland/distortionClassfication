import keras
from keras.models import load_model
from argparse import ArgumentParser
from math import ceil

""" Evaluator of VGG model 1 (based on 8 labels and dataset v1)
    Outputs loss and accuracy on specified test set
"""

# The following 5 imports can be skipped if not running from a Jupiter notebook
# Instead, simply use the following:
# from data.environment import Environment
import importlib.util
spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment-dir.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()

# Key 2 command lines are data_dir and label_scheme (number of classes as per environment2.py)
parser = ArgumentParser()
parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur", type=str,
                    help='directory containing input images and labels')
args = parser.parse_args()

# Generate train, dev and test sets via the generate_train_dev_test_lists generator
# If evaluating LIVE dataset, set 100% to test_list (since dataset is small)
# If evaluating original dataset, keep original split (eg 0.075 for test_list)

#train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, .95, .025, .025)
train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0, 0, 1)

custom_model = load_model('vgg16-adam-05-128-0.9892-0.00.d.h5') 
#custom_model = load_model('vgg16-d1-RMSProp-05-256-0.99-0.25.c.h5') 

batch_size = 128
test_steps = ceil(len(test_list)/batch_size)

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps), steps=test_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
