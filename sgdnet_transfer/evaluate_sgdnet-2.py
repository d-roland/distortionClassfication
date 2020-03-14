import keras
from keras.models import load_model
from argparse import ArgumentParser
from math import ceil

# The following 5 imports can be skipped if not running from a Jupiter notebook
# Instead, simply use the following:
# from data.environment import Environment
import importlib.util
spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment-dir2.py")
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
# If evaluating LIVE dataset, set 100% to test_list (since dataset is small)
# If evaluating original dataset, keep original split (eg 0.075 for test_list)

train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0, 0, 1)
#train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0.85, 0.075, 0.075)

# Load relevant model for evaluation

#custom_model = load_model('sgdnet50-d3-l1-adam-02-128-0.6962-0.79.h5', custom_objects={'BatchNorm':keras.layers.BatchNormalization}) 
custom_model = load_model('sgdnet50-d2-RMSProp-05-256-0.8366-0.54.h5', custom_objects={'BatchNorm':keras.layers.BatchNormalization})

batch_size = 256
test_steps = ceil(len(test_list)/batch_size)

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps, label_scheme = args.label_scheme), steps=test_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])