import keras
from keras.models import load_model
from argparse import ArgumentParser
from math import ceil

""" Evaluator of VGG transfered model (based on 8 or 3 classes)
    Outputs loss and accuracy on specified test set
    Usage: python3 evaluate_vgg16.py --data_dir <DATASET_DIR> --label_scheme <0 for 8 classes or 1 for 3 classes>
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
# If evaluating LIVE dataset, set 100% to test_list (since dataset is small)
# If evaluating original dataset, keep original split (eg 0.075 for test_list)

train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0, 0, 1, label_scheme = args.label_scheme)
#train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0.85, 0.075, 0.075, label_scheme = args.label_scheme)

# Load relevant model for evaluation

# custom_model = load_model('vgg16-d0-adam-05-128-0.9892-0.00.d.h5') 
# custom_model = load_model('vgg16-d0-RMSProp-05-256-0.99-0.25.c.h5') 
# custom_model = load_model('vgg16-d1-l1-RMSProp-06-256-0.8408-0.36.i.h5') 
custom_model = load_model('vgg16scratch-d1-l1-RMSProp-04-128-0.8410-0.34.k.h5') 

# Set batch size and steps number
batch_size = 128
test_steps = ceil(len(test_list)/batch_size)

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps, label_scheme = args.label_scheme), steps=test_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
