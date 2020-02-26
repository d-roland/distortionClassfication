import keras
from keras.models import load_model
from argparse import ArgumentParser
import importlib.util

spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment-dir.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()

parser = ArgumentParser()
parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur", type=str,
                    help='directory containing input images and labels')
args = parser.parse_args()

train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0, 0, 1)

custom_model = load_model('vgg16-RMSProp-05-256-0.99-0.25.c.h5') 

batch_size = 128
test_steps = 5

score = custom_model.evaluate_generator(env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps), steps=test_steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
