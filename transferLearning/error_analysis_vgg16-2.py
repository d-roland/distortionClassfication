import keras
from keras.models import load_model
from argparse import ArgumentParser
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" This script computes predictions for VGG model version 2
    and provides all relevant data for error analysis: confusion matrix, precision, recall and F1 score
"""

# The following 7 imports can be skipped if not running from a Jupiter notebook
# Instead, simply use the following:
# from data.environment import Environment
import importlib.util
spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment-dir2.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Key 2 command lines are data_dir and label_scheme (number of classes as per environment2.py)
parser = ArgumentParser()
parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur", type=str,
                    help='directory containing input images and labels')
parser.add_argument('--label_scheme', default=1, type=int,
                    help='number and type of labels')
args = parser.parse_args()

# Generate train, dev and test sets via the generate_train_dev_test_lists generator:
# If analyzing performance on LIVE dataset, set 100% to test_list (since dataset is small)
# If analyzing performance on original dataset, keep original split (eg 0.075 for test_list)

#train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0, 0, 1)
train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, .85, .075, .075)

# Load relevant model for analysis:
custom_model = load_model('vgg16-d2-l1-RMSProp-10-256-0.8516-0.79.f.h5', custom_objects={'BatchNorm':keras.layers.BatchNormalization}) 

# Set batch size and steps number
batch_size = 256
test_steps = ceil(len(test_list)/batch_size)

# Compute predictions on test set
valid_generator = env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps, label_scheme = args.label_scheme)

y_true = []
y_pred = []

for step in range(test_steps):
    x,y = next(valid_generator)
    pred = custom_model.predict(x)
    y_pred.append(pred)
    y_true.append(y)

# Reshape data for analysis
y_true = np.array(y_true, dtype=np.float32)
y_true = y_true.argmax(axis=2)
y_true = y_true.reshape((batch_size*test_steps))

y_pred = np.array(y_pred, dtype=np.float32)
y_pred = y_pred.argmax(axis=2)
y_pred = y_pred.reshape((batch_size*test_steps))

# Compute and print confusion matrix on both basic and enhanced ways
matrix = confusion_matrix(y_true, y_pred)
print(matrix)

class_names = range(max(y_true.max()+1,y_pred.max()+1))
df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names, )

fig = plt.figure(figsize=(10,7))

try:
    sns.set(font_scale=1.8)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=20)
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Compute and print performance metrics

precision, recall, fscore, support = score(y_true, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
