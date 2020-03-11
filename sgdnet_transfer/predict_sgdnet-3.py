import keras
from keras.models import load_model
from argparse import ArgumentParser
import importlib.util
from math import ceil
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import random

spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment-dir3.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()

parser = ArgumentParser()
parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur", type=str,
                    help='directory containing input images and labels')
parser.add_argument('--label_scheme', default=1, type=int,
                    help='number and type of labels')
args = parser.parse_args()

train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0, 0, 1)
#train_list, dev_list, test_list = env.generate_train_dev_test_lists(args.data_dir, 0.85, 0.075, 0.075)


custom_model = load_model('sgdnet50-d2-RMSProp-05-256-0.8366-0.54.h5', custom_objects={'BatchNorm':keras.layers.BatchNormalization}) 


batch_size = 256
test_steps = ceil(len(test_list)/batch_size)
y_true = []
y_pred = []
inputs = []
input_names = []

valid_generator = env.single_distortion_data_generator(test_list, args.data_dir, batch_size=batch_size, flatten=False, batch_name="test", steps=test_steps, label_scheme = args.label_scheme)

for step in range(test_steps):
    x,y,z = next(valid_generator)
    pred = custom_model.predict(x)
    inputs.append(x)
    y_true.append(y)
    input_names.append(z)
    y_pred.append(pred)

inputs = np.array(inputs, dtype=np.uint8)
inputs = inputs.reshape((batch_size*test_steps,224,224,3))
    
y_true = np.array(y_true, dtype=np.float32)
y_true = y_true.argmax(axis=2)
y_true = y_true.reshape((batch_size*test_steps))

y_pred = np.array(y_pred, dtype=np.float32)
y_pred = y_pred.argmax(axis=2)
y_pred = y_pred.reshape((batch_size*test_steps))

input_names = np.array(input_names, dtype=np.unicode_)
input_names = input_names.reshape((batch_size*test_steps))

matrix = confusion_matrix(y_true, y_pred)

print(matrix)

class_names = range(max(y_true.max()+1,y_pred.max()+1))
df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names, )

fig = plt.figure(figsize=(10,7))

try:
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')

tp_c=np.zeros((y_true.max()+1,len(y_true)))
fp_c=np.zeros((y_true.max()+1,len(y_true)))
tn_c=np.zeros((y_true.max()+1,len(y_true)))
fn_c=np.zeros((y_true.max()+1,len(y_true)))

for i in range(y_true.max()+1):
    tp_c[i,] = (y_true==i)*(y_pred==i)
    fp_c[i,] = (y_true==i)*(y_pred!=i)
    tn_c[i,] = (y_true!=i)*(y_pred!=i)
    fn_c[i,] = (y_true!=i)*(y_pred==i)

print("\nAnalysis of False negatives:")
fn_c0 = np.argwhere(fn_c[0,]==1)
print("Total number of images:",len(fn_c0))
choice = random.randint(0,len(fn_c0))
sample = int(fn_c0[choice])
print("Index of sampled image:",sample)
print("Label of sampled image:",y_true[sample])
print("Prediction for sampled image:",y_pred[sample])
print("Name of sampled image:",input_names[sample])
img = Image.fromarray(inputs[sample,], mode = 'RGB')
display(img)

print("\nZoom on specific case:")
case_01 = (y_true==0)*(y_pred==1)
index_01 = np.argwhere(case_01==1)
print("Label of sampled image:",y_true[int(index_01[0])])
print("Prediction for sampled image:", y_pred[int(index_01[0])])
print("Name of sampled image:",input_names[int(index_01[0])])
img = Image.fromarray(inputs[int(index_01[0]),], mode = 'RGB')
display(img)
