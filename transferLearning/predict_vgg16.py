import keras
from keras.models import load_model
from argparse import ArgumentParser
import imageio
import os
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("Environment", "/home/jupyter/Env/keras_ve/transfer-learning/data/environment-dir.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
env = foo.Environment()


def predict_label(file):
    try:
        filename_parts = file.split(".")
        filenames.append(file)
        extension = filename_parts[-1]
        if extension == "bmp":
            format_image = "BMP-PIL"
        elif extension == "jpg" or extension == "jpeg":
            format_image = "JPEG-PIL"
        else:
            print('File format problem with {}'.format(file))
            return
        im = imageio.imread(os.path.join(args.data_dir, file), format = format_image)
        Xnew = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
        Ynew.append(filename_parts[1])
        Ypred.append(custom_model.predict(Xnew))
    except (IOError, SyntaxError) as e:
        print('Bad file:', file)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="/home/jupyter/Env/keras_ve/transfer-learning/data/LIVE_224/gblur/", type=str,
                    help='directory containing test image and label')
    parser.add_argument('--file',
                    help='image to test the model')
    args = parser.parse_args()
    
    custom_model = load_model('vgg16-RMSProp-05-256-0.99-0.25.c.h5') 
    
    batch_size = 128
    test_steps = 1
    Xnew = []
    Ynew = []
    Ypred = []
    filenames = []
    
    if args.file is not None:
        predict_label(args.file)
    else:
        for file in os.listdir(args.data_dir):
            predict_label(file)
            
    for i in range(len(Ynew)):
#        print("Image %s: Predicted_detailed=%s, Predicted_class=%s, Truth=%s" % (filenames[i], np.rint(Ypred[i]), np.argmax(Ypred[i]), Ynew[i]))
        print("Image %s: Predicted_class=%s, Truth=%s" % (filenames[i], np.argmax(Ypred[i]), Ynew[i]))
