import os
import imageio
import numpy as np

DATA_DIR = "C:/out/"

class Environment:
    def get_single_distortion_data(self):
        images = None;
        list = os.listdir(DATA_DIR)
        list.sort()
        for f in list:
            filename_parts = f.split(".")
            # skip images with two distortions
            if(len(filename_parts[1])>1 or filename_parts[1] == "6"):
                continue
            im = imageio.imread(os.path.join(DATA_DIR, f))
            if(images is None):
                images = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
            else:
                images = np.append(images, np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2])), axis=0)
            print(f)
        print(images.shape)
env = Environment()
env.get_single_distortion_data()
