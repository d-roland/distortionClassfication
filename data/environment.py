import os
import imageio

DATA_DIR = "C:/out/"

class Environment:
    def get_single_distortion_data(self):
        list = os.listdir(DATA_DIR)
        list.sort()
        for f in list:
            filename_parts = f.split(".")
            # skip images with two distortions
            if(len(filename_parts[1])>1):
                continue
            im = imageio.imread(os.path.join(DATA_DIR, f))
            print(im.shape)
            print(f)
env = Environment()
env.get_single_distortion_data()
