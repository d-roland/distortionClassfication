import os
import imageio
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

class Environment(Sequence):
    # usage:
    # env = Environment()
    # train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)
    def generate_train_dev_test_lists(self, DATA_DIR, ratio_train, ratio_dev, ratio_test, fix_seed=True):
        list = os.listdir(DATA_DIR)
        list.sort()
        filtered_list = list

        if fix_seed:
            np.random.seed(1) # let the results be the same

        np.random.shuffle(filtered_list)

        train_pct = float(ratio_train) / float(ratio_train+ratio_dev+ratio_test)
        dev_pct = float(ratio_dev) / float(ratio_train+ratio_dev+ratio_test)

        train_index = int(train_pct*len(filtered_list))
        dev_index = int((dev_pct+train_pct)*len(filtered_list))

        return filtered_list[:train_index], filtered_list[train_index:dev_index], filtered_list[dev_index:]

    
    # 00000001-blur-3.3907833—organism.jpg
    # 3.3907833 is the blur radius.  This is a float between 1.0 and 11.0
    # 00000001 / blur / 3.3907833 / organism.jpg
    
    # 00000001-noise-8-0.12586156-organism.jpg
    # 8 is the noise amount.  This is an int between 1 and 26 inclusive.
    # 0.12586156 is the noise density.  This is a float between .1 and 1.1.  Note, everything over 1.0 is treated as 1.0 which means that every pixel has noise added.
    # 00000001 / noise / 8 / 0.12586156 / organism.jpg

    
    # usage:
    # for (images, labels) in env.single_distortion_data_generator(train_list):
    def single_distortion_data_generator(self, list, DATA_DIR, batch_size=512, flatten=False, batch_name="", steps=0):
        images = []
        labels = []
        imageNumbers = []
        badImages = np.zeros(len(list)) # array to hold 0/1 values, 1 means that imageNumber is not loadable

        repeatMarker = len(list) # this is used to ensure that each epoch holds the same set of images
        if(steps > 0):
            repeatMarker = steps*batch_size
        while True:
            imageNumber = 0
            while imageNumber < repeatMarker and imageNumber < len(list):
                if (badImages[imageNumber] == 1): # save a minute amount of time
                    imageNumber += 1
                    continue
                f = list[imageNumber]
                filename_parts = f.split("-")
                if len(filename_parts) > 1:
                    distortion_type = filename_parts[1]
                    distortion_param1 = filename_parts[2]
                    distortion_param2 = filename_parts[3]

                label = 0
                num_labels = 3
                if distortion_type == "blur":
                    label = 1
                if distortion_type == "noise":
                    label = 2
                try:
                    im = imageio.imread(os.path.join(DATA_DIR, f))
                except ValueError:
                    print("skipping unloadable "+f)
                    if(badImages[imageNumber]==0):
                        badImages[imageNumber]=1
                        repeatMarker+=1
                    imageNumber+=1
                    continue
                if(flatten):
                    im=im.flatten()
                images.append(im)
                imageNumbers.append(imageNumber)
                #images.append(np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2])))
                one_hot_label = np.zeros(num_labels)
                one_hot_label[int(label)] = 1
                labels.append(one_hot_label)
                if (len(images) >= batch_size):
                    print("yielding "+batch_name+" "+str(np.min(imageNumbers))+" to "+str(np.max(imageNumbers)))
                    yield (np.array(images), np.array(labels))
                    images = []
                    labels = []
                    imageNumbers =[]
                imageNumber += 1
                # print(imageNumber)
