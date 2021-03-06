import os
import imageio
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

class Environment(Sequence):
    # usage:
    # env = Environment()
    # train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)
    def generate_train_dev_test_lists(self, DATA_DIR, ratio_train, ratio_dev, ratio_test, fix_seed=True, label_scheme=0):
        list = os.listdir(DATA_DIR)
        list.sort()
        filtered_list = []
        if label_scheme==0:
            for f in list:
                filename_parts = f.split(".")
                # skip images with two distortions, or "Ripple" distortion #6 since "Ripple" causes image size to change
                if (len(f) == 0 or len(filename_parts[1]) > 1 or filename_parts[1] == "6"):
                    continue
                filtered_list.append(f)
        elif label_scheme==1:
            filtered_list = list
        else:
            print("Wront label scheme: currently 0 or 1")

        if fix_seed:
            np.random.seed(1) # let the results be the same

        np.random.shuffle(filtered_list)

        train_pct = float(ratio_train) / float(ratio_train+ratio_dev+ratio_test)
        dev_pct = float(ratio_dev) / float(ratio_train+ratio_dev+ratio_test)

        train_index = int(train_pct*len(filtered_list))
        dev_index = int((dev_pct+train_pct)*len(filtered_list))

        return filtered_list[:train_index], filtered_list[train_index:dev_index], filtered_list[dev_index:]

    # usage:
    # for (images, labels) in env.single_distortion_data_generator(train_list):
    def single_distortion_data_generator(self, list, DATA_DIR, batch_size=512, flatten=False, batch_name="", steps=0, label_scheme=0):
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
                if label_scheme == 0:
                    filename_parts = f.split(".")
                    num_labels = 7
                    label = filename_parts[1]
                    if label == "7":
                        label = "6"
                elif label_scheme == 1:
                    distortion_type = ""
                    distortion_param1 = ""
                    distortion_param2 = ""
                    filename_parts = f.split("-")
                    if len(filename_parts) > 1:
                        distortion_type = filename_parts[1]
                        distortion_param1 = filename_parts[2]
                        distortion_param2 = filename_parts[3]
                    num_labels = 3
                    label = 0
                    if distortion_type == "blur":
                        label = 1
                    if distortion_type == "noise":
                        label = 2
                else:
                    print("Wront label scheme: currently 0 or 1")
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
