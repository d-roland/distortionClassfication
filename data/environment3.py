import os
import imageio
import numpy as np

DATA_DIR = "C:/out3/"

class Environment3:
    # usage:
    # env = Environment()
    # train_list, dev_list, test_list = env.generate_train_dev_test_lists(.95, .025, .025)
    def generate_train_dev_test_lists(self, ratio_train, ratio_dev, ratio_test, fix_seed=True):
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

    # usage:
    # for (images, labels) in env.single_distortion_data_generator(train_list):
    def single_distortion_data_generator(self, list, batch_size=1000, flatten=False, batch_name="", steps=0, label_scheme=0):
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
                distortion_type = filename_parts[1]
                distortion_param1 = filename_parts[2]
                distortion_param2 = filename_parts[3]

                label = 0
                num_labels = 1
                if label_scheme == 0:
                    num_labels = 11
                    if distortion_type == "blur":
                        distortion_param1 = float(distortion_param1)
                        if distortion_param1 >= 1 and distortion_param1 < 3:
                            label = 1
                        if distortion_param1 >= 3 and distortion_param1 < 5:
                            label = 2
                        if distortion_param1 >= 5 and distortion_param1 < 7:
                            label = 3
                        if distortion_param1 >= 7 and distortion_param1 < 9:
                            label = 4
                        if distortion_param1 >= 9 and distortion_param1 <= 11:
                            label = 5
                    elif distortion_type == "noise":
                        product = float(distortion_param1) * float(distortion_param2)
                        if product >= 1 and product < 6:
                            label = 6
                        if product >= 6 and product < 11:
                            label = 7
                        if product >= 11 and product < 16:
                            label = 8
                        if product >= 16 and product < 21:
                            label = 9
                        if product >= 21:
                            label = 10
                if label_scheme == 1:
                    num_labels = 3
                    if distortion_type == "blur":
                        label = 1
                    if distortion_type == "noise":
                        label = 2
                if label_scheme == 2:
                    num_labels = 15
                    if distortion_type == "blur":
                        distortion_param1 = float(distortion_param1)
                        if distortion_param1 >= 1 and distortion_param1 < 3:
                            label = 1
                        if distortion_param1 >= 3 and distortion_param1 < 5:
                            label = 2
                        if distortion_param1 >= 5 and distortion_param1 < 7:
                            label = 3
                        if distortion_param1 >= 7 and distortion_param1 < 9:
                            label = 4
                        if distortion_param1 >= 9 and distortion_param1 <= 11:
                            label = 5
                    elif distortion_type == "noise":
                        distortion_param1 = float(distortion_param1)
                        distortion_param2 = float(distortion_param2)
                        if distortion_param1 >= 1 and distortion_param1 <= 9:
                            if distortion_param2 >= .1 and distortion_param2 <= .4:
                                label = 6
                            if distortion_param2 > .4 and distortion_param2 <= .7:
                                label = 7
                            if distortion_param2 > .7:
                                label = 8
                        if distortion_param1 > 9 and distortion_param1 <= 18:
                            if distortion_param2 >= .1 and distortion_param2 <= .4:
                                label = 9
                            if distortion_param2 > .4 and distortion_param2 <= .7:
                                label = 10
                            if distortion_param2 > .7:
                                label = 11
                        if distortion_param1 > 18:
                            if distortion_param2 >= .1 and distortion_param2 <= .4:
                                label = 12
                            if distortion_param2 > .4 and distortion_param2 <= .7:
                                label = 13
                            if distortion_param2 > .7:
                                label = 14
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
                # images.append(np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2])))
                one_hot_label = np.zeros(num_labels)
                one_hot_label[int(label)] = 1
                labels.append(one_hot_label)
                # print(f+","+str(label))
                if (len(images) >= batch_size):
                    print("yielding "+batch_name+" "+str(np.min(imageNumbers))+" to "+str(np.max(imageNumbers)))
                    yield (np.array(images), np.array(labels))
                    images = []
                    labels = []
                    imageNumbers =[]
                imageNumber += 1
                # print(imageNumber)
