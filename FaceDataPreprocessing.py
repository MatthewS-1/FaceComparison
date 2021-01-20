import numpy as np
from PIL import Image
import os
import random
import itertools

data = []


# This part will have to be adjusted accordingly based on the file location of the dataset
# NOTE: This is not my dataset; dataset can be downloaded at http://vision.ucsd.edu/content/yale-face-database
def normalize(np_arr):  # normalize the data in the interval [-0.5, 0.5]
    return np_arr / 255 - 0.5


def reverse_normalize(np_arr):  # reverse the normalization; used when taking model output
    return (np_arr + 0.5) * 255


def process(image):  # process an image into an np array that the NN can easily use
    image = image.crop((0, 12, 168, 180))
    image = image.resize((128, 128))
    # change image size to a cube so it's easier to work with
    pixel_data = np.asarray(image)
    pixel_data = normalize(pixel_data)
    pixel_data = pixel_data.reshape(128, 128, 1)
    return pixel_data


def gather():  # gathers the data in the CroppedYale folder
    global data
    for file in os.listdir("CroppedYale"):
        person = []
        for dir in os.listdir("CroppedYale\\" + file):
            if not dir[-4:] == ".pgm" or "Ambient" in dir:  # ensures we are reading proper gray-scaled photos
                pass
            else:
                image = Image.open("CroppedYale\\" + file + "\\" + dir)
                person.append(process(image))
        data.append(person)


def std_filter(std_range=(1, None)):
    flattened_array = list(itertools.chain(*data))
    mean = np.mean(flattened_array)
    std = np.std(flattened_array)

    for person in range(len(data)):
        for i, image in enumerate(data[person]):
            img_mean = np.mean(image)

            if std_range[0]:
                if not mean - (std_range[0] * std) < img_mean:
                    """print(np.array(image).shape)
                    img = Image.fromarray(reverse_normalize(image)[:,:,0])
                    img.show()
                    break"""
                    data[person].pop(i)
                    i -= 1
            elif std_range[1]:
                if not img_mean < mean + (std_range[1] * std):
                    data[person].pop(i)
                    i -= 1


def init(num_data=15000, guaranteed_true=None, repeats=False, filters=[], params=[]):  # initializes the data
    global data

    in_1 = []
    in_2 = []
    out_y = []

    if not data:
        gather()

    params += []*(len(filters) - len(params))  # pad params
    for func, param in zip(filters, params):
        func(*param)

    if not guaranteed_true:  # makes it so about half of the data will be true
        guaranteed_true = (num_data // 2) - int(num_data / len(data))

    def get_random(index) -> np.array([]):  # gets a random array of pixel_data from an index
        return data[index][random.randrange(len(data[index]))]

    for _ in range(num_data - guaranteed_true): # randomly selects two images
        pick_1 = random.randrange(len(data))
        pick_2 = random.randrange(len(data))
        not_same = int(not pick_1 == pick_2)

        in_1.append(get_random(pick_1))
        in_2.append(get_random(pick_2))
        out_y.append(not_same)

    for _ in range(guaranteed_true): # adds two faces that are guaranteed to be the same person
        pick = random.randrange(len(data))
        in_1.append(get_random(pick))
        in_2.append(get_random(pick))
        out_y.append(0)

    if not repeats: # removes repeated items in array
        seen = {}
        for i, (arr_1, arr_2) in enumerate(zip(in_1, in_2)):
            hashed = hash(str(np.concatenate((arr_1, arr_2), axis=0).data))
            if not hashed in seen:
                seen[hashed] = True
            else:
                in_1.pop(i), in_2.pop(i), out_y.pop(i)
                i -= 1

    in_1, in_2, out_y = np.array(in_1), np.array(in_2), np.array(out_y)
    shuffle_permutation = np.random.permutation(len(in_1))
    in_1, in_2, out_y = in_1[shuffle_permutation], in_2[shuffle_permutation], out_y[shuffle_permutation]

    return in_1, in_2, out_y


if __name__ == '__main__':  # test the data to ensure it's correctly labeled
    a, b, c = init(filters=[std_filter], params=[[(0.5, None)]])
    """a = reverse_normalize(a)
    b = reverse_normalize(b)
    print(a.shape, b.shape, c.shape, a[0, :, :, 0].shape)
    img = Image.fromarray(a[-1, :, :, 0])
    img.show()
    img = Image.fromarray(b[-1, :, :, 0])
    img.show()
    print(c)"""
