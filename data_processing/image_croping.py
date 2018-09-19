import os

import numpy as np
import pydicom
import scipy.misc
from PIL import Image

INPUT_TRAIN = '../data/input/stage_1_train_images'
INPUT_TEST = '../data/input/stage_1_test_images'
OUTPUT_TRAIN = '../data/preprocessed_input/stage_1_train_images'
OUTPUT_TEST = '../data/preprocessed_input/stage_1_test_images'


def load_image(infilename):
    img = Image.open(infilename)
    return np.asarray(img, dtype="int32")


if __name__ == '__main__':
    paths = [INPUT_TRAIN, INPUT_TEST]
    processed_paths = [OUTPUT_TRAIN, OUTPUT_TEST]

    for path, processed_path in zip(paths, processed_paths):
        print('path ', path)
        img_names = os.listdir(path)
        for i, img_name in enumerate(img_names):
            print('{} out of {}'.format(i, len(img_names)))
            print(load_image('{}/{}.jpg'.format(processed_path, img_name)))
