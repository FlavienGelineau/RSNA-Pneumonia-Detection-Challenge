import os

import numpy as np
from PIL import Image

from paths import INPUT_TRAIN, INPUT_TEST, OUTPUT_TRAIN, OUTPUT_TEST

"""The goal here is to segment images, and associated masks, into smaller images that are easier to treat in CNNs."""


def load_image(infilename):
    img = Image.open(infilename)
    return np.asarray(img, dtype="int32")


def segment_one_image_into(image_shape, sliding):
    pass


if __name__ == '__main__':
    paths = [INPUT_TRAIN, INPUT_TEST]
    processed_paths = [OUTPUT_TRAIN, OUTPUT_TEST]

    for path, processed_path in zip(paths, processed_paths):
        print('path ', path)
        img_names = os.listdir(path)
        for i, img_name in enumerate(img_names):
            print('{} out of {}'.format(i, len(img_names)))
            print(load_image('{}/{}.jpg'.format(processed_path, img_name)))
