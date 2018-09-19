import os
from joblib import Parallel, delayed
import pydicom
from PIL import Image
from skimage import exposure
import scipy.misc

INPUT_TRAIN = '../data/input/stage_1_train_images'
INPUT_TEST = '../data/input/stage_1_test_images'
OUTPUT_TRAIN = '../data/preprocessed_input/stage_1_train_images'
OUTPUT_TEST = '../data/preprocessed_input/stage_1_test_images'


def equalize_and_convert(dicom):
    img = dicom.pixel_array
    img = exposure.equalize_adapthist(img) * 255
    img = img.astype("uint8")

    return img


def equalize(i, img_name, path, processed_path):
    print('{} out of {}'.format(i, len(img_names)))
    dicom = pydicom.dcmread(os.path.join(path, img_name))
    img_array = equalize_and_convert(dicom)
    data = Image.fromarray(img_array)
    data.convert("L").save(os.path.join(processed_path, img_name + ".bmp"))


if __name__ == '__main__':
    paths = [INPUT_TEST]
    processed_paths = [OUTPUT_TEST]

    for path, processed_path in zip(paths, processed_paths):
        print('path ', path)
        img_names = os.listdir(path)
        Parallel(n_jobs=8)(delayed(equalize)(i, img_name, path, processed_path) for i, img_name in enumerate(img_names))
