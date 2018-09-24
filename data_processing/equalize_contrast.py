import os

import pydicom
import scipy.misc
from joblib import Parallel, delayed
from PIL import Image
from skimage import exposure

from data_processing.utils import progress
from paths import INPUT_TEST, INPUT_TRAIN, OUTPUT_TEST, OUTPUT_TRAIN


def equalize_and_convert(dicom):
    img = dicom.pixel_array
    img_processed = exposure.equalize_adapthist(img) * 255
    img = img.astype("uint8")
    img_processed = img_processed.astype("uint8")

    return img, img_processed


def equalize(i, img_name, path, processed_path, img_names):
    # print('{} out of {}'.format(i, len(img_names)))
    progress("Equalizing", i, len(img_names))
    dicom = pydicom.dcmread(os.path.join(path, img_name))
    img_array, img_processed = equalize_and_convert(dicom)

    data = Image.fromarray(img_array)
    data.convert("L").save(os.path.join(path, img_name + ".bmp"))

    data = Image.fromarray(img_processed)
    data.convert("L").save(os.path.join(processed_path, img_name + ".bmp"))


if __name__ == '__main__':
    paths = [INPUT_TEST]
    processed_paths = [OUTPUT_TEST]

    for path, processed_path in zip(paths, processed_paths):
        print('path ', path)
        img_names = os.listdir(path)
        img_names = [name for name in img_names if '.bmp' not in name]
        Parallel(n_jobs=-1)(delayed(equalize)(i, img_name, path, processed_path, img_names)
                            for i, img_name in enumerate(img_names))
