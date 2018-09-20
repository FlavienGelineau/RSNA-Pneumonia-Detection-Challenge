import os
from joblib import Parallel, delayed
import pydicom
from PIL import Image
from skimage import exposure
import scipy.misc

from paths import INPUT_TRAIN, INPUT_TEST, OUTPUT_TRAIN, OUTPUT_TEST


def equalize_and_convert(dicom):
    img = dicom.pixel_array
    img = exposure.equalize_adapthist(img) * 255
    img = img.astype("uint8")

    return img


def equalize(i, img_name):
    # print('{} out of {}'.format(i, len(img_names)))
    progress("Equalizing", i, len(img_names))
    dicom = pydicom.dcmread(os.path.join(path, img_name))
    img_array = equalize_and_convert(dicom)
    data = Image.fromarray(img_array)
    data.convert("L").save(os.path.join(processed_path, img_name + ".bmp"))


if __name__ == '__main__':
    paths = [INPUT_TRAIN, INPUT_TEST]
    processed_paths = [OUTPUT_TRAIN, OUTPUT_TEST]

    for path, processed_path in zip(paths, processed_paths):
        print('path ', path)
        img_names = os.listdir(path)
        Parallel(n_jobs=-1)(delayed(equalize)(i, img_name) for i, img_name in enumerate(img_names))
