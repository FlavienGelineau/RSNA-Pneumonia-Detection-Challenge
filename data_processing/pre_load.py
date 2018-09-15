from skimage.transform import resize
from skimage import exposure
import os
import pydicom
from PIL import Image

INPUT_TRAIN = '../data/input/stage_1_train_images'
INPUT_TEST =  '../data/input/stage_1_test_images'
OUTPUT_TRAIN = '../data/preprocessed_input/stage_1_train_images'
OUTPUT_TEST = '../data/preprocessed_input/stage_1_test_images'

if __name__ == '__main__':
    for img_name in os.listdir(INPUT_TRAIN):
        dicom = pydicom.dcmread(os.path.join(INPUT_TRAIN, img_name))
        img = dicom.pixel_array
        print(img.mean())
        img = exposure.equalize_adapthist(img)
        print(img.mean())
        data = Image.fromarray(img*256)
        data.convert("L").save(os.path.join(OUTPUT_TRAIN, img_name + ".bmp"))

    for img_name in os.listdir(INPUT_TEST):
        dicom = pydicom.dcmread(os.path.join(INPUT_TRAIN, img_name)).pixel_array
        img = dicom.pixel_array
        img = exposure.equalize_adapthist(img)
        data = Image.fromarray(img*256)
        data.convert("L").save(os.path.join(OUTPUT_TEST, img_name + ".bmp"))

