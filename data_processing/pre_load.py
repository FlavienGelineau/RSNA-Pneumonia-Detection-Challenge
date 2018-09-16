from skimage.transform import resize
from skimage import exposure
import os
import pydicom
from PIL import Image

INPUT_TRAIN = '../data/input/stage_1_train_images'
INPUT_TEST =  '../data/input/stage_1_test_images'
OUTPUT_TRAIN = '../data/preprocessed_input/stage_1_train_images'
OUTPUT_TEST = '../data/preprocessed_input/stage_1_test_images'


def equalize(dicom):
    img = dicom.pixel_array
    print(img)
    img = exposure.equalize_adapthist(img)*255
    print(img.astype("uint8"))
    dicom.PixelData = img.astype("uint8").tobytes()
    #Dirty fix of a bug from pydicom guys
    dicom[(0x7fe0,0x0010)].is_undefined_length = False
    return dicom 

if __name__ == '__main__':
    for img_name in os.listdir(INPUT_TRAIN):
        dicom = pydicom.dcmread(os.path.join(INPUT_TRAIN, img_name))
        dicom = equalize(dicom)
        pydicom.filewriter.write_file(os.path.join(OUTPUT_TRAIN, img_name), dicom, write_like_original=True)

    for img_name in os.listdir(INPUT_TEST):
        dicom = pydicom.dcmread(os.path.join(INPUT_TEST, img_name))
        dicom = equalize(dicom)
        pydicom.filewriter.write_file(os.path.join(OUTPUT_TEST, img_name), dicom, write_like_original=True)


