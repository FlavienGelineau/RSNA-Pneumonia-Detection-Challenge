import gc
import pandas as pd
import pydicom
import numpy as np
import os

def compute_metadata_file(df_meta, input_file = 'stage_1_train_images', filename='train_metadata.csv'):
    file_names = os.listdir('../data/input/{}'.format(input_file))
    file_names = [name for name in file_names if '.bmp' not in name]
    dcm_columns = None
    for n, pid in enumerate(file_names):
        if n % 100 == 0:
            print(n, len(file_names))
        dcm_file = '../data/input/{}/%s'.format(input_file) % pid
        dcm_data = pydicom.read_file(dcm_file)

        if not dcm_columns:
            dcm_columns = dcm_data.dir()
            dcm_columns.remove('PixelSpacing')
            dcm_columns.remove('PixelData')

        for col in dcm_columns:
            if not (col in df_meta.columns):
                df_meta[col] = np.nan
            index = df_meta[df_meta['patientId'] == pid].index
            df_meta.loc[index, col] = dcm_data.data_element(col).value

        del dcm_data

    gc.collect()

    df_meta.to_csv(filename)


def compute_metadata_test_file(input_file = 'stage_1_test_images', filename='test_metadata.csv'):
    file_names = os.listdir('../data/input/{}'.format(input_file))
    file_names = [name for name in file_names if '.bmp' not in name]
    view_pos = []
    age = []
    sex = []
    dcm_columns = None
    test_metadata = pd.DataFrame()
    test_metadata['filenames'] = file_names

    for n, pid in enumerate(file_names):
        if n % 100 == 0:
            print(n, len(file_names))
        dcm_file = '../data/input/{}/%s'.format(input_file) % pid
        dcm_data = pydicom.read_file(dcm_file)

        if not dcm_columns:
            dcm_columns = dcm_data.dir()
            dcm_columns.remove('PixelSpacing')
            dcm_columns.remove('PixelData')

        view_pos.append(dcm_data.data_element('ViewPosition').value)
        age.append(dcm_data.data_element('PatientAge').value)
        sex.append(dcm_data.data_element('PatientSex').value)

        del dcm_data

    gc.collect()

    test_metadata['ViewPosition'] = view_pos
    test_metadata['PatientAge'] = age
    test_metadata['PatientSex'] = sex
    test_metadata.to_csv(filename)


if __name__ == '__main__':
    detailed_class_info = pd.read_csv('../data/input/stage_1_detailed_class_info.csv')
    train_labels = pd.read_csv('../data/input/stage_1_train_labels.csv')
    df_train_meta = pd.merge(left=detailed_class_info, right=train_labels, how='left', on='patientId')

    df_test_meta = pd.read_csv('../data/input/stage_1_detailed_class_info.csv')

    #compute_metadata_file(df_test_meta, input_file='stage_1_test_images', filename='test_metadata.csv')
    compute_metadata_test_file()