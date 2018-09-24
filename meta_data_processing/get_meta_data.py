import gc
import pandas as pd
import pydicom
import numpy as np


def compute_metadata_file(df_meta, filename='train_metadata.csv'):
    dcm_columns = None
    for n, pid in enumerate(df_meta['patientId'].unique()):
        if n % 100 == 0:
            print(n, len(df_meta['patientId'].unique().tolist()))
        dcm_file = '../data/input/stage_1_train_images/%s.dcm' % pid
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


if __name__ == '__main__':
    detailed_class_info = pd.read_csv('../data/input/stage_1_detailed_class_info.csv')
    train_labels = pd.read_csv('../data/input/stage_1_train_labels.csv')
    df_train_meta = pd.merge(left=detailed_class_info, right=train_labels, how='left', on='patientId')

    df_test_meta = pd.read_csv('../data/input/stage_1_detailed_class_info.csv')

