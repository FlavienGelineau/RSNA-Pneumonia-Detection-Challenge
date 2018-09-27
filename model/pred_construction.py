from skimage import measure
import numpy as np
import pandas as pd


def is_bounding_box_ok(y, x, y2, x2):
    if abs(y - y2) * abs(x - x2) <= 2320:
        return False
    if abs(y - y2) / abs(x - x2) <= 0.2292:
        return False

    return True


def compute_pred_with_mask(pred, filename):
    threshold = compute_optimal_threshold(filename)
    comp = pred[:, :, 0] > threshold
    # apply connected components
    comp = measure.label(comp)
    # apply bounding boxes
    prediction_string = ''
    for region in measure.regionprops(comp):
        # retrieve x, y, height and width
        y, x, y2, x2 = region.bbox
        if is_bounding_box_ok(y, x, y2, x2):
            height = y2 - y
            width = x2 - x
            # proxy for confidence score
            conf = np.mean(pred[y:y + height, x:x + width])
            # add to predictionString
            prediction_string += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        else:
            print('pred not taken ! ')
    return prediction_string


def compute_optimal_threshold(filename):
    filename_formatted = filename.split('.bmp')[0]
    metadata = pd.read_csv('../meta_data_processing/test_metadata.csv')
    data = metadata[metadata['filenames'] == filename_formatted]
    if data['ViewPosition'].values[0] == 'AP':
        return 0.4
    else:
        return 0.5


if __name__ == '__main__':
    compute_optimal_threshold("13752451-7571-4b2a-8f3e-b71a6b6e91a2.dcm")
