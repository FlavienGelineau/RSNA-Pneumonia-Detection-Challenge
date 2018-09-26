from skimage import measure
import numpy as np
import pandas as pd

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
        height = y2 - y
        width = x2 - x
        # proxy for confidence score
        conf = np.mean(pred[y:y + height, x:x + width])
        # add to predictionString
        prediction_string += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
    return prediction_string


def compute_optimal_threshold(filename):
    filename_formatted = filename.split('.bmp')[0]
    metadata = pd.read_csv('../meta_data_processing/test_metadata.csv')
    data = metadata[metadata['filenames']==filename_formatted]
    if data['ViewPosition'].values[0]=='AP':
        return 0.45
    else:
        return 0.5

compute_optimal_threshold("13752451-7571-4b2a-8f3e-b71a6b6e91a2.dcm")