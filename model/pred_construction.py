from skimage import measure
import numpy as np


def compute_pred_with_mask(pred):
    comp = pred[:, :, 0] > 0.5
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
