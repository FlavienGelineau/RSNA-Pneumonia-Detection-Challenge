from skimage.transform import resize
from skimage import measure
import numpy as np
import pandas as pd


def make_submission(model,
                    test_gen,
                    test_filenames,
                    submission_name='submission.csv'):
    submission_dict = {}
    # loop through testset
    for imgs, filenames in test_gen:
        # predict batch of images
        preds = model.predict(imgs)
        # loop through batch
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (1024, 1024), mode='reflect')
            # threshold predicted mask
            comp = pred[:, :, 0] > 0.5
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            predictionString = ''
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                # proxy for confidence score
                conf = np.mean(pred[y:y + height, x:x + width])
                # add to predictionString
                predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
            # add filename and predictionString to dictionary
            filename = filename.split('.')[0]
            submission_dict[filename] = predictionString
        # stop if we've got them all
        if len(submission_dict) >= len(test_filenames):
            break

    print("Done predicting...")

    # save dictionary as csv file
    sub = pd.DataFrame.from_dict(submission_dict, orient='index')
    sub.index.names = ['patientId']
    sub.columns = ['PredictionString']
    sub.to_csv(submission_name)
