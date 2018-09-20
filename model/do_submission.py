import os

from check_validation_results import check_model_on_val
from data_processing.generator_definition import generator
from data_processing.loading_data import load_filenames, load_pneumonia_locations
from model.cnn_segmentation import create_network, iou_bce_loss, mean_iou

from skimage.transform import resize
from skimage import measure
import numpy as np
import pandas as pd

from model.pred_construction import compute_pred_with_mask
from paths import OUTPUT_TEST, OUTPUT_TRAIN, INPUT_TRAIN, INPUT_TRAIN_MODEL, INPUT_TEST_MODEL


def make_submission(model,
                    test_gen,
                    test_filenames,
                    submission_name='submission.csv'):
    submission_dict = {}
    # loop through testset
    for imgs, filenames in test_gen:
        print(imgs)
        # predict batch of images
        preds = model.predict(imgs)
        # loop through batch
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (1024, 1024), mode='reflect')
            prediction_string = compute_pred_with_mask(pred)

            # add filename and prediction_string to dictionary
            filename = filename.split('.')[0]
            submission_dict[filename] = prediction_string
        # stop if we've got them all
        if len(submission_dict) >= len(test_filenames):
            break

    print("Done predicting...")

    # save dictionary as csv file
    sub = pd.DataFrame.from_dict(submission_dict, orient='index')
    sub.index.names = ['patientId']
    sub.columns = ['PredictionString']
    sub.to_csv(submission_name)



if __name__ == '__main__':
    BATCH_SIZE = 32
    IMAGE_SIZE = 320

    test_filenames = os.listdir(OUTPUT_TEST)

    model = create_network(input_size=IMAGE_SIZE, channels=32, n_blocks=2, depth=4)
    model.compile(optimizer='adam',
                  loss=iou_bce_loss,
                  metrics=['accuracy', mean_iou])
    model.load_weights('weights/weights-improvement-01-0.43782.hdf5')

    train_filenames, valid_filenames = load_filenames(n_valid_samples=2560, folder = INPUT_TRAIN_MODEL)
    pneumonia_locations = load_pneumonia_locations()

    valid_gen = generator(INPUT_TRAIN_MODEL, valid_filenames, pneumonia_locations, batch_size=BATCH_SIZE,
                          image_size=IMAGE_SIZE, shuffle=False, predict=False)

    train_gen = generator(INPUT_TRAIN_MODEL, train_filenames, pneumonia_locations, batch_size=BATCH_SIZE,
                          image_size=IMAGE_SIZE, shuffle=False, predict=False)

    check_model_on_val(train_gen, model)

    test_gen = generator(INPUT_TEST_MODEL,
                         filenames=test_filenames,
                         pneumonia_locations=None,
                         batch_size=BATCH_SIZE,
                         image_size=IMAGE_SIZE,
                         shuffle=False,
                         predict=True)

    # create submission
    make_submission(model,
                    test_gen,
                    test_filenames,
                    submission_name='submission.csv')
