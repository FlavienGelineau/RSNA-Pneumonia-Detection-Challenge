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


def aggregate_preds(models):
    pass


def get_models():
    model1 = create_network(input_size=512, channels=24, n_blocks=2, depth=4)
    model1.load_weights('weights/weights-22-0.40075-512-10_public_0118.hdf5')

    model2 = create_network(input_size=320, channels=32, n_blocks=2, depth=4)
    model2.load_weights('weights/weights-improvement-13-0.39958_public_0122.hdf5')

    model3 = create_network(input_size=320, channels=32, n_blocks=2, depth=4)
    model3.load_weights('weights/weights-improvement-18-0.40025_public_0118.hdf5')

    model4 = create_network(input_size=512, channels=24, n_blocks=2, depth=4)
    model4.load_weights('weights/weights-10-0.40313-512-10.hdf5')

    model5 = create_network(input_size=512, channels=24, n_blocks=2, depth=4)
    model5.load_weights('weights/weights-08-0.40680-512-10.hdf5')

    models = [
        [model1, 512, 0.2], [model4, 512, 0.2], [model5, 512, 0.2],
        [model2, 320, 0.2], [model3, 320, 0.2]
    ]
    for model, _, _ in models:
        model.compile(optimizer='adam',
                      loss=iou_bce_loss,
                      metrics=['accuracy', mean_iou])
    return models


def make_submission(models,
                    test_gen_320,
                    test_gen_512,
                    test_filenames,
                    batch_size,
                    submission_name='submission.csv'):
    """Compute submission with precendtly trained neural network. Assert neural network has precedently been trained."""
    submission_dict = {}
    # loop through testset
    for (imgs_320, _), (imgs_512, filenames) in zip(test_gen_320, test_gen_512):
        # predict batch of images
        model, format, weight = models[0]
        if format == 320:
            preds = weight * model.predict(imgs_320)
        if format == 512:
            preds = weight * model.predict(imgs_512)
        preds = np.array([resize(pred, (1024, 1024), mode='reflect') for pred in preds])

        for model, format, weight in models[1:]:
            if format == 320:
                sub_pred = weight * model.predict(imgs_320)
                sub_pred = np.array([resize(pred, (1024, 1024), mode='reflect') for pred in sub_pred])
                preds = preds + sub_pred

            if format == 512:
                sub_pred = weight * model.predict(imgs_512)
                sub_pred = np.array([resize(pred, (1024, 1024), mode='reflect') for pred in sub_pred])
                preds = preds + sub_pred

        # loop through batch
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            # pred = resize(pred, (1024, 1024), mode='reflect')
            prediction_string = compute_pred_with_mask(pred, filename)

            # add filename and prediction_string to dictionary
            filename = filename.split('.')[0]
            submission_dict[filename] = prediction_string
        # stop if we've got them all
        if len(submission_dict) >= len(test_filenames):
            break
        print(len(submission_dict), '/', len(test_filenames))

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

    models = get_models()

    train_filenames, valid_filenames = load_filenames(n_valid_samples=2560, folder=INPUT_TRAIN_MODEL)
    pneumonia_locations = load_pneumonia_locations()

    valid_gen = generator(INPUT_TRAIN_MODEL,
                          valid_filenames,
                          pneumonia_locations,
                          batch_size=BATCH_SIZE,
                          image_size=IMAGE_SIZE,
                          shuffle=False,
                          predict=False)

    check_model_on_val(valid_gen, models[3][0])


    test_gen_320 = generator(INPUT_TEST_MODEL,
                             filenames=test_filenames,
                             pneumonia_locations=None,
                             batch_size=BATCH_SIZE,
                             image_size=320,
                             shuffle=False,
                             predict=True)

    test_gen_512 = generator(INPUT_TEST_MODEL,
                             filenames=test_filenames,
                             pneumonia_locations=None,
                             batch_size=BATCH_SIZE,
                             image_size=512,
                             shuffle=False,
                             predict=True)

    # create submission
    make_submission(models,
                    test_gen_320,
                    test_gen_512,
                    test_filenames,
                    batch_size=BATCH_SIZE,
                    submission_name='submission.csv')
