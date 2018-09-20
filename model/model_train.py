import os

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from data_processing.generator_definition import generator
from data_processing.loading_data import load_filenames, load_pneumonia_locations
import numpy as np
from model.cnn_segmentation import create_network, iou_bce_loss, mean_iou
from paths import INPUT_TRAIN, INPUT_TEST, OUTPUT_TRAIN, OUTPUT_TEST, INPUT_TRAIN_MODEL, INPUT_TEST_MODEL

np.random.seed(42)

def get_callbacks():
    early_stop = EarlyStopping(patience=5)

    filepath = "weights/weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1)

    # cosine learning rate annealing
    def cosine_annealing(x):
        lr = 0.001
        epochs = 20
        return lr * (np.cos(np.pi * x / epochs) + 1.) / 2


    learning_rate = LearningRateScheduler(cosine_annealing)

    return [early_stop, checkpoint, learning_rate]


if __name__ == '__main__':
    BATCH_SIZE = 16
    IMAGE_SIZE = 320
    do_plot_graphs = False

    folder_train = INPUT_TRAIN_MODEL
    folder_test = INPUT_TEST_MODEL

    test_filenames = os.listdir(folder_test)

    model = create_network(input_size=IMAGE_SIZE, channels=32, n_blocks=2, depth=4)
    model.compile(optimizer='adam',
                  loss=iou_bce_loss,
                  metrics=['accuracy', mean_iou])
    try:
        model.load_weights('weights/weights-improvement-01-0.97.hdf5')
    except:
        print('model weights couldnt have been loaded')

    # create train and validation generators

    train_filenames, valid_filenames = load_filenames(n_valid_samples=2560, folder = folder_train)
    pneumonia_locations = load_pneumonia_locations()

    train_gen = generator(folder_train, train_filenames, pneumonia_locations, batch_size=BATCH_SIZE,
                          image_size=IMAGE_SIZE, shuffle=True, augment=True, predict=False)
    valid_gen = generator(folder_train, valid_filenames, pneumonia_locations, batch_size=BATCH_SIZE,
                          image_size=IMAGE_SIZE, shuffle=False, predict=False)



    keras.backend.get_session().run(tf.global_variables_initializer())

    history = model.fit_generator(train_gen,
                                  validation_data=valid_gen,
                                  callbacks=get_callbacks(),
                                  epochs=30000,
                                  shuffle=True)

