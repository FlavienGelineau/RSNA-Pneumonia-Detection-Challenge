import os

import gc
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from data_processing.generator_definition import generator
from data_processing.loading_data import load_filenames, load_pneumonia_locations
import numpy as np
from model.cnn_segmentation import create_network, iou_bce_loss, mean_iou
from paths import INPUT_TRAIN, INPUT_TEST, OUTPUT_TRAIN, OUTPUT_TEST, INPUT_TRAIN_MODEL, INPUT_TEST_MODEL

np.random.seed(42)

def get_callbacks(img_size, batch_size):
    early_stop = EarlyStopping(patience=5)

    filepath = "weights/weights-{epoch:02d}-{val_loss:.5f}"+"-{}-{}.hdf5".format(img_size, batch_size)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, verbose=1)

    # cosine learning rate annealing
    def cosine_annealing(x):
        lr = 0.001
        epochs = 20
        return lr * (np.cos(np.pi * x / epochs) + 1.) / 2


    learning_rate = LearningRateScheduler(cosine_annealing)

    return [early_stop, checkpoint, learning_rate]


if __name__ == '__main__':
    BATCH_SIZE = 9
    IMAGE_SIZE = 512
    do_plot_graphs = False

    folder_train = INPUT_TRAIN_MODEL
    folder_test = INPUT_TEST_MODEL

    test_filenames = os.listdir(folder_test)

    model = create_network(input_size=IMAGE_SIZE, channels=24, n_blocks=2, depth=4)
    model.save_weights('weights/initial_weights_{}.hdf5'.format(IMAGE_SIZE))
    model.compile(optimizer='adam',
                  loss=iou_bce_loss,
                  metrics=['accuracy', mean_iou])
    try:
        model.load_weights('weights/weights-improvement-12-0.39942.hdf5')
    except:
        print('model weights couldnt have been loaded')

    # create train and validation generators
    print(model.summary())
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
    del model

    gc.collect()
    model = create_network(input_size=IMAGE_SIZE, channels=32, n_blocks=2, depth=4)
    model.load_weights('weights/initial_weights.hdf5')

    history = model.fit_generator(train_gen,
                                  validation_data=valid_gen,
                                  callbacks=get_callbacks(),
                                  epochs=30000,
                                  shuffle=True)