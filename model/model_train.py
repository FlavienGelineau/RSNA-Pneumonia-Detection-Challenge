import os
from data_processing.generator_definition import generator
from data_processing.loading_data import load_filenames, load_pneumonia_locations
from model.cnn_segmentation import create_network, iou_bce_loss, mean_iou
import tensorflow as tf

import numpy as np
import keras

import matplotlib.pyplot as plt
import pandas as pd
from skimage.transform import resize
from skimage import measure

BATCH_SIZE = 16
IMAGE_SIZE = 400

model = create_network(input_size=IMAGE_SIZE, channels=32, n_blocks=2, depth=4)
model.compile(optimizer='adam',
              loss=iou_bce_loss,
              metrics=['accuracy', mean_iou])


# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 20
    return lr * (np.cos(np.pi * x / epochs) + 1.) / 2


learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
from keras.callbacks import EarlyStopping, ModelCheckpoint

# create train and validation generators
folder = '../data/input/stage_1_train_images'

train_filenames, valid_filenames = load_filenames(n_valid_samples=2500)
pneumonia_locations = load_pneumonia_locations()

train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                      shuffle=True, augment=True, predict=False)
valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                      shuffle=False, predict=False)

print(model.summary())
keras.backend.get_session().run(tf.global_variables_initializer())

early_stop = EarlyStopping(patience=10)
checkpoint = ModelCheckpoint(filepath='weights_rsna.hdf5')
history = model.fit_generator(train_gen,
                              validation_data=valid_gen,
                              callbacks=[early_stop, checkpoint],
                              epochs=30000,
                              shuffle=True)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(history.epoch, history.history["loss"], label="Train loss")
plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(history.epoch, history.history["mean_iou"], label="Train iou")
plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
plt.legend()
plt.show()

# load and shuffle filenames
folder = '../input/stage_1_test_images'
test_filenames = os.listdir(folder)
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = generator(folder, test_filenames, None, batch_size=16, image_size=IMAGE_SIZE, shuffle=False, predict=True)

# create submission dictionary
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
sub.to_csv('submission.csv')
