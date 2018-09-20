# empty dictionary
import csv
import os
import random


def load_pneumonia_locations():
    """
    Compute and returns pneumonia locations from label file.

    :return: pneumonia_locations, a dict. keys are filename, values are masks.
    """
    pneumonia_locations = {}
    # load table

    with open(os.path.join('../data/input/stage_1_train_labels.csv'), mode='r') as infile:
        # open reader
        reader = csv.reader(infile)
        # skip header
        next(reader, None)
        # loop through rows
        for rows in reader:
            # retrieve information
            filename = rows[0]
            location = rows[1:5]
            pneumonia = rows[5]
            # if row contains pneumonia add label to dictionary
            # which contains a list of pneumonia locations per filename
            if pneumonia == '1':
                # convert string to float to int
                location = [int(float(i)) for i in location]
                # save pneumonia location in dictionary
                if filename in pneumonia_locations:
                    pneumonia_locations[filename].append(location)
                else:
                    pneumonia_locations[filename] = [location]

    return pneumonia_locations


def load_filenames(n_valid_samples, folder):
    # load and shuffle filenames
    filenames = os.listdir(folder)
    random.shuffle(filenames)
    # split into train and validation filenames
    train_filenames = filenames[n_valid_samples:]
    valid_filenames = filenames[:n_valid_samples]
    print('n train samples', len(train_filenames))
    print('n valid samples', len(valid_filenames))

    return train_filenames, valid_filenames
