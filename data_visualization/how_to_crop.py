from data_processing.loading_data import load_pneumonia_locations
import numpy as np

pneumonia_locs = load_pneumonia_locations()
areas = [elt[2] * elt[3] for _, pneumonia_loc in pneumonia_locs.items() for elt in pneumonia_loc]
x = [elt[3]/elt[2] for _, pneumonia_loc in pneumonia_locs.items() for elt in pneumonia_loc]
print(np.mean(areas))
print(np.min(areas))
print(np.max(areas))

print(np.mean(x))
print(np.min(x))
print(np.max(x))

