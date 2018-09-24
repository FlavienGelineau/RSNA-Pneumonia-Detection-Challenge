import pandas as pd

metadata = pd.read_csv('train_metadata.csv')

metadata_aggreged = metadata.groupby('patientId')['x'].mean()
print(metadata_aggreged)