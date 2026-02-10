import pickle
import os

#load the processed files
#see: ./batch_processing/dump_batch_processing.py to generate the required .pkl file
batch_data_path = f'../data/testICE_30_01_26/batch_testICE_30_01_26.pkl'

if not os.path.isfile(batch_data_path):
    raise FileNotFoundError(f'[load_batch_processing] Batch data file {batch_data_path} not found!')

with open(batch_data_path, 'rb') as handle:
    batch_obj = pickle.load(handle)
