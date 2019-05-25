import h5py
import argparse
import os

parser = argparse.ArgumentParser(description='Mid-Air hdf5 unix to windows image paths converter')
parser.add_argument('--hdf5_path', type=str, help='path to hdf5 file',required=True)

args = parser.parse_args()

if __name__ == '__main__':
    database = h5py.File(args.hdf5_path, "a")
    db_path = os.path.dirname(args.hdf5_path)

    for dataset in database:
        print("Processing %s" % dataset)
        for type in database[dataset]["camera_data"]:
            for i, (unix_path) in enumerate(database[dataset]["camera_data"][type]):
                win_path = unix_path.replace("/","\\")
                database[dataset]["camera_data"][type][i] = win_path.encode('utf-8')

    database.close()