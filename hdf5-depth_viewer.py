import cv2
import h5py
import numpy as np
from PIL import Image
import argparse
import os

def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img


parser = argparse.ArgumentParser(description='Mid-Air hdf5 use example, also shows how to decode depth maps')
parser.add_argument('--hdf5_path', type=str, help='path to hdf5 file',required=True)

args = parser.parse_args()

if __name__ == '__main__':
    database = h5py.File(args.hdf5_path, "r")
    db_path = os.path.dirname(args.hdf5_path)

    for dataset in database:
        print("Currently displaying : %s" % dataset)
        position_sampling_rate  = database[dataset]["groundtruth"]["position"].attrs["sampling_frequency"]
        print("Position sampling rate [Hz] :  %.2f" % position_sampling_rate)

        for i, (rgb_path) in enumerate(database[dataset]["camera_data"]["color_left"]):
            depth_path = database[dataset]["camera_data"]["depth"][i]
            color_left = cv2.imread(os.path.join(db_path,rgb_path))
            depth = open_float16(os.path.join(db_path,depth_path)) # depth in meters

            # Process depth for displaying purpose
            np.clip(depth, 1, 1250, depth)
            depth = (np.log(depth)-1) / (np.log(1250)-1)

            position = database[dataset]["groundtruth"]["position"][i*4,:]
            print("Position [m] : x %.2f\ty %.2f\tz %.2f" % (position[0], position[1], position[2]))
            cv2.imshow('color_left', color_left)
            cv2.imshow('depth', depth.astype(np.float32))
            cv2.waitKey(1)

    database.close()