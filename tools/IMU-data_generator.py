import h5py
import numpy as np
from pyquaternion import Quaternion
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Generate new IMU measurements for all trajectories in the given hdf5 file')
parser.add_argument('--hdf5_path', type=str, help='path to hdf5 file',required=True)

args = parser.parse_args()

if __name__ == '__main__':

    answer = str(input("Warning: this script will overwrite IMU measurements stored in the given hdf5 dataset. \n"+ \
                       "Do you want to proceed? (y/n): "))
    if not(answer=="y" or answer=="Y"):
        sys.exit(0)

    database = h5py.File(args.hdf5_path, "a")
    db_path = os.path.dirname(args.hdf5_path)

    # IMU noise parameters chosen randomly in a range of values encountered in real devices
    noise_acc = 2 * np.power(10., -np.random.uniform(low=1., high=3., size=(1, 3)))
    noise_gyr = np.power(10., -np.random.uniform(low=1., high=3., size=(1, 3)))
    imu_bias_acc_rw = 2 * np.power(10., -np.random.uniform(low=3., high=6., size=(1, 3)))
    imu_bias_gyr_rw = np.power(10., -np.random.uniform(low=4., high=6., size=(1, 3)))

    for dataset in database:
        print("Currently processing : %s" % dataset)
        gt_group = database[dataset]["groundtruth"]
        gt_attitude = gt_group["attitude"]
        gt_angular_vel = gt_group["angular_velocity"]
        gt_accelerations = gt_group["acceleration"]

        imu_group = database[dataset]["imu"]

        # Set init parameters
        imu_accelerometer = np.zeros(gt_attitude.shape, dtype=float)
        imu_gyroscope = np.zeros(gt_attitude.shape, dtype=float)

        imu_bias_acc = np.random.normal([0., 0., 0.], imu_bias_acc_rw)
        imu_bias_gyr = np.random.normal([0., 0., 0.], imu_bias_gyr_rw)

        init_bias_est_acc = imu_bias_acc + np.random.normal([0., 0., 0.], noise_acc / 50)
        init_bias_est_gyr = imu_bias_gyr + np.random.normal([0., 0., 0.], noise_gyr / 50)
        
        imu_group["accelerometer"].attrs["init_bias_est"] = init_bias_est_acc
        imu_group["gyroscope"].attrs["init_bias_est"] = init_bias_est_gyr

        # Pass over trajectory to generate simulated sensor measurements
        for i in range(gt_attitude.shape[0]):
            attitude = Quaternion(gt_attitude[i, :])
            imu_accelerometer = attitude.conjugate.rotate(gt_accelerations[i, :] + np.array([0., 0., -9.81])) \
                                                            + imu_bias_acc + np.random.normal([0., 0., 0.], noise_acc)
            imu_gyroscope = gt_angular_vel[i, :] + imu_bias_gyr + np.random.normal([0., 0., 0.], noise_gyr)
            imu_bias_acc += np.random.normal([0., 0., 0.], imu_bias_acc_rw)
            imu_bias_gyr += np.random.normal([0., 0., 0.], imu_bias_gyr_rw)
            imu_group["accelerometer"][i] = imu_accelerometer
            imu_group["gyroscope"][i] = imu_gyroscope

database.close()
