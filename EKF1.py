import pandas as pd
import numpy as np
from quaternion import *
import matplotlib.pyplot as plt
import config
import os
from trajectories import *

# multiplying two quaternions: 
# pq == matrix_left(p) @ q == matrix_right(q) @ p
def quaternion_matrix_left(q):
    q0, q1, q2, q3 = q.q
    Q_left = array([
        [q0, -q1, -q2, -q3],
        [q1, q0, -q3, q2],
        [q2, q3, q0, -q1],
        [q3, -q2, q1, q0]
    ])
    return Q_left


def matrix_left(q):
    q0, q1, q2, q3 = q
    Q_left = array([
        [q0, -q1, -q2, -q3],
        [q1, q0, -q3, q2],
        [q2, q3, q0, -q1],
        [q3, -q2, q1, q0]
    ])
    return Q_left


def quaternion_matrix_right(q):
    q0, q1, q2, q3 = q.q
    Q_right = np.array([
        [q0, -q1, -q2, -q3],
        [q1, q0, q3, -q2],
        [q2, -q3, q0, q1],
        [q3, q2, -q1, q0]
    ])
    return Q_right


def matrix_right(q):
    q0, q1, q2, q3 = q
    Q_right = np.array([
        [q0, -q1, -q2, -q3],
        [q1, q0, q3, -q2],
        [q2, -q3, q0, q1],
        [q3, q2, -q1, q0]
    ])
    return Q_right


# input: v and v'=qvq*(quat rotation) 
# output: q
def quat_from_vec(v, v_prime):
    if len(v) == 4: v = v[1:]
    if len(v_prime) == 4: v_prime = v_prime[1:]
    v = np.array(v)
    v_prime = np.array(v_prime)
    v /= np.linalg.norm(v)
    v_prime /= np.linalg.norm(v_prime)
    axis = np.cross(v, v_prime)
    axis_norm = np.linalg.norm(axis)
    
    # Handle the case of zero rotation (vectors are parallel or anti-parallel)
    if axis_norm < 1e-8:  # Parallel vectors
        if np.dot(v, v_prime) > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        else:
            # 180-degree rotation (pick any orthogonal axis)
            orthogonal_axis = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(v, orthogonal_axis)
            axis /= np.linalg.norm(axis)
            return np.array([0.0, *axis])
    
    axis /= axis_norm  # Normalize the axis
    angle = np.arccos(np.clip(np.dot(v, v_prime), -1.0, 1.0)) / 2
    
    w = np.cos(angle)
    x, y, z = axis * np.sin(angle)
    return np.array([w, x, y, z])


def covariance(v):
    covar = [[0 for _ in range(len(v))] for _ in range(len(v))]
    for i in range(len(v)):
        covar[i][i] = v[i] ** 2 
    return covar


class data_format:
    def __init__(self, file_path):
        file_path = os.path.abspath(file_path)
        self.data = pd.read_csv(file_path)
        self.file_path = file_path
        self.gyro0 = 0
        self.mg = None
        self.gg = None
        self.dt = 0.1

        self.acc = {
            'x': self.data['xAxisAcc'] * (-1),
            'y': self.data['yAxisAcc'] * (-1),
            'z': self.data['zAxisAcc'] * (-1)
        }

        self.gyro = {
            'x': self.data['xAxisAngVal'] * np.pi / 180,
            'y': self.data['yAxisAngVal'] * np.pi / 180,
            'z': self.data['zAxisAngVal'] * np.pi / 180
        }

        self.mag = {
            'x': self.data['xAxisMagF'],
            'y': self.data['yAxisMagF'],
            'z': self.data['zAxisMagF']
        }

        self.pos = {
            'x': [np.array([]) for _ in range(len(self.mag['x']))],
            'y': [np.array([]) for _ in range(len(self.mag['x']))],
            'z': [np.array([]) for _ in range(len(self.mag['x']))]
        }

        self.latitude = self.data['gpsLatitude'] * 1e-7
        self.longitude = self.data['gpsLongitude'] * 1e-7
        self.pressure = self.data['pressure']
        self.altitude = 44307.69396 * (1.0 - pow(self.pressure / 101325, 0.190284))
        self.ground_altitude = min(self.altitude)
        self.altitude -= self.ground_altitude
        self.time = self.data['deviceTime']/1000/self.dt

        # self.rotator = [quaternion() for _ in range(len(self.mag['x']))]
        self.rotator = [np.array([]) for _ in range(len(self.mag['x']))]
        self.v = [np.array([]) for _ in range(len(self.mag['x']))]
        self.r = [np.array([]) for _ in range(len(self.mag['x']))]

        self.P = None
        self.Q = None
        self.v = None
        self.R = None

    # How to separate states?
    def find_index(self):
        i = 0

        config.index["stand by"] = 0
        i = config.index["stand by"]

        self.g0 = np.sqrt(self.acc['x'][config.index["stand by"]] ** 2 +
                          self.acc['y'][config.index["stand by"]] ** 2 +
                          self.acc['z'][config.index["stand by"]] ** 2)
        while abs(self.g0 - np.sqrt(
                self.acc['x'][i] ** 2 + self.acc['y'][i] ** 2 + self.acc['z'][i] ** 2)) < self.g0 * 0.05:
            i += 1
        config.index["launch"] = i
        
        i = len(self.acc['x']) - 1
        while abs(self.g0 - np.sqrt(
                self.acc['x'][i] ** 2 + self.acc['y'][i] ** 2 + self.acc['z'][i] ** 2)) < self.g0 * 0.05:
            i -= 1
        config.index["touchdown"] = i
        print(config.index)
        
    def set_initial_condition(self):
        self.find_index()
        gx0 = self.acc['x'][config.index["stand by"]:config.index["launch"]].mean()
        gy0 = self.acc['y'][config.index["stand by"]:config.index["launch"]].mean()
        gz0 = self.acc['z'][config.index["stand by"]:config.index["launch"]].mean()
        gg0 = [gx0, gy0, gz0]
        gg0 = np.array(gg0)

        self.gg = [0, 0, self.g0]
        self.gg = np.array(self.gg)

        axis = np.cross(gg0, self.gg)
        half_angle = np.arcsin(np.linalg.norm(axis) / self.g0 / np.linalg.norm(gg0)) / 2
        print(f'launch angle: {2*half_angle * 180 / np.pi} deg')

        axis = axis / norm(axis)
        # self.rotator[config.index["launch"]] = quaternion.build_from_angle(0, angle, axis)
        self.rotator[config.index["launch"]] = \
            [np.cos(half_angle),
             np.sin(half_angle) * axis[0],
             np.sin(half_angle) * axis[1],
             np.sin(half_angle) * axis[2]]

        mx0 = self.mag['x'][config.index["stand by"]:config.index["launch"]].mean()
        my0 = self.mag['y'][config.index["stand by"]:config.index["launch"]].mean()
        mz0 = self.mag['z'][config.index["stand by"]:config.index["launch"]].mean()
        self.magg = [0, mx0, my0, mz0]

        rot = self.rotator[config.index["launch"]]
        rot2 = [rot[0], -rot[1], -rot[2], -rot[3]]
        temp1 = matrix_left(rot) @ matrix_right(rot2) @ self.magg
        temp2 = matrix_left(rot) @ matrix_right(rot2) @ [0, 0, -1, 0]
        temp1[3] = temp2[3] = 0
        self.rotator_direction = quat_from_vec(temp2, temp1)


        self.P = np.eye(4) * 0.1
        self.w = [0.0,
                  self.mag['x'][config.index["stand by"]:config.index["launch"]].std(),
                  self.mag['y'][config.index["stand by"]:config.index["launch"]].std(),
                  self.mag['z'][config.index["stand by"]:config.index["launch"]].std()
                  ]
        self.Q = covariance(self.w)
        self.v = [0.0,
                  self.gyro['x'][config.index["stand by"]:config.index["launch"]].std() * self.dt,
                  self.gyro['y'][config.index["stand by"]:config.index["launch"]].std() * self.dt,
                  self.gyro['z'][config.index["stand by"]:config.index["launch"]].std() * self.dt
                  ]
        self.R = covariance(self.v)

    def predict(self, idx):
        axis = np.array([self.gyro['x'][idx], self.gyro['y'][idx], self.gyro['z'][idx]])
        # theta = norm(axis) * self.dt
        #theta: the 'roll' rotation of the rocket -> only x and y axis affects the roll
        tx, ty = np.sin(axis[0])**2, np.sin(axis[1])**2
        theta = np.arctan(np.sqrt((tx+ty-2*tx*ty)/(1-tx*ty))) * self.dt
        axis = axis / norm(axis)

        q = [cos(theta / 2), sin(theta / 2) * axis[0], sin(theta / 2) * axis[1], sin(theta / 2) * axis[2]]
        self.rotator[idx + 1] = matrix_left(q) @ self.rotator[idx] 
        self.P = matrix_left(q) @ self.P @ np.transpose(matrix_left(q)) + self.Q

    def update(self, idx):
        q = self.rotator[idx]
        m0 = self.magg
        z = quat_from_vec(m0, np.array([0, self.mag['x'][idx], self.mag['y'][idx], self.mag['z'][idx]]))
        H = np.eye(4)

        # Kalman gain
        S = H @ self.P @ np.transpose(H) + self.R
        K = self.P @ np.transpose(H) @ np.linalg.inv(S)

        # State update
        self.rotator[idx + 1] = self.rotator[idx + 1] + K @ (z - H @ self.rotator[idx + 1])
        self.rotator[idx + 1] = self.rotator[idx + 1] / norm(self.rotator[idx + 1])

        # Covariance update
        self.P = self.P - K @ H @ self.P

    def ekf(self):
        for idx in range(config.index["launch"], config.index["touchdown"]):
            self.predict(idx)
            self.update(idx)

    def summary(self):
        pass

    def plot_data(self, x, y):
        plt.figure(figsize=config.fig_size)
        plt.plot(x, y)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_kalman_trajectory(self, ax, color=None, label=None):
        imu_trajectory = IMUTrajectory(rotator=self.rotator, direction=self.rotator_direction, acc=self.acc, dt=self.dt, gravity=self.g0)
        imu_trajectory.calculate_trajectory(config.index["launch"], config.index["touchdown"])
        imu_trajectory.plot_trajectory(ax, color, label)
        # rocket.animate_trajectory()

    def plot_gps_trajectory(self, ax, color=None, label=None):
        gps_trajectory = GPSTrajectory(latitude=self.latitude, longitude=self.longitude,
                                                  altitude=self.altitude)
        gps_trajectory.calculate_trajectory()
        gps_trajectory.plot_trajectory(ax, color, label)
        for i in range(len(gps_trajectory.altitude)):
            if max(gps_trajectory.altitude) == gps_trajectory.altitude[i]:
                config.index["apogee"] = i
                print(i)

    def print_time(self):
        for x in config.index:
            if x == "stand_by":
                pass
            else:
                config.time[x] = (config.index[x] - config.index["launch"]) * self.dt
        print(config.time)

    def plot_simple_trajectory(self, ax, color=None, label=None):
        simple_trajectory \
            = SIMPLETrajectory(acc=self.acc, dt=self.dt, gravity=self.g0, gyro=self.gyro,
                                          rotator0=self.rotator[config.index["launch"]], direction=self.rotator_direction)
        simple_trajectory.calculate_trajectory(config.index["launch"], config.index["touchdown"])
        simple_trajectory.plot_trajectory(ax, color, label)
