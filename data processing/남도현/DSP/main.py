import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< Updated upstream:data processing/남도현/DSP/main.py
import config
import data
from trajectories import IMUTrajectory, GPSTrajectory, SimpleTrajectory
=======
import EKF1
>>>>>>> Stashed changes:main.py

file_path = r"C:\Users\82108\Desktop\Hanaro\Data Processing\Data\identity3-B_lora_data.csv"
fig_size = (10, 10)

if __name__ == '__main__':
<<<<<<< Updated upstream:data processing/남도현/DSP/main.py
    obj = data.data_format(file_path)
=======
    print("x: East, y: North")
    obj = EKF1.data_format(file_path)
>>>>>>> Stashed changes:main.py
    obj.set_initial_condition()
    obj.ekf()

    # obj.plot_data(
    #     obj.time[config.index["launch"]:config.index["touchdown"]],
    #     obj.altitude[config.index["launch"]:config.index["touchdown"]],
    # )

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    obj.plot_kalman_trajectory(ax, color="red", label="Kalman")
    obj.plot_gps_trajectory(ax, color="green", label="GPS")
    obj.plot_simple_trajectory(ax, color="blue", label="Simple")
    obj.print_time()

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
<<<<<<< Updated upstream:data processing/남도현/DSP/main.py
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
=======
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
>>>>>>> Stashed changes:main.py
    ax.set_zlim(0, 300)
    ax.legend()

    plt.show()
<<<<<<< Updated upstream:data processing/남도현/DSP/main.py
=======
    plt.savefig("./results/Trajectories.png")
    plt.close()
>>>>>>> Stashed changes:main.py
