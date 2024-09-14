#  DL-Clip: Online D-Learning with Clipping Operation for Fast Model-Free Stabilizing Control

The implemention of the paper * DL-Clip: Online D-Learning with Clipping Operation for Fast Model-Free Stabilizing Control* .

## Configuring the RflySim simulation environment

Refer to [RflySim](https://rflysim.com/) .

⚠Note: RflySim can only be installed on Windows 10 or Windows 11. Computer configuration includes at least a high-performance CPU and discrete graphics card. MATLAB r2017b or above is already installed on the system.

Since RflySim takes up a lot of system space (more than 10 G), it is recommended to install it on a non-system disk (e.g. D:)

`goal.jpg` is the goal image for the IBVS experiment.

## Training Script

Since RflySim can only run on Windows 10 or Windows 11, please copy this code base to `D:\Dlearning_PPO\` folder and run it on Windows.

The version of `Python` is `3.8.19`.

Before running, please `pip install jupyter pywin32`,  and `pip install -r requirements.txt`.

And run `jupyter notebook dl_clip.ipynb`.

## Datasets

Flight datasets after feature extraction are structured as

```
./flight_data_metric/flight_data_metric_<TIME>/cwp_dataengine_<DATA_INDEX>.csv
```

Available fields are given in the following table.

| `field`                     | description                                |
| --------------------------- | ------------------------------------------ |
| `mav_pos_<x,y,z>`           | position vector in meters                  |
| `mav_vel_<x,y,z>`           | velocity vector in m/s                     |
| `mav_<yaw,pitch,roll>`      | attitude represented as Euler angle in rad |
| `mav_<yaw,pitch,roll>_rate` | angular velocity in rad/s                  |
| `delta_<u,v>`               | tracking error                             |
| `init_pos_<x,y,z>`          | initial position of multicopter            |
| `timestamp`                 | timestamp                                  |
| `encoding_<1-32>`           | encoding from real-time image              |

## Files

```
.
├── Bats # Stores batch files on Windows
├── Compare_Pendulum.ipynb
├── CrossRingSample.py # IBVS Simulation Experiment Controller Code
├── cwp_train.py
├── DL-Clip_dataset # The datasets for DL-Clip based online iterative controller
├── dl_clip.ipynb # Code for IBVS controller training and online iteration
├── DL-Clip-Pendulum.ipynb
├── DL-Online_dataset # The datasets for DL-Online based online iterative controller
├── DL-Online-Pendulum.ipynb
├── flight_data # The folder where flight data is stored
├── flight_data_metric # The folder where the flight data is stored after feature extraction
├── goal.jpg # Target image for IBVS
├── images # The folder where image is stored
├── ImageToMetric.py # Script that take an image through feature extraction
├── MetricLearning # Library files for deep metric learning
│   └── data_helper.py
├── models # Model weights for feature extraction, D-function and controller
├── Networks # Network structure 
│   ├── cwp_net.py
│   ├── ResNet18.py
│   └── se3_net.py
├── PPO_Pendulum.ipynb
├── README.md
├── requirements.txt
├── ResNet18.py
├── result
│   ├── controller_DL_Clip.pth
│   ├── controller_DL_Online.pth
│   ├── fig2_a.png
│   ├── fig2_b.png
│   ├── fig2_c.png
│   ├── fig2_d.png
│   ├── is_init_dataset.txt
│   ├── lyapunov_DL_Clip.pth
│   ├── lyapunov_DL_Online.pth
│   ├── lyapunov.png
│   ├── stepconverge_list_DL_Clip.pk
│   ├── stepconverge_list_DL_Online.pk
│   └── stepconverge_list_PPO.pk
├── RflySimWindows # Library files for RflySim simulation communication
│   ├── Config.json
│   ├── PX4MavCtrlV4.py
│   ├── ScreenCapApiV4.py
│   └── VisionCaptureApi.py
├── sample_init_data.csv # The list of starting positions of the IBVS
├── systems_and_functions
│   ├── cart_pole_system.py
│   ├── control_affine_system.py
│   ├── cwp_train.py
│   ├── d_function_4_linear_system.py
│   ├── __init__.py
│   ├── inverted_pendulum_system.py
│   ├── kits.py
│   └── linear_control_affine_system.py
├── test_dump.txt
└── TrainMetric.py # Script for deep metric learning
```
