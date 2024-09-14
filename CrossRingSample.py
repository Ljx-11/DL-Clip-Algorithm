# import required libraries
import time
import numpy as np
import cv2
import sys
import os
# import RflySim APIs
import RflySimWindows.PX4MavCtrlV4 as PX4MavCtrl
import RflySimWindows.ScreenCapApiV4 as sca
import RflySimWindows.VisionCaptureApi as VisionCaptureApi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from Networks.cwp_net import ufunc
from ResNet18 import ResNet18Enc
from PIL import Image

import csv
import datetime

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

time1 = datetime.datetime.now()
data_save_index = time1.strftime('%Y-%m-%d_%H-%M-%S')

##########################################################################################
# VisionCaptureApi 中的配置函数
vis = VisionCaptureApi.VisionCaptureApi()
vis.jsonLoad() # 加载Config.json中的传感器配置文件
# nop
isSuss = vis.sendReqToUE4() # 向RflySim3D发送取图请求，并验证
if not isSuss: # 如果请求取图失败，则退出
    sys.exit(0)
vis.startImgCap(True) # 开启取图，并启用共享内存图像转发，转发到填写的目录

# Send command to UE4 Window 1 to change resolution 
vis.sendUE4Cmd(b'r.setres 720x405w',0) # 设置UE4窗口分辨率，注意本窗口仅限于显示，取图分辨率在json中配置，本窗口设置越小，资源需求越少。
vis.sendUE4Cmd(b't.MaxFPS 30',0) # 设置UE4最大刷新频率，同时也是取图频率
time.sleep(2)    

width = 720
height = 405
channel = 4
index_image = 1

# init pos while image servo starts
init_servo_pos = [0, 0, 0]

# define same functions for computaion
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def diagonal_check(p):
    d1 = np.sqrt(np.dot(p[0]-p[2], p[0]-p[2]))
    d2 = np.sqrt(np.dot(p[1]-p[3], p[1]-p[3]))
    return abs(d1-d2)*1.0/d1

def saturationYawRate(yaw_rate):
    yr_bound = 20.0
    if yaw_rate > yr_bound:
        yaw_rate = yr_bound
    if yaw_rate < -yr_bound:
        yaw_rate = -yr_bound
    return yaw_rate

def sat(inPwm,thres=1):
    outPwm= inPwm
    if inPwm>thres:
        outPwm = thres
    elif inPwm<-thres:
        outPwm = -thres
    return outPwm

# Get image servoing data
def dataEngine(pos_y, pos_z, max_vel_y, max_vel_z):
    global init_servo_pos,index_image
    mav.SendPosNED(0, pos_y, pos_z, 0)
    time.sleep(5)
    # 记录起始位置
    init_servo_pos[0] = mav.uavPosNED[0]
    init_servo_pos[1] = mav.uavPosNED[1]
    init_servo_pos[2] = mav.uavPosNED[2]
    print('init pos',init_servo_pos)

    task = "range1"
    # 1. start
    startAppTime= time.time()
    lastTime = time.time()

    timeInterval = 1/30.0 #here is 0.0333s (30Hz)
    stateDataAll = []
    num=0
    lastClock=time.time()
    while (task != "finish") & (task != "land"):
        lastTime = lastTime + timeInterval
        sleepTime = lastTime - time.time()
        if sleepTime > 0:
            time.sleep(sleepTime) # sleep until the desired clock
        else:
            lastTime = time.time()
        # The following code will be executed 30Hz (0.0333s)
        num=num+1
        if num%100==0:
            tiem=time.time()
            print('MainThreadFPS: '+str(100/(tiem-lastClock)))
            print('mav pos',mav.uavPosNED)
            print('ex',ex,'ey',ey)
            lastClock=tiem   
        
        # 1.1.检测是否到达了目标
        ex = mav.uavPosNED[1] + goal_x_pos
        ey = mav.uavPosNED[2] + goal_y_pos
        if np.abs(ex) < 0.2 and np.abs(ey) < 0.2:
            break

        # 1.2. 获取前视摄像头的图像，显示，并且保存
        if vis.hasData[0]:           
            img_bgr=vis.Img[0]
            cv2.imshow("img_bgr", img_bgr)
            cv2.moveWindow("img_bgr",xx,0)
            cv2.waitKey(1)
        if vis.hasData[1]:           
            img_bgr_1=vis.Img[1]
            cv2.imwrite('./images/image_dataset_{}/image_{}/{}.jpg'.format(data_save_index,str(data_index),str(index_image)),img_bgr_1)
            index_image = index_image + 1

        with open('./flight_data/flight_data_metric_{}/cwp_dataengine_{}.csv'.format(data_save_index, str(data_index)),mode='a', newline="") as cfa:
            wf = csv.writer(cfa)
            timestamp = time.time()
            data2 = [[mav.uavPosNED[0], mav.uavPosNED[1], mav.uavPosNED[2],
                         mav.uavVelNED[0],mav.uavVelNED[1],mav.uavVelNED[2],
                         mav.uavAngEular[0],mav.uavAngEular[1],mav.uavAngEular[2],
                         mav.uavAngRate[0],mav.uavAngRate[1],mav.uavAngRate[2],
                         ex, ey,
                         init_servo_pos[0],init_servo_pos[1],init_servo_pos[2],
                         index_image,timestamp
                         ]]
            for i in data2:
                wf.writerow(i)
        vx = 0
        vy = sat(-1 * ex, max_vel_y)
        vz = sat(-1 * ey, max_vel_z)
        yawrate = 0

        mav.SendVelFRD(vx, vy, vz, yawrate)
    
    print(data_index,'epoch is finished!')
    index_image = 1
    lastTime = time.time()

# 基于神经网络的图像伺服控制器
def imageServoController(pos_y, pos_z):
    global init_servo_pos,index_image
    mav.SendPosNED(0, pos_y, pos_z, 0)
    time.sleep(5)
    # 记录起始位置
    init_servo_pos[0] = mav.uavPosNED[0]
    init_servo_pos[1] = mav.uavPosNED[1]
    init_servo_pos[2] = mav.uavPosNED[2]
    print('init pos',init_servo_pos)

    task = "range1"
    # 1. start
    startAppTime= time.time()
    lastTime = time.time()

    timeInterval = 1/30.0 #here is 0.0333s (30Hz)
    stateDataAll = []
    num=0
    lastClock=time.time()
    while (task != "finish") & (task != "land"):
        lastTime = lastTime + timeInterval
        sleepTime = lastTime - time.time()
        if sleepTime > 0:
            time.sleep(sleepTime) # sleep until the desired clock
        else:
            lastTime = time.time()
        # The following code will be executed 30Hz (0.0333s)
        num=num+1
        if num%100==0:
            tiem=time.time()
            print('MainThreadFPS: '+str(100/(tiem-lastClock)))
            print('mav pos',mav.uavPosNED)
            print('ex',ex,'ey',ey)
            lastClock=tiem   

        # 1.1.检测是否到达了目标
        ex = mav.uavPosNED[1] + 0.0
        ey = mav.uavPosNED[2] + 9.0
        if np.abs(ex) < 1.0 and np.abs(ey) < 1.0:
            break
        if time.time() - startAppTime >25:
            break

        # 1.2. 获取前视摄像头的图像，显示，并且保存
        if vis.hasData[0]:           
            img_bgr=vis.Img[0]
            cv2.imshow("img_bgr", img_bgr)
            cv2.moveWindow("img_bgr",xx,0)
            cv2.waitKey(1)
        if vis.hasData[1]:           
            img_bgr_1=vis.Img[1]
            cv2.imwrite('./images/image_dataset_{}/image_{}/{}.jpg'.format(data_save_index,str(data_index),str(index_image)),img_bgr_1)
            index_image = index_image + 1
        
        encoding = ImageTensor(img_bgr_1, FeatureNet, TransformImage, Normalizer)
        # encoding = encoding - goal_encoding

        with open('./flight_data/flight_data_metric_{}/cwp_dataengine_{}.csv'.format(data_save_index, str(data_index)),mode='a', newline="") as cfa:
            wf = csv.writer(cfa)
            timestamp = time.time()
            
            data2 = [[mav.uavPosNED[0], mav.uavPosNED[1], mav.uavPosNED[2],
                         mav.uavVelNED[0],mav.uavVelNED[1],mav.uavVelNED[2],
                         mav.uavAngEular[0],mav.uavAngEular[1],mav.uavAngEular[2],
                         mav.uavAngRate[0],mav.uavAngRate[1],mav.uavAngRate[2],
                         ex, ey,
                         init_servo_pos[0],init_servo_pos[1],init_servo_pos[2],
                         index_image,timestamp
                         ]]
            for i in data2:
                wf.writerow(i)
        
        x_feature = encoding - goal_encoding
        x_feature = torch.from_numpy(x_feature).view(1,-1).cuda().to(torch.float32)
        cmd = ControllerNet(x_feature)
        cmd = cmd.cpu().detach().numpy()
        cmd = cmd[0]
        vx = 0
        vy = cmd[1]
        vz = cmd[2]
        yawrate = 0
        mav.SendVelFRD(vx, vy, vz, yawrate)
    print(data_index,'epoch is finished!')
    index_image = 1
    lastTime = time.time()

def NeuralControllerInit(
        controller_path = './models/k_net_cuda_still.pt',
        encoder_path = './models/model.train',
        feature_dim = 32,
        image_size = (256,256)
):
    ControllerNet = ufunc().to(device)
    # ControllerNet.load_state_dict(torch.load(controller_path))
    # ControllerNet = torch.load(controller_path)
    ControllerNet.load_state_dict(
            torch.load(controller_path)
        )
    ControllerNet.eval()

    FeatureNet = ResNet18Enc(num_Blocks=[2, 2, 2, 2], z_dim=feature_dim, nc=3).to(device)
    FeatureNet.load_state_dict(torch.load(encoder_path))
    FeatureNet.eval()

    transform_ops = []
    transform_ops.append(transforms.Resize(image_size))
    transform_ops.append(transforms.ToTensor())
    #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
    TransformImage = transforms.Compose(transform_ops)
    Normalizer = transforms.Normalize(mean=0.5, std=0.5)
    return ControllerNet, FeatureNet, TransformImage, Normalizer

def ImageTensor(img_bgr_1, FeatureNet, TransformImage, Normalizer):
    global device
    img_bgr = Image.fromarray(cv2.cvtColor(img_bgr_1,cv2.COLOR_BGR2RGB))
    img_as_tensor = TransformImage(img_bgr)
    img_as_tensor = Normalizer(img_as_tensor)
    img_as_tensor = img_as_tensor.unsqueeze(0)
    img_tensor = img_as_tensor.to(device)
    encoding = FeatureNet(img_tensor)
    if torch.cuda.is_available() == True:
        encoding = encoding.cpu().detach().numpy().flatten()
    else:
        encoding = encoding.detach().numpy().flatten()
    # encoding_d = stateDataAll1[-1,19:]
    return encoding

# main function
if __name__ == '__main__':
    goal_x_pos = 0 # pickle
    goal_y_pos = 9.0
    is_init_dataset_file = "D:\\Dlearning_PPO\\result\\is_init_dataset.txt"
    is_sample_end_path = 'D:\\Dlearning_PPO\\result\\is_sample_end.txt'


    if os.path.exists(is_init_dataset_file):
        is_init_dataset = True
    else:
        is_init_dataset = False
    
    # print(device)
    ControllerNet, FeatureNet, TransformImage, Normalizer = NeuralControllerInit(
        controller_path = './models/controller.pth',
        encoder_path = './models/model.train',
        feature_dim = 32,
        image_size = (256,256)
    )
    

    mav = PX4MavCtrl.PX4MavCtrler(20100)

    mav.InitMavLoop()
    window_hwnds = sca.getWndHandls()
    time.sleep(2) 
    xx,yy=sca.moveWd(window_hwnds[0],0,0,True)   
    print("Simulation Start.")

    print("Enter Offboard mode.")
    time.sleep(10)    
    mav.initOffboard()
    time.sleep(5)
    mav.SendMavArm(True) # Arm the drone
    mav.SendPosNED(0, 0, -9, 0) # Fly to target position 0,0，-5    

    if vis.hasData[0]:           
        img_bgr=vis.Img[0]
        cv2.imshow("img_bgr", img_bgr)
        cv2.moveWindow("img_bgr",xx,0)
        cv2.waitKey(1)
    if vis.hasData[1]:           
        img_bgr_1=vis.Img[1]
        # cv2.imshow("img_bgr_1", img_bgr_1)
        # cv2.moveWindow("img_bgr_1",0,yy)
        # cv2.waitKey(1)
    time.sleep(8)

    print('pos',mav.uavPosNED)
    print('vel',mav.uavVelNED)
    
    sample_init_pos = np.loadtxt('sample_init_data.csv',delimiter=",")

    # while data_index <= 300:
    flight_data_folder_name = './flight_data/flight_data_metric_{}'.format(data_save_index)
    if not os.path.exists(flight_data_folder_name):
        os.makedirs(flight_data_folder_name)
    image_dataset_parent_folder = './images/image_dataset_{}'.format(data_save_index)
    if not os.path.exists(image_dataset_parent_folder):
        os.makedirs(image_dataset_parent_folder)

    img_goal_path = 'goal.jpg'
    img_goal = Image.open(img_goal_path)
    img_goal_tensor = TransformImage(img_goal)
    img_goal_tensor = Normalizer(img_goal_tensor)
    img_goal_tensor = img_goal_tensor.unsqueeze(0)
    img_goal_tensor = img_goal_tensor.to(device)
    goal_encoding = FeatureNet(img_goal_tensor)
    
    if torch.cuda.is_available() == True:
        goal_encoding = goal_encoding.cpu().detach().numpy().flatten()
    else:
        goal_encoding = goal_encoding.detach().numpy().flatten()
    for data_index in range(len(sample_init_pos)):
        # print('now data_index',data_index)
        
        # filename1 = './flight_data_metric/cwp_dataengine_{}.csv'.format(str(data_index))
        # stateDataAll1 = np.loadtxt(open(filename1,"rb"),delimiter=",",skiprows=1)
        init_pos_x = sample_init_pos[data_index,0]
        init_pos_y = sample_init_pos[data_index,1]
        # if data_index <= 40:
        #     init_pos_x = init_pos_x - 0.2
        # elif data_index > 40 and data_index <= 150:
        #     init_pos_y = init_pos_y + 0.2
        # elif data_index > 150 and data_index <= 190:
        #     init_pos_x = init_pos_x + 0.2
        # else:
        #     init_pos_y = init_pos_y - 0.2

        # if data_index %1 == 0:
        image_folder_name = './images/image_dataset_{}/image_{}'.format(data_save_index,str(data_index))
        if not os.path.exists(image_folder_name):
            os.makedirs(image_folder_name)
        with open('./flight_data/flight_data_metric_{}/cwp_dataengine_{}.csv'.format(data_save_index, str(data_index)),mode='w', newline="") as cf:
            wf=csv.writer(cf)
            header_list = ['mav_pos_x', 'mav_pos_y', 'mav_pos_z',
                     'mav_vel_x','mav_vel_y','mav_vel_z',
                     'mav_yaw','mav_pitch','mav_roll',
                     'mav_yaw_rate','mav_pitch_rate','mav_roll_rate',
                     'delta_u','delta_v',
                     'init_pos_x','init_pos_y','init_pos_z',
                     'index_image','timestamp'
                     ]
            # for ii in range(32):
            #     header_list.append('encoding_{}'.format(str(ii+1)))
            wf.writerow(header_list)
            # data2 = [[mav.uavPosNED[0], mav.uavPosNED[1], mav.uavPosNED[2],
            #                      mav.uavVelNED[0],mav.uavVelNED[1],mav.uavVelNED[2],
            #                      mav.uavAngEular[0],mav.uavAngEular[1],mav.uavAngEular[2],
            #                      mav.uavAngRate[0],mav.uavAngRate[1],mav.uavAngRate[2],
            #                      ex, ey,index_image,timestamp
            #                      ]]
            print('write first line')
        if is_init_dataset == True:
            dataEngine(
                pos_y = init_pos_y,
                pos_z = init_pos_x,
                max_vel_y = 0.3,
                max_vel_z = 0.3 
            )
        else:
            imageServoController(
                init_pos_y,
                init_pos_x
            )
        # data_index = data_index + 1
    with open(is_sample_end_path,'w') as file_sample:
        file_sample.write('Sample End')
    # pos [0.021083889529109, 11.107991218566895, -5.18729829788208]
    # vel [-0.0024721664376556873, -0.3554072678089142, 0.21858523786067963]
    # start crossing ring task    
    # approachObjective()
