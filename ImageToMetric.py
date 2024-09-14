import numpy as np
from PIL import Image
import glob
import os
import time
import torch
from MetricLearning.data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Networks.ResNet18 import ResNet18Enc
from Networks.se3_net import SE3Net
from tqdm import trange, tqdm
from itertools import chain
import torchvision.utils as vutils

import csv
import cv2
from torch.utils.data.sampler import Sampler
from torchvision import transforms

image_parent = 'D:\\Dlearning_PPO\\images\\'
pose_parent = 'D:\\Dlearning_PPO\\flight_data\\'
metric_parent = 'D:\\Dlearning_PPO\\flight_data_metric\\'
is_metric_end_path = 'D:\\Dlearning_PPO\\result\\is_metric_end.txt'
# path = 'task_6'
files = os.listdir(image_parent)
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(image_parent, x)))
image_dir = image_parent + files[-1]
# print(image_dir)

files = os.listdir(pose_parent)
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(pose_parent, x)))
pose_dir = pose_parent + files[-1]

metric_dataset_dir = metric_parent + files[-1]
if not os.path.exists(metric_dataset_dir):
    os.makedirs(metric_dataset_dir)

# print(pose_dir )

if __name__ == "__main__":

    transf = transforms.ToTensor()
    transform_ops = []
    transform_ops.append(transforms.Resize((256,256)))
    transform_ops.append(transforms.ToTensor())
    #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
    transform = transforms.Compose(transform_ops)
    normalizer = transforms.Normalize(mean=0.5, std=0.5)
    use_cuda = torch.cuda.is_available()
    device = torch.device(0 if use_cuda else "cpu")
    fnet = ResNet18Enc(num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3)
    se3_net = SE3Net()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        fnet = fnet.cuda()
        se3_net = se3_net.cuda()
        fnet.load_state_dict(torch.load('./models/model.train'))
    fnet.eval()
    
    for i in trange(301):
        
        filename = pose_dir + '/cwp_dataengine_{}.csv'.format(str(i))
        if not os.path.exists(metric_dataset_dir):
            break
        if not os.path.exists(filename):
            break
        stateDataAll1 = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=1)

        with open( metric_dataset_dir + '/cwp_dataengine_{}.csv'.format(str(i)), mode='w', newline="") as cf:
            wf=csv.writer(cf)
            header_list = ['mav_pos_x', 'mav_pos_y', 'mav_pos_z',
                         'mav_vel_x','mav_vel_y','mav_vel_z',
                         'mav_yaw','mav_pitch','mav_roll',
                         'mav_yaw_rate','mav_pitch_rate','mav_roll_rate',
                         'delta_u','delta_v',
                         'init_pos_x','init_pos_y','init_pos_z',
                         'index_image','timestamp'
                         ]
            for ii in range(32):
                header_list.append('encoding_{}'.format(str(ii+1)))
            wf.writerow(header_list)
        
        for j in trange(len(stateDataAll1) , leave= False):
            image_index = stateDataAll1[j,17] - 1
            # img_bgr = cv2.imread('./image_dataset/image_{}/{}.jpg'.format(str(i),str(int(image_index))))
            if os.path.exists( image_dir + '/image_{}/{}.jpg'.format(str(i),str(int(image_index)))) is False:
                 break
            img_bgr = Image.open( image_dir + '/image_{}/{}.jpg'.format(str(i),str(int(image_index))))
            # print(i)
            if img_bgr is None:
                 break
            img_as_tensor = transform(img_bgr)
            img_as_tensor = normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            img_tensor = img_as_tensor.to(device)
            encoding = fnet(img_tensor)
            encoding = encoding.cpu().detach().numpy().flatten()
            line_data = np.hstack((stateDataAll1[j,:],encoding))
            with open( metric_dataset_dir + '/cwp_dataengine_{}.csv'.format(str(i)),mode='a', newline="") as cfa:
                    wf = csv.writer(cfa)
                    data2 = line_data.tolist()
                    data2 = [data2]
                    for ij in data2:
                        wf.writerow(ij)
                        # print(image_feature_1.size())
                        # pca_list.append(image_feature_1[0].cpu().detach().numpy())

    with open(is_metric_end_path,'w') as file_sample:
        file_sample.write('Metric End')