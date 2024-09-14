import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
import random

image_parent = 'D:\\Dlearning_PPO\\images\\'
pose_parent = 'D:\\Dlearning_PPO\\flight_data\\'
# path = 'task_6'
files = os.listdir(image_parent)
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(image_parent, x)))
image_dir = image_parent + files[-1]
print(image_dir)

files = os.listdir(pose_parent)
files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(pose_parent, x)))
pose_dir = pose_parent + files[-1]
print(pose_dir )
random.seed(42)

def get_data_info(folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        # poses = np.load('{}{}.npy'.format(pose_dir, folder))  # (n_images, 6)
        stateDataAll = np.loadtxt(open('{}/cwp_dataengine_{}.csv'.format(pose_dir, str(folder)),"rb"),delimiter=",",skiprows=1)
        fpaths = glob.glob('{}/image_{}/*.jpg'.format(image_dir, str(folder)))
        fpaths.sort(key = os.path.getctime)
        # Fixed seq_len
        for i in range(len(stateDataAll)):
            image_index = stateDataAll[i,17] - 1
            final_image_index = stateDataAll[-1,17] - 1
            x_segs = []
            x_seg = '{}/image_{}/{}.jpg'.format(image_dir, str(folder),str(int(image_index)))
            x_segs.append(x_seg)
            # x_seg_final = '{}image_{}/{}.jpg'.format(image_dir, str(folder),str(int(final_image_index)))
            # x_segs.append(x_seg_final)
            X_len.append(len(x_segs))
            X_path.append(x_segs)
            x_poses = []
            x_poses.append(np.hstack((stateDataAll[i,0:3],stateDataAll[i,6:9])))
            Y.append(x_poses)
        print('Folder {} finish in {} sec'.format(str(folder), time.time()-start_t))
    
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='rescale', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        # print('Item after transform: ' + str(index) + '   ' + str(groundtruth_sequence))
        groundtruth_sequence = self.groundtruth_arr[index]
        image_path_sequence = self.image_arr[index]
        index_d = random.randint(0,len(self.data_info.index)-1)
        while index_d == index:
            index_d = random.randint(1,len(self.data_info.index)-1)
        image_path_sequence.append(self.image_arr[index_d][0])
        groundtruth_sequence.append(self.groundtruth_arr[index_d][0])
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        groundtruth_sequence = np.vstack(groundtruth_sequence)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

# Example of usage
if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    sample_times = 1
    folder_list = []
    for i in range(301):
        folder_list.append(i)
    folder_list = [0]
    seq_len_range = [5, 7]
    df = get_data_info(folder_list, seq_len_range, overlap, sample_times)
    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))
    # Customized Dataset, Sampler
    n_workers = 4
    resize_mode = 'crop'
    new_size = (150, 600)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean)
    # sorted_sampler = SortedRandomBatchSampler(df, batch_size=1, drop_last=True)
    # dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=n_workers)
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))
    metric_p = torch.tensor(0.0)
    metric_theta = torch.tensor(0.0)
    for batch in dataloader:
        s, x, y = batch
        # print('='*50)
        # print(y)
        # x:torch.Size([1, 2, 3, 150, 600])
        # y:torch.Size([1, 2, 6])
        metric_p = metric_p + torch.norm(y[0,1,0:3]-y[0,0,0:3])
        metric_theta = metric_theta + torch.norm(y[0,1,3:6]-y[0,0,3:6])
        # print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))
        # print(x[0,0].shape)
        x2 = torch.cat(( x[0,0], x[0,1]), dim=0)
        # print(x2.shape)
    print(metric_p)
    print(metric_theta)
    c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
    print(c.size())
    # metric_p / metric_theta = 130
    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)