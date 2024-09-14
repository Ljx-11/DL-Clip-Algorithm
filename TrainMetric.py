# predicted as a batch
# from model import FeatureNet
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

def d_se3(rj,rk):
    d = torch.norm(rj[:,0:3]-rk[:,0:3] , dim=1)  + torch.norm(rj[:,3:6]-rk[:,3:6] , dim=1)
    # print(d.size())
    return d

if __name__ == '__main__':    

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    videos_to_test = []
    sample_init_pos = np.loadtxt('sample_init_data.csv',delimiter=",")
    for i in range(len(sample_init_pos)):
        videos_to_test.append(i)

    save_dir = 'result/'  # directory to save prediction answer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    # Load model
    fnet = ResNet18Enc(num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3)

    se3_net = SE3Net()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        fnet = fnet.cuda()
        se3_net = se3_net.cuda()
        fnet.load_state_dict(torch.load('./models/model-old.train'))
    else:
        fnet = fnet.cpu()
        se3_net = se3_net.cpu()

    n_workers = 1
    
    optimizer = torch.optim.Adam(chain(fnet.parameters(),se3_net.parameters()), lr=0.001, betas=(0.9, 0.999))
    fd=open('test_dump.txt', 'w')
    fd.write('\n'+'='*50 + '\n')
    model_loss = np.array([])
    # for test_video in videos_to_test:
    df = get_data_info(folder_list=videos_to_test, seq_len_range=0, overlap=0, sample_times=1, shuffle=False, sort=False)
    # df = df.loc[df.seq_len == seq_len]  # drop last
    df.to_csv('test_df.csv')
    dataset = ImageSequenceDataset(df, 'rescale', (256, 256), 0.5, 0.5, False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=n_workers)

    is_init_dataset = True
    if is_init_dataset == True:
        num_epoch = 200
    else:
        num_epoch = 1
    for epoch in trange(num_epoch,leave=False):
        # Predict
        fnet.train()
        se3_net.train()
        
        has_predict = False
        # answer = [[0.0]*6, ]
        st_t = time.time()
        n_batch = len(dataloader)
        # loss = torch.tensor(0.0)
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            # print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            _, x, y = batch
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            
            image_feature_1 = fnet(x[:,0])
            image_feature_2 = fnet(x[:,1])
            se3_feature_1 = se3_net(y[:,0,:])
            se3_feature_2 = se3_net(y[:,1,:])
            
            loss1 = torch.norm(d_se3(y[:,0,:],y[:,1,:]) - torch.norm(se3_feature_1 - se3_feature_2 , dim=1)).pow(2)
            loss2 = torch.norm(torch.norm(image_feature_1 - se3_feature_2 , dim=1) - torch.norm(se3_feature_1 - se3_feature_2 , dim=1)).pow(2)
            loss3 = torch.norm(torch.norm(image_feature_1 - image_feature_2 , dim=1) - torch.norm(se3_feature_1 - se3_feature_2 , dim=1)).pow(2)
            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_loss=np.append(model_loss,loss.cpu().detach().numpy())
        np.savetxt(r'test.csv', model_loss, delimiter=',')
        torch.save(fnet.state_dict(), 'model.train')
        torch.save(optimizer.state_dict(), 'optim.train')

