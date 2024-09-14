import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
from tqdm import trange

def my_pca(A):
    MEAN = np.mean(A, axis=0)
    X = np.subtract(A, MEAN)
    COV = np.dot(X.T, X)
    W, V = np.linalg.eig(COV)
    sum_lambda = np.sum(W)
    f = np.divide(W, sum_lambda)
    e1 = V.T[0]
    e2 = V.T[1]
    z1 = np.dot(X, e1)
    z2 = np.dot(X, e2)
    RES = np.array([z1,z2]).T
    return RES,np.array([e1,e2]).T

def dataset_split(stateData_one_index):
    uavPosNED          =   stateData_one_index[:,0:3]
    uavVelNED          =   stateData_one_index[:,3:6]
    uavAngEular        =   stateData_one_index[:,6:9]
    uavAngRate         =   stateData_one_index[:,9:12]
    delta_uv           =   stateData_one_index[:,12:14]
    timestamp          =   stateData_one_index[:,18]
    metric_encoding    =   stateData_one_index[:,19:]
    metric_encoding_d  =   stateData_one_index[-1,19:]
    return [
        uavPosNED,         
        uavVelNED,         
        uavAngEular,       
        uavAngRate,        
        delta_uv,          
        timestamp,         
        metric_encoding,   
        metric_encoding_d 
    ]

def calc_lyap_value(
    stateData_all: list,
    P_pca,
    P1_value
):
    delta_v_list = []
    lyap_dot_label_list = []
    input_features_list = []


    for j in range(len(stateData_all)):
        dataLength = len(stateData_all[j])
        [uavPosNED, uavVelNED, uavAngEular, uavAngRate,
          delta_uv, timestamp, metric_encoding, metric_encoding_d] = dataset_split(stateData_all[j])
        input_features = np.hstack((uavVelNED[0:dataLength-1,:],metric_encoding[0:dataLength-1,:] ))
        lyap_dot_label = np.zeros(dataLength-1)
        # lyap_dot_label = np.zeros(dataLength-1)
        for i in range(dataLength-1):
            lyap_dot_label[i] = ((metric_encoding[i+1,:] @ P_pca - metric_encoding_d @ P_pca ) @ P1_value @ (metric_encoding[i+1,:]  @ P_pca - metric_encoding_d @ P_pca ) 
                        - (metric_encoding[i,:] @ P_pca - metric_encoding_d @ P_pca ) @ P1_value @ (metric_encoding[i,:] @ P_pca - metric_encoding_d @ P_pca ) )
        lyap_dot_label_list.append(lyap_dot_label)
        input_features_list.append(input_features)
    
    return lyap_dot_label_list, input_features_list


class CWP_PPO():
    def __init__(self,
        DFunction: torch.nn.Module,
        optimizer_dfunc: torch.optim.Optimizer,
        ControllerNet: torch.nn.Module,
        optimizer_ufunc: torch.optim.Optimizer,
        device
    ):
        super(CWP_PPO, self).__init__()
        self.DFunction = DFunction
        self.optimizer_dfunc = optimizer_dfunc
        self.ControllerNet = ControllerNet
        self.optimizer_ufunc = optimizer_ufunc
        self.device = device

    def train_d_function(
        self,
        train_epoch: int,
        input_features_list: list,
        lyap_dot_label_list: list,
        stateData_all: list,

    ):

        # DFunction = dNet().to(device)
        self.DFunction.train()
        self.ControllerNet.eval()
        criterion = torch.nn.MSELoss(reduction = 'sum')
        # optimizer = torch.optim.Adam(DFunction.parameters(),lr =0.001,betas = (0.9,0.999),foreach=False)

        model_loss = np.array([])

        loss_list = []
        test_loss = np.array([])

        for epoch in trange(train_epoch):
            for j in range(len(input_features_list)):
                # Select 285 trajectories as the training set
                if j % 20 != 0 :
                    dataLength = len(input_features_list[j])
                    y_label = lyap_dot_label_list[j]
                    delete_list = []
                    # Because numerical simulation sometimes produces a certain delay,
                    # At this time, the approximate derivative of the Lyapunov function may have a very small value,
                    # This data needs to be cleared to reduce the impact on the solution of the D function.
                    for ii in range(len(y_label)):
                        if np.abs(y_label[ii])< 2e-3:
                            delete_list.append(ii)
                        if np.abs(y_label[ii])> 1.0:
                            delete_list.append(ii)  
                    y_label = np.delete(y_label,delete_list,0)
                    y_label = torch.from_numpy(y_label).to(self.device)

                    [uavPosNED, uavVelNED, uavAngEular, uavAngRate,
                     delta_uv, timestamp, metric_encoding, metric_encoding_d] = dataset_split(stateData_all[j])
                    metric_encoding = metric_encoding - metric_encoding_d
                    d_feature = np.hstack((uavVelNED[0:dataLength,:],metric_encoding[0:dataLength,:] ))
                    # Delete the index corresponding to the previously mentioned very small value
                    d_feature = np.delete(d_feature,delete_list,0)
                    d_feature = torch.from_numpy(d_feature).to(self.device)

                    d_function = self.DFunction(d_feature.to(torch.float32))
                    loss = criterion(d_function.to(torch.float32),y_label.view(-1,1).to(torch.float32))
                    model_loss=np.append(model_loss,loss.cpu().detach().numpy())

                    self.optimizer_dfunc.zero_grad()
                    loss.backward()
                    self.optimizer_dfunc.step()

                # Select 15 trajectories as the testing set
                else:
                    with torch.no_grad():
                        dataLength = len(input_features_list[j])
                        y_label = lyap_dot_label_list[j]
                        delete_list = []
                        # Because numerical simulation sometimes produces a certain delay,
                        # At this time, the approximate derivative of the Lyapunov function may have a very small value,
                        # This data needs to be cleared to reduce the impact on the solution of the D function.
                        for ii in range(len(y_label)):
                            if np.abs(y_label[ii])< 2e-3:
                                delete_list.append(ii)
                            if np.abs(y_label[ii])> 1.0:
                                delete_list.append(ii)  
                        y_label = np.delete(y_label,delete_list,0)
                        y_label = torch.from_numpy(y_label).to(self.device)

                        [uavPosNED, uavVelNED, uavAngEular, uavAngRate,
                         delta_uv, timestamp, metric_encoding, metric_encoding_d] = dataset_split(stateData_all[j])
                        metric_encoding = metric_encoding - metric_encoding_d
                        d_feature = np.hstack((uavVelNED[0:dataLength,:],metric_encoding[0:dataLength,:] ))
                        # Delete the index corresponding to the previously mentioned very small value
                        d_feature = np.delete(d_feature,delete_list,0)
                        d_feature = torch.from_numpy(d_feature).to(self.device)

                        d_function = self.DFunction(d_feature.to(torch.float32))
                        loss = criterion(d_function.to(torch.float32),y_label.view(-1,1).to(torch.float32))
                        test_loss=np.append(test_loss,loss.cpu().detach().numpy())
        # DFunction_eval = self.DFunction.eval()

        return model_loss, test_loss

    def train_cwp_controller(
        self,
        train_epoch: int,
        input_features_list: list,
        lyap_dot_label_list: list,
        stateData_all: list,
    ):
        # Use D function to analyze the stability
        self.DFunction.eval()
        # Neural Network Controller
        # ControllerNet = ufunc().to(device)
        self.ControllerNet.train()
        criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(ControllerNet.parameters(),lr =5e-4,betas = (0.9,0.999))

        u_train_loss = np.array([])
        a_list = []

        for epoch in trange(train_epoch):
            for j in range(len(input_features_list)):
                dataLength = len(input_features_list[j])
                [uavPosNED, uavVelNED, uavAngEular, uavAngRate,
                 delta_uv, timestamp, metric_encoding, metric_encoding_d] = dataset_split(stateData_all[j])
                metric_encoding = metric_encoding - metric_encoding_d
                output_acc_NED_tensor = torch.from_numpy(uavVelNED).to(self.device)
                # input of controller
                x_feature = metric_encoding
                u_label = uavVelNED
                u_label = torch.from_numpy(u_label).to(self.device)
                x_feature = torch.from_numpy(x_feature).to(self.device)
                # output of controller
                control_output = self.ControllerNet(x_feature.to(torch.float32))
                # Calculate L1 loss
                lyap_dot = self.ControllerNet(torch.cat((control_output,x_feature),dim=1).to(torch.float32))
                lyap_dot_1 = torch.zeros(dataLength-1)
                for i in range(dataLength-1):
                    lyap_dot_1[i] = lyap_dot[i] / (x_feature[i,:] @ x_feature[i,:]) 
                lyap_dot_max = torch.max(lyap_dot_1)

                loss1 = F.relu(lyap_dot_max)
                # Calculate L2 loss
                loss2 = criterion(control_output.to(torch.float32),u_label.to(torch.float32))
                # L = L1 + L2

                loss = loss1 + 10*loss2

                u_train_loss=np.append(u_train_loss,loss.cpu().detach().numpy())

                self.optimizer_ufunc.zero_grad()
                loss.backward()
                self.optimizer_ufunc.step()
        # ControllerNet_eval = ControllerNet.eval()

        return u_train_loss