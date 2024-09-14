import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jacobian
from typing import Callable, Tuple, Optional, List
from systems_and_functions.control_affine_system import ControlAffineSystem


class InvertedPendulum(ControlAffineSystem):
    N_DIMS = 2
    N_CONTROLS = 1
    __g = 9.8
    
    def __init__(
        self,
        system_params: dict = {'m': 2.0,'L': 1.0, 'b': 0.01},
        controller_params: Optional[dict] = None,
        dt: float = 0.01,
        controller_period: float = 0.01,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        super().__init__(system_params, controller_params, dt, controller_period)
        # system_params{'m': mass,'L': length, 'b': friction}
        self.device = device
        self.m =  self.system_params["m"]
        self.L =  self.system_params["L"]
        self.b =  self.system_params["b"]
        if controller_params is None:
            print('No controller is involved.')
            self.K = np.zeros([self.N_CONTROLS, self.N_DIMS])
        else:
            print('Controller is involved.')
            self.K = controller_params['K']

    # DONE: 建模，正负号
    def _f(self,x: torch.Tensor):
        theta = x[0]
        theta_dot = x[1]
        m, L, b = self.m, self.L, self.b
        g = self.__g
        f = torch.zeros(self.N_DIMS, 1)
        f[0, 0] = theta_dot
        f[1, 0] = (g / L)*torch.sin(theta) - b * theta_dot / (m * L**2) 
        return f
        
    def _g(self,x: torch.Tensor):
        g = torch.zeros(self.N_DIMS, self.N_CONTROLS)
        g = g.type_as(x)
        m, L = self.m, self.L
        g[1, 0] = 1 / (m * L ** 2)
        return g

    
    def x_dot(
            self,
            x: torch.Tensor,
            u: torch.Tensor
    ):
        f = self._f(x).to(self.device).float()
        g = self._g(x).to(self.device).float()
        x_dot = f + g @ u
        return x_dot 


    def linearized_ct_system(
            self,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        goal_point = goal_point.to(self.device)
        u_eq = u_eq.to(self.device)
        dynamics = lambda x: self.x_dot(x, u_eq).squeeze()
        print(dynamics(goal_point))
        A = jacobian(dynamics, goal_point).squeeze().cpu().detach().numpy()
        A = np.reshape(A, (self.N_DIMS, self.N_DIMS))
        B = self._g(goal_point).squeeze().cpu().numpy()
        B = np.reshape(B, (self.N_DIMS, self.N_CONTROLS))
        print("linearized_ct_system:\n A{},\n B{}".format(A, B))
        return A, B

    
    def state_dims(self)->int:
        return InvertedPendulum.N_DIMS
        
 
    def control_dims(self)->int:
        return InvertedPendulum.N_CONTROLS
        
    
    def linearize_and_compute_LQR(self):
        goal_point = torch.Tensor([[0.0],[0.0]]).to(self.device)
        u_eq = torch.Tensor([[0.0]]).to(self.device)
        """
        u_eq should be 
        torch.Tensor([[0.0]])
        instead of 
        torch.Tensor([0.0])
        """
        Act, Bct = self.linearized_ct_system(goal_point, u_eq)
        # Adt, Bdt = self.linearized_dt_system(goal_point, u_eq)
        K_np = self.compute_LQR_controller(Act, Bct)
        self.K = torch.tensor(K_np, dtype=torch.float)
        print('computed LQR controller is {}'.format(K_np))
        # return self.K


    def controller(
        self,
        x: torch.tensor
    )->torch.tensor:
        K = torch.tensor(self.K).type_as(x)
        u = -K@x
        return u

 
    def plot_phase_portrait(
        self, 
        data_sim: torch.Tensor,
        arrow_on: bool = False,
        title = 'inverted pendulum phase portrait'
    ):
        # data_sim = torch.cat((x_sim.unsqueeze(0), x_dot_sim.unsqueeze(0)), dim=0)
        # x_sim = data_sim[0]
        # x_dot_sim = data_sim[1]
        data_sim = data_sim.cpu().detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        x_dot_sim = data_sim[:, 1, :, :]
        plt.figure()
        # data is in the form of torch.Tensor
        num = x_sim.shape[0]
        dim = x_sim.shape[1]
        # state tensors are column vectors as default
        x_value = x_sim[:, 0, 0]
        y_value = x_sim[:, 1, 0]
        plt.plot(x_value, y_value, label='State trajectory')
        if arrow_on is True:
            interval = 200  # 每隔 interval 个点进行绘制
            for i in range(0, num, interval):
                x_dot_x = x_dot_sim[i, 0, 0]  # x 维度上的导数
                x_dot_y = x_dot_sim[i, 1, 0]  # y 维度上的导数
                dx = 2*x_dot_x/((x_dot_x**2+x_dot_y**2)**0.5)
                dy = 2*x_dot_y/((x_dot_x**2+x_dot_y**2)**0.5)
                plt.arrow(x_sim[i, 0, 0], x_sim[i, 1, 0], dx, dy, head_width=3, head_length=4, fc='r', ec='r')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlabel(r"$\mathregular{\theta}$")
        plt.ylabel(r"$\mathregular{\dot{\theta}}$")
        plt.title(title)
        plt.show()


    def plot_phase_portrait_2(
        self, 
        data_sim: torch.Tensor,
        arrow_on: bool = False,
        title = 'inverted pendulum phase portrait'
    ):
        # data_sim = torch.cat((x_sim.unsqueeze(0), x_dot_sim.unsqueeze(0)), dim=0)
        # x_sim = data_sim[0]
        # x_dot_sim = data_sim[1]
        data_sim = data_sim.cpu().detach().numpy()
        x_sim = data_sim[:, 0, :, :]
        x_dot_sim = data_sim[:, 1, :, :]
        plt.figure()
        # data is in the form of torch.Tensor
        num = x_sim.shape[0]
        dim = x_sim.shape[1]
        # state tensors are column vectors as default
        x_value = x_sim[:, 0, 0]
        y_value = x_sim[:, 1, 0]
        plt.scatter(x_value[0], y_value[0], marker='o', color='green')
        plt.plot(x_value, y_value, label='State trajectory')
        
        if arrow_on is True:
            interval = 200  # 每隔 interval 个点进行绘制
            for i in range(0, num, interval):
                x_dot_x = x_dot_sim[i, 0, 0]  # x 维度上的导数
                x_dot_y = x_dot_sim[i, 1, 0]  # y 维度上的导数
                dx = 2*x_dot_x/((x_dot_x**2+x_dot_y**2)**0.5)
                dy = 2*x_dot_y/((x_dot_x**2+x_dot_y**2)**0.5)
                plt.arrow(x_sim[i, 0, 0], x_sim[i, 1, 0], dx, dy, head_width=3, head_length=4, fc='r', ec='r')
        
        plt.scatter(x_value[num-1], y_value[num-1], marker='o', color='red')

        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlabel(r"$\mathregular{\theta}$")
        plt.ylabel(r"$\mathregular{\dot{\theta}}$")
        plt.title(title)
        plt.show()


    # def plot_phase_portrait_3(
    #     self, 
    #     data_sim: torch.Tensor,
    #     data_sim1: torch.Tensor,
    #     arrow_on: bool = False,
    #     title = 'inverted pendulum phase portrait'
    # ):
    #     data_sim = data_sim.cpu().detach().numpy()
    #     x_sim = data_sim[:, 0, :, :]
    #     x_dot_sim = data_sim[:, 1, :, :]
    #     plt.figure()
    #     # data is in the form of torch.Tensor
    #     num = x_sim.shape[0]
    #     dim = x_sim.shape[1]
    #     # state tensors are column vectors as default
    #     x_value = x_sim[:, 0, 0]
    #     y_value = x_sim[:, 1, 0]
    #     plt.scatter(x_value[0], y_value[0], s=30, alpha=0.5, c='blue', marker='o', label='Initial States')
    #     plt.plot(x_value, y_value, label='State trajectory')
        
    #     if arrow_on is True:
    #         interval = 200  # 每隔 interval 个点进行绘制
    #         for i in range(0, num, interval):
    #             x_dot_x = x_dot_sim[i, 0, 0]  # x 维度上的导数
    #             x_dot_y = x_dot_sim[i, 1, 0]  # y 维度上的导数
    #             dx = 2*x_dot_x/((x_dot_x**2+x_dot_y**2)**0.5)
    #             dy = 2*x_dot_y/((x_dot_x**2+x_dot_y**2)**0.5)
    #             plt.arrow(x_sim[i, 0, 0], x_sim[i, 1, 0], dx, dy, head_width=3, head_length=4, fc='r', ec='r',c='c',label='Controller_Dlearning_PPO')
        
    #     plt.scatter(x_value[num-1], y_value[num-1], s=30, alpha=0.8, c='red', marker='*', label='Stable States')

    #     ##########################################################
    #     data_sim1 = data_sim1.cpu().detach().numpy()
    #     x_sim1 = data_sim1[:, 0, :, :]
    #     x_dot_sim1 = data_sim1[:, 1, :, :]
    #     # data is in the form of torch.Tensor
    #     num1 = x_sim1.shape[0]
    #     dim1 = x_sim1.shape[1]
    #     # state tensors are column vectors as default
    #     x_value1 = x_sim1[:, 0, 0]
    #     y_value1 = x_sim1[:, 1, 0]
    #     plt.scatter(x_value1[0], y_value1[0], s=30, alpha=0.5, c='blue', marker='o')
    #     plt.plot(x_value1, y_value1, label='State trajectory')
        
    #     if arrow_on is True:
    #         interval = 200  # 每隔 interval 个点进行绘制
    #         for i in range(0, num1, interval):
    #             x_dot_x1 = x_dot_sim1[i, 0, 0]  # x 维度上的导数
    #             x_dot_y1 = x_dot_sim1[i, 1, 0]  # y 维度上的导数
    #             dx1 = 2*x_dot_x1/((x_dot_x1**2+x_dot_y1**2)**0.5)
    #             dy1 = 2*x_dot_y1/((x_dot_x1**2+x_dot_y1**2)**0.5)
    #             plt.arrow(x_sim1[i, 0, 0], x_sim1[i, 1, 0], dx1, dy1, head_width=3, head_length=4, fc='r', ec='r',c='c',label='Controller_Dlearning_without_PPO')
        
    #     plt.scatter(x_value1[num1-1], y_value1[num1-1], s=30, alpha=0.8, c='red', marker='*')

    #     ax = plt.gca()
    #     ax.set_aspect(1)
    #     plt.xlabel(r"$\mathregular{\theta}$")
    #     plt.ylabel(r"$\mathregular{\dot{\theta}}$")
    #     plt.title(title)
    #     plt.show()